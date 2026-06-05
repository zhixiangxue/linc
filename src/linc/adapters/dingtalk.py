"""DingTalk adapter — Stream SDK for receiving + OpenAPI for proactive sending.

Uses the official `dingtalk-stream` SDK:
  - WebSocket Stream connection for receiving bot messages (ChatbotHandler).
  - SDK-managed access_token + httpx for proactive message sending.

The SDK's `reply_text`/`reply_markdown` methods require `session_webhook` from
an incoming message. For LINC's proactive `send(conv_id, content)` interface,
we use the OpenAPI (oToMessages/batchSend, groupMessages/send) with the
SDK-managed access_token.

conv_id conventions:
  - Single chat: "user:{sender_staff_id}"
  - Group chat: "group:{openConversationId}"
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
import time
from typing import Any
from urllib.parse import quote_plus

import httpx
from pydantic import BaseModel, Field

from ..core.adapter import Adapter, ParsedInbound
from ..core.errors import SendError
from ..core.models import Attachment, Content, Sender
from . import register

log = logging.getLogger(__name__)

DINGTALK_API = "https://api.dingtalk.com"


class DingtalkConfig(BaseModel):
    """DingTalk bot credentials.

    Create a bot at https://open-dev.dingtalk.com:
      - client_id: AppKey of the application
      - client_secret: AppSecret of the application
      - robot_code: Robot code (usually same as client_id)
    """

    client_id: str = Field(..., description="AppKey (client_id)")
    client_secret: str = Field(..., description="AppSecret (client_secret)")
    robot_code: str = Field("", description="Robot code, defaults to client_id if empty")


@register
class DingtalkAdapter(Adapter):
    name = "dingtalk"
    Config = DingtalkConfig

    def __init__(self, config: DingtalkConfig, hub, store) -> None:  # type: ignore[override]
        super().__init__(config, hub, store)
        self._stream_client: Any = None
        self._handler: Any = None
        self._task: asyncio.Task | None = None
        self._background_tasks: set[asyncio.Task] = set()
        self._robot_code: str = ""
        self._stopping = False

    # ----------------------------------------------------------------- lifecycle

    async def start(self) -> None:
        import dingtalk_stream
        from dingtalk_stream import AckMessage

        cfg: DingtalkConfig = self.config  # type: ignore[assignment]
        self._robot_code = cfg.robot_code or cfg.client_id
        self._stopping = False

        credential = dingtalk_stream.Credential(cfg.client_id, cfg.client_secret)
        self._stream_client = dingtalk_stream.DingTalkStreamClient(credential)

        # Create a handler that bridges to our on_event
        adapter = self

        class LincChatbotHandler(dingtalk_stream.ChatbotHandler):
            async def process(self, callback: dingtalk_stream.CallbackMessage):
                """Called by the SDK in its async context."""
                try:
                    incoming = dingtalk_stream.ChatbotMessage.from_dict(callback.data)
                    raw = {
                        "sender_staff_id": incoming.sender_staff_id or "",
                        "sender_nick": incoming.sender_nick or "",
                        "sender_id": incoming.sender_id or "",
                        "conversation_id": incoming.conversation_id or "",
                        "conversation_type": incoming.conversation_type or "",
                        "message_id": incoming.message_id or "",
                        "text": incoming.text.content if incoming.text else "",
                        "message_type": incoming.message_type or "",
                        "robot_code": incoming.robot_code or "",
                        "session_webhook": incoming.session_webhook or "",
                    }
                    # Extract image download codes for picture / richText messages
                    if incoming.message_type == "picture":
                        raw["_image_codes"] = incoming.get_image_list() or []
                    elif incoming.message_type == "richText":
                        raw["_image_codes"] = incoming.get_image_list() or []
                        # Also collect text fragments from richText
                        raw["_rich_texts"] = incoming.get_text_list() or []
                    await adapter.on_event(raw)
                except Exception:
                    log.exception("dingtalk adapter: process callback failed")
                return AckMessage.STATUS_OK, "OK"

        self._handler = LincChatbotHandler()
        self._stream_client.register_callback_handler(
            dingtalk_stream.chatbot.ChatbotMessage.TOPIC,
            self._handler,
        )

        # Start stream in background task
        self._task = asyncio.create_task(self._run_stream())
        log.info("dingtalk adapter: stream connected")

    async def _run_stream(self) -> None:
        """Run a cancellable DingTalk stream loop.

        The official SDK's start() catches CancelledError and keeps reconnecting,
        so LINC owns the loop here while still using SDK open/route/keepalive APIs.
        """
        import websockets

        client = self._stream_client
        if client is None:
            return

        client.pre_start()
        while not self._stopping:
            try:
                connection = await asyncio.to_thread(client.open_connection)
                if self._stopping:
                    break
                if not connection:
                    log.error("dingtalk adapter: open connection failed")
                    await self._sleep_or_stop(10)
                    continue

                client.logger.info("endpoint is %s", connection)
                uri = f'{connection["endpoint"]}?ticket={quote_plus(connection["ticket"])}'
                async with _dingtalk_ws_connect(websockets, uri) as websocket:
                    client.websocket = websocket
                    keepalive_task = asyncio.create_task(client.keepalive(websocket))
                    self._background_tasks.add(keepalive_task)
                    keepalive_task.add_done_callback(self._background_tasks.discard)
                    try:
                        async for raw_message in websocket:
                            if self._stopping:
                                break
                            task = asyncio.create_task(client.background_task(json.loads(raw_message)))
                            self._background_tasks.add(task)
                            task.add_done_callback(self._background_tasks.discard)
                    finally:
                        keepalive_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError, Exception):
                            await keepalive_task
                        client.websocket = None
            except asyncio.CancelledError:
                break
            except websockets.exceptions.ConnectionClosedError as e:
                if self._stopping:
                    break
                log.error("dingtalk adapter: stream network exception: %s", e)
                await self._sleep_or_stop(10)
            except Exception:
                if self._stopping:
                    break
                log.exception("dingtalk adapter: stream client crashed")
                await self._sleep_or_stop(3)

    async def _sleep_or_stop(self, seconds: float) -> None:
        end = asyncio.get_running_loop().time() + seconds
        while not self._stopping:
            remaining = end - asyncio.get_running_loop().time()
            if remaining <= 0:
                return
            await asyncio.sleep(min(0.2, remaining))

    async def stop(self) -> None:
        self._stopping = True
        client = self._stream_client
        if client is not None:
            websocket = getattr(client, "websocket", None)
            if websocket is not None:
                with contextlib.suppress(Exception):
                    await websocket.close()
        if self._task is not None:
            self._task.cancel()
            try:
                await asyncio.wait_for(self._task, timeout=5)
            except (asyncio.CancelledError, TimeoutError):
                pass
            except Exception:
                log.debug("dingtalk adapter: stream task shutdown failed", exc_info=True)
            self._task = None
        for task in list(self._background_tasks):
            task.cancel()
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()
        self._stream_client = None
        self._handler = None
        log.info("dingtalk adapter: stream disconnected")

    # ----------------------------------------------------------------- inbound

    async def on_event(self, raw: dict[str, Any]) -> int | None:
        """Override to download DingTalk image attachments before persisting."""
        parsed = self.parse_inbound(raw)
        if parsed is None:
            return None

        content = parsed.content
        # Download images referenced by download_code
        if content.attachments:
            attachments_dir = (self.store.db_path.parent / "attachments").resolve()
            attachments_dir.mkdir(parents=True, exist_ok=True)
            new_attachments: list[Attachment] = []
            for att in content.attachments:
                att_dict = att.model_dump()
                download_code = att.meta.get("download_code", "") if att.meta else ""
                if download_code:
                    try:
                        download_url = await self._get_image_download_url(download_code)
                        if download_url:
                            async with httpx.AsyncClient() as client:
                                resp = await client.get(download_url, timeout=30)
                                resp.raise_for_status()
                                data = resp.content
                            safe_name = att.file_id or download_code[:12]
                            local_path = attachments_dir / f"{safe_name}.png"
                            local_path.write_bytes(data)
                            att_dict["path"] = str(local_path)
                            att_dict["mime"] = "image/png"
                            log.debug(
                                "dingtalk: downloaded image %s -> %s (%d bytes)",
                                download_code[:12], local_path, len(data),
                            )
                    except Exception:
                        log.exception("dingtalk: image download failed for %s", download_code[:12])
                new_attachments.append(Attachment(**att_dict))
            content = Content(text=content.text, attachments=new_attachments)

        row_id = await self.store.insert_inbound(
            platform=self.name,
            conv_id=parsed.conv_id,
            msg_id=parsed.msg_id,
            ts=parsed.ts,
            sender=parsed.sender,
            content=content,
            raw=raw,
        )
        if row_id is not None:
            text_preview = (content.text or "")[:80]
            log.info(
                "\u2b07 [%s] %s | %s (%s): %s",
                self.name, parsed.conv_id,
                parsed.sender.name or parsed.sender.id, parsed.sender.id,
                text_preview,
            )
        return row_id

    async def _get_image_download_url(self, download_code: str) -> str | None:
        """Call DingTalk OpenAPI to exchange download_code for a temporary download URL."""
        if self._stream_client is None:
            return None
        access_token = self._stream_client.get_access_token()
        if not access_token:
            log.error("dingtalk: cannot get access_token for image download")
            return None
        url = f"{DINGTALK_API}/v1.0/robot/messageFiles/download"
        headers = {
            "x-acs-dingtalk-access-token": access_token,
            "Content-Type": "application/json",
        }
        payload = {
            "robotCode": self._robot_code,
            "downloadCode": download_code,
        }
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(url, headers=headers, json=payload, timeout=15)
                resp.raise_for_status()
                return resp.json().get("downloadUrl", "")
        except Exception:
            log.exception("dingtalk: get_image_download_url failed")
            return None

    def parse_inbound(self, raw: dict[str, Any]) -> ParsedInbound | None:
        msg_type = raw.get("message_type", "")

        # Build text and attachments based on message type
        text = ""
        attachments: list[Attachment] = []

        if msg_type == "text":
            text = raw.get("text", "")
            if not text:
                return None
        elif msg_type == "picture":
            text = "[图片]"
            for code in raw.get("_image_codes", []):
                attachments.append(Attachment(
                    kind="image",
                    file_id=raw.get("message_id", ""),
                    meta={"download_code": code},
                ))
        elif msg_type == "richText":
            # Combine text fragments from richText
            rich_texts = raw.get("_rich_texts", [])
            text = "\n".join(rich_texts) if rich_texts else ""
            for code in raw.get("_image_codes", []):
                attachments.append(Attachment(
                    kind="image",
                    file_id=raw.get("message_id", ""),
                    meta={"download_code": code},
                ))
            if not text and not attachments:
                return None
        else:
            # Unsupported message type
            if msg_type:
                log.debug("dingtalk: skipping unsupported msgtype=%s", msg_type)
            return None

        conversation_type = raw.get("conversation_type", "")
        sender_staff_id = raw.get("sender_staff_id", "")
        conversation_id = raw.get("conversation_id", "")

        # Build conv_id with type prefix
        if conversation_type == "1":
            # Single chat: use user ID as conv_id
            conv_id = f"user:{sender_staff_id}"
        elif conversation_type == "2":
            # Group chat: use conversation ID
            conv_id = f"group:{conversation_id}"
        else:
            return None

        if not conv_id:
            return None

        msg_id = raw.get("message_id", "")
        sender = Sender(
            id=sender_staff_id,
            name=raw.get("sender_nick", "") or sender_staff_id,
        )
        content = Content(text=text, attachments=attachments)

        return ParsedInbound(
            conv_id=conv_id,
            msg_id=msg_id,
            ts=time.time(),
            sender=sender,
            content=content,
        )

    # ----------------------------------------------------------------- outbound

    async def _upload_media(self, file_path: str, media_type: str = "image") -> str:
        """Upload a local file to DingTalk and return the media_id.

        Uses the SDK's built-in upload_to_dingtalk() method which calls the
        oapi.dingtalk.com/media/upload endpoint. Returns a media_id in the
        format '@lADP...' suitable for sampleImageMsg's photoURL field.
        """
        if self._stream_client is None:
            raise SendError("dingtalk adapter not started; cannot upload media")

        from pathlib import Path
        import mimetypes

        path = Path(file_path)
        if not path.is_file():
            raise SendError(f"dingtalk adapter: file not found: {file_path}")

        file_bytes = path.read_bytes()
        filename = path.name
        mimetype = mimetypes.guess_type(filename)[0] or "application/octet-stream"

        # SDK's upload_to_dingtalk is synchronous; run in thread pool
        media_id = await asyncio.to_thread(
            self._stream_client.upload_to_dingtalk,
            file_bytes,
            filetype=media_type,
            filename=filename,
            mimetype=mimetype,
        )

        if not media_id:
            raise SendError(f"dingtalk adapter: upload_to_dingtalk returned no media_id")
        log.debug("dingtalk: uploaded %s -> media_id=%s", filename, media_id[:20])
        return media_id

    async def _send_single(
        self,
        conv_id: str,
        msg_key: str,
        msg_param: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        """Low-level helper: send one robot message to the given conv_id."""
        if self._stream_client is None:
            raise SendError("dingtalk adapter not started; call start() first")

        access_token = self._stream_client.get_access_token()
        if not access_token:
            raise SendError("dingtalk adapter: failed to get access_token")

        headers = {
            "x-acs-dingtalk-access-token": access_token,
            "Content-Type": "application/json",
        }

        if conv_id.startswith("user:"):
            user_id = conv_id[5:]
            url = f"{DINGTALK_API}/v1.0/robot/oToMessages/batchSend"
            payload = {
                "robotCode": self._robot_code,
                "userIds": [user_id],
                "msgKey": msg_key,
                "msgParam": json.dumps(msg_param, ensure_ascii=False),
            }
        elif conv_id.startswith("group:"):
            group_id = conv_id[6:]
            url = f"{DINGTALK_API}/v1.0/robot/groupMessages/send"
            payload = {
                "robotCode": self._robot_code,
                "openConversationId": group_id,
                "msgKey": msg_key,
                "msgParam": json.dumps(msg_param, ensure_ascii=False),
            }
        else:
            raise SendError(f"dingtalk adapter: invalid conv_id format: {conv_id!r}")

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as e:
            raise SendError(
                f"dingtalk send HTTP {e.response.status_code}: {e.response.text}"
            ) from e
        except Exception as e:
            raise SendError(f"dingtalk send failed: {e}") from e

        msg_id = data.get("processQueryKey", "") or data.get("messageId", "")
        return str(msg_id), data

    async def _resolve_file_ref(self, att: "Attachment") -> tuple[str, str]:
        """Resolve a non-image file attachment to (media_id, filename).

        Downloads from URL if needed, then uploads to DingTalk.
        """
        import os
        import tempfile

        filename = att.name or "file"
        if att.path:
            from pathlib import Path
            filename = att.name or Path(att.path).name
            media_id = await self._upload_media(att.path, media_type="file")
            return media_id, filename
        elif att.url:
            async with httpx.AsyncClient() as client:
                resp = await client.get(att.url, timeout=60)
                resp.raise_for_status()
                file_data = resp.content
            suffix = att.ext or ""
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(file_data)
                tmp_path = tmp.name
            try:
                media_id = await self._upload_media(tmp_path, media_type="file")
                return media_id, filename
            finally:
                os.unlink(tmp_path)
        return "", filename

    async def _resolve_image_ref(self, att: "Attachment") -> str:
        """Resolve an image attachment to a markdown-embeddable reference.

        Returns a media_id (e.g. '@lADP...') for local files, or the original
        URL for network images. DingTalk markdown supports both:
          ![image](@lADPxxx)   — uploaded media_id
          ![image](https://…)  — public URL
        """
        import os
        import tempfile

        if att.path:
            return await self._upload_media(att.path, media_type="image")
        elif att.url:
            # Download from URL to a temp file, then upload
            async with httpx.AsyncClient() as client:
                resp = await client.get(att.url, timeout=30)
                resp.raise_for_status()
                img_data = resp.content
            suffix = att.ext or ".png"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(img_data)
                tmp_path = tmp.name
            try:
                return await self._upload_media(tmp_path, media_type="image")
            finally:
                os.unlink(tmp_path)
        return ""

    async def send(
        self,
        conv_id: str,
        content: Content,
    ) -> tuple[str, dict[str, Any]]:
        if self._stream_client is None:
            raise SendError("dingtalk adapter not started; call start() first")

        text = content.text or ""
        images = [att for att in content.attachments if att.is_image]
        files = [att for att in content.attachments if not att.is_image]

        # Resolve all image attachments to embeddable references (media_id or URL)
        image_refs: list[str] = []
        for att in images:
            try:
                ref = await self._resolve_image_ref(att)
                if ref:
                    image_refs.append(ref)
                else:
                    log.warning("dingtalk: image attachment has no path or url, skipping")
            except Exception as e:
                log.error("dingtalk: failed to resolve image: %s", e)

        # Build a single markdown body with text + embedded images
        parts: list[str] = []
        if text:
            parts.append(text)
        for ref in image_refs:
            parts.append(f"![image]({ref})")

        markdown_body = "\n\n".join(parts)

        last_msg_id = ""
        last_data: dict[str, Any] = {}

        # Send text + images as one unified markdown message
        if markdown_body:
            last_msg_id, last_data = await self._send_single(
                conv_id,
                msg_key="sampleMarkdown",
                msg_param={"title": "消息", "text": markdown_body},
            )

        # Send each file attachment as a separate sampleFile message
        for att in files:
            try:
                media_id, filename = await self._resolve_file_ref(att)
                if media_id:
                    last_msg_id, last_data = await self._send_single(
                        conv_id,
                        msg_key="sampleFile",
                        msg_param={"mediaId": media_id, "fileName": filename, "fileType": att.ext.lstrip(".") if att.ext else ""},
                    )
            except Exception as e:
                log.error("dingtalk: failed to send file attachment: %s", e)

        # Fallback: empty content
        if not markdown_body and not files:
            last_msg_id, last_data = await self._send_single(
                conv_id,
                msg_key="sampleMarkdown",
                msg_param={"title": "消息", "text": ""},
            )

        return last_msg_id, last_data


def _dingtalk_ws_connect(websockets_module: Any, uri: str) -> Any:
    kwargs: dict[str, Any] = {}
    params = inspect.signature(websockets_module.connect).parameters
    if "proxy" in params:
        kwargs["proxy"] = None
    if "open_timeout" in params:
        kwargs["open_timeout"] = 30
    return websockets_module.connect(uri, **kwargs)


def _json_escape(s: str) -> str:
    """JSON-encode a string value (with surrounding quotes)."""
    import json
    return json.dumps(s)
