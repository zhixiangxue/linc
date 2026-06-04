"""Feishu (Lark) adapter — WebSocket event subscription + REST API sending.

Uses the official `lark-oapi` SDK:
  - WebSocket long connection for receiving events (no public webhook needed).
  - REST API `im.v1.message.create` for sending messages.

conv_id conventions:
  - Group chat: chat_id (starts with "oc_")
  - P2P (single) chat: sender's open_id (starts with "ou_")
  The adapter detects the type from the prefix when sending.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from typing import Any

from pydantic import BaseModel, Field

from ..core.adapter import Adapter, ParsedInbound
from ..core.errors import SendError
from ..core.models import Attachment, Content, Sender
from . import register

log = logging.getLogger(__name__)


def _guess_ext(kind: str, name: str | None) -> str:
    """Guess file extension for a downloaded resource."""
    if name:
        from pathlib import PurePosixPath
        ext = PurePosixPath(name).suffix
        if ext:
            return ext
    # Fallback by kind
    return {"image": ".png", "audio": ".ogg", "video": ".mp4", "file": ".bin"}.get(kind, ".bin")


class FeishuConfig(BaseModel):
    """Feishu app credentials.

    Create an app at https://open.feishu.cn/app and enable:
      - Event subscription (WebSocket mode)
      - im:message, im:message:send_as_bot permissions
    """

    app_id: str = Field(..., description="App ID from Feishu developer console")
    app_secret: str = Field(..., description="App Secret from Feishu developer console")

def _patch_lark_ws_client(ws: Any, adapter: "FeishuAdapter") -> None:
    original_connect = ws._connect
    original_receive_loop = ws._receive_message_loop

    async def patched_connect() -> None:
        import lark_oapi.ws.client as lark_ws_client

        real_create_task = lark_ws_client.loop.create_task

        def create_task(coro: Any) -> asyncio.Task:
            task = real_create_task(coro)
            code = getattr(coro, "cr_code", None)
            if getattr(code, "co_name", "") == "_receive_message_loop":
                adapter._receive_task = task
            return task

        lark_ws_client.loop.create_task = create_task
        try:
            await original_connect()
        finally:
            lark_ws_client.loop.create_task = real_create_task

    async def patched_receive_loop() -> None:
        try:
            await original_receive_loop()
        except Exception as exc:
            if not getattr(ws, "_auto_reconnect", True):
                log.debug("feishu adapter: receive loop stopped: %s", exc)
                return
            raise

    ws._connect = patched_connect
    ws._receive_message_loop = patched_receive_loop


async def _cancel_task(task: asyncio.Task | None, name: str) -> None:
    if task is None or task.done():
        return
    task.cancel()
    try:
        await asyncio.wait_for(task, timeout=2)
    except (asyncio.CancelledError, TimeoutError):
        pass
    except Exception:
        log.debug("feishu adapter: %s task shutdown failed", name, exc_info=True)


@register
class FeishuAdapter(Adapter):
    name = "feishu"
    Config = FeishuConfig

    def __init__(self, config: FeishuConfig, hub, store) -> None:  # type: ignore[override]
        super().__init__(config, hub, store)
        self._lark_client: Any = None  # lark.Client for REST API
        self._ws_client: Any = None  # lark_oapi.ws.Client for WebSocket
        self._ping_task: asyncio.Task | None = None
        self._receive_task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._user_cache: dict[str, str] = {}  # open_id -> display name

    # ----------------------------------------------------------------- lifecycle

    async def start(self) -> None:
        import lark_oapi as lark

        cfg: FeishuConfig = self.config  # type: ignore[assignment]

        # REST client for sending messages
        self._lark_client = (
            lark.Client.builder()
            .app_id(cfg.app_id)
            .app_secret(cfg.app_secret)
            .log_level(lark.LogLevel.WARNING)
            .build()
        )

        # Event dispatcher: subscribe to im.message.receive_v1
        handler = (
            lark.EventDispatcherHandler.builder("", "")
            .register_p2_im_message_receive_v1(self._on_message_event)
            .build()
        )

        # WebSocket client. Do NOT call `start()` here: the official SDK's
        # `start()` is a blocking convenience entry point. LINC already owns an
        # asyncio loop, so we use the SDK's async connection primitives directly.
        from lark_oapi.ws import Client as LarkWsClient
        import lark_oapi.ws.client as lark_ws_client

        self._loop = asyncio.get_running_loop()
        lark_ws_client.loop = self._loop
        self._ws_client = LarkWsClient(
            cfg.app_id,
            cfg.app_secret,
            event_handler=handler,
            log_level=lark.LogLevel.WARNING,
        )
        _patch_lark_ws_client(self._ws_client, self)
        await self._ws_client._connect()
        self._ping_task = asyncio.create_task(self._ws_client._ping_loop())
        log.info(
            "feishu adapter: websocket connected conn_id=%s subscribed=im.message.receive_v1",
            getattr(self._ws_client, "_conn_id", ""),
        )

    async def stop(self) -> None:
        ws = self._ws_client
        ping_task = self._ping_task
        receive_task = self._receive_task
        self._ws_client = None
        self._lark_client = None
        self._ping_task = None
        self._receive_task = None
        self._loop = None
        if ws is not None:
            ws._auto_reconnect = False
            with contextlib.suppress(Exception):
                await asyncio.wait_for(ws._disconnect(), timeout=2)
        await _cancel_task(ping_task, "ping")
        await _cancel_task(receive_task, "receive")
        log.info("feishu adapter: websocket disconnected")

    # ----------------------------------------------------------------- inbound

    def _on_message_event(self, event: Any) -> None:
        """Callback from lark-oapi WebSocket event dispatcher."""
        try:
            # event is P2ImMessageReceiveV1 with event.message, event.sender etc.
            msg = event.event.message
            sender_info = event.event.sender

            raw = {
                "message_id": msg.message_id,
                "chat_id": msg.chat_id,
                "chat_type": msg.chat_type,  # "p2p" or "group"
                "message_type": msg.message_type,
                "content": msg.content,  # JSON string
                "create_time": msg.create_time,
                "sender_id": sender_info.sender_id.open_id if sender_info.sender_id else "",
                "sender_type": sender_info.sender_type,  # "user" or "app"
            }
            log.info(
                "feishu adapter: event received chat=%s type=%s sender=%s sender_type=%s msg_type=%s",
                raw["chat_id"],
                raw["chat_type"],
                raw["sender_id"],
                raw["sender_type"],
                raw["message_type"],
            )

            # Schedule on_event in LINC's main asyncio loop.
            if self._loop is not None and self._loop.is_running():
                task = self._loop.create_task(self.on_event(raw))
                task.add_done_callback(self._task_done_cb)
            else:
                log.warning("feishu adapter: event loop not running, dropping message")
        except Exception:
            log.exception("feishu adapter: _on_message_event failed")

    async def _resolve_user_name(self, user_id: str) -> str:
        """Resolve Feishu open_id to a readable user name, with in-memory cache."""
        if not user_id:
            return user_id
        if user_id in self._user_cache:
            return self._user_cache[user_id]
        if self._lark_client is None:
            return user_id

        from lark_oapi.api.contact.v3 import GetUserRequest

        request = (
            GetUserRequest.builder()
            .user_id_type("open_id")
            .user_id(user_id)
            .build()
        )
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self._lark_client.contact.v3.user.get, request
                ),
                timeout=10.0,
            )
            if not response.success():
                log.warning(
                    "feishu adapter: contact.v3.user.get failed for %s: code=%s msg=%s",
                    user_id,
                    response.code,
                    response.msg,
                )
                self._user_cache[user_id] = user_id
                return user_id

            user = response.data.user if response.data else None
            name = (
                getattr(user, "nickname", None)
                or getattr(user, "name", None)
                or getattr(user, "en_name", None)
                or user_id
            )
            log.debug(
                "feishu user resolve: %s -> %s (nickname=%r, name=%r, en_name=%r)",
                user_id,
                name,
                getattr(user, "nickname", None),
                getattr(user, "name", None),
                getattr(user, "en_name", None),
            )
            self._user_cache[user_id] = name
            return name
        except Exception as e:
            log.warning("feishu adapter: contact.v3.user.get raised for %s: %s", user_id, e)
            self._user_cache[user_id] = user_id
            return user_id

    @staticmethod
    def _task_done_cb(task: asyncio.Task) -> None:
        """Log unhandled exceptions from on_event tasks."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            log.error("feishu on_event task failed: %s", exc, exc_info=exc)

    async def on_event(self, raw: dict[str, Any]) -> int | None:
        """Override to resolve user name and download media before persisting."""
        log.info("feishu on_event: processing msg_type=%s msg_id=%s", raw.get("message_type"), raw.get("message_id", "")[:8])
        parsed = self.parse_inbound(raw)
        if parsed is None:
            log.warning("feishu on_event: parse_inbound returned None for msg_type=%s", raw.get("message_type"))
            return None
        log.info("feishu on_event: parsed ok, text=%r attachments=%d", (parsed.content.text or "")[:40], len(parsed.content.attachments))
    
        # Download media attachments
        content = parsed.content
        if content.attachments and self._lark_client is not None:
            attachments_dir = (self.store.db_path.parent / "attachments").resolve()
            attachments_dir.mkdir(parents=True, exist_ok=True)
            message_id = raw.get("message_id", "")
            new_attachments: list[Attachment] = []
            for att in content.attachments:
                att_dict = att.model_dump()
                file_key = att.meta.get("file_key", "") if att.meta else ""
                if file_key and message_id:
                    try:
                        resource_type = "image" if att.kind == "image" else "file"
                        data = await self._download_resource(
                            message_id, file_key, resource_type
                        )
                        if data:
                            ext = _guess_ext(att.kind, att.name)
                            safe_name = f"{message_id}_{file_key[:8]}{ext}"
                            local_path = attachments_dir / safe_name
                            local_path.write_bytes(data)
                            att_dict["path"] = str(local_path)
                            if not att_dict.get("mime"):
                                import mimetypes
                                mime, _ = mimetypes.guess_type(safe_name)
                                att_dict["mime"] = mime
                            log.debug(
                                "feishu: downloaded %s %s -> %s (%d bytes)",
                                resource_type, file_key[:8], local_path, len(data),
                            )
                    except Exception:
                        log.exception("feishu: download failed for %s", file_key[:8])
                new_attachments.append(Attachment(**att_dict))
            content = Content(text=content.text, attachments=new_attachments)
    
        display_name = await self._resolve_user_name(parsed.sender.id)
        sender = Sender(id=parsed.sender.id, name=display_name)
        row_id = await self.store.insert_inbound(
            platform=self.name,
            conv_id=parsed.conv_id,
            msg_id=parsed.msg_id,
            ts=parsed.ts,
            sender=sender,
            content=content,
            raw=raw,
        )
        if row_id is not None:
            text_preview = (content.text or "")[:80]
            log.info(
                "\u2b07 [%s] %s | %s (%s): %s",
                self.name,
                parsed.conv_id,
                display_name,
                parsed.sender.id,
                text_preview,
            )
        return row_id
    
    async def _download_resource(
        self, message_id: str, file_key: str, resource_type: str
    ) -> bytes | None:
        """Download a message resource using the Feishu REST API."""
        from lark_oapi.api.im.v1.model.get_message_resource_request import (
            GetMessageResourceRequest,
        )
    
        req = (
            GetMessageResourceRequest.builder()
            .message_id(message_id)
            .file_key(file_key)
            .type(resource_type)
            .build()
        )
        try:
            log.info("feishu: downloading resource msg_id=%s file_key=%s type=%s", message_id[:8], file_key[:8], resource_type)
            resp = await asyncio.wait_for(
                asyncio.to_thread(
                    self._lark_client.im.v1.message_resource.get, req
                ),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            log.error("feishu: download_resource timed out (30s): msg_id=%s file_key=%s", message_id[:8], file_key[:8])
            return None
        except Exception:
            log.exception("feishu: download_resource request failed")
            return None
        code = getattr(resp, "code", None)
        if code is not None and code != 0:
            log.warning(
                "feishu: download_resource failed: msg_id=%s file_key=%s code=%s msg=%s",
                message_id, file_key, code, getattr(resp, "msg", ""),
            )
            return None
        f = getattr(resp, "file", None)
        if hasattr(f, "read"):
            return f.read()
        if isinstance(f, (bytes, bytearray)):
            return bytes(f)
        return None
    
    def parse_inbound(self, raw: dict[str, Any]) -> ParsedInbound | None:
        # Filter out bot's own messages
        if raw.get("sender_type") == "app":
            return None
    
        import json as json_mod
    
        message_type = raw.get("message_type", "")
        content_str = raw.get("content") or ""
    
        text = ""
        attachments: list[Attachment] = []
    
        if not content_str:
            return None
    
        try:
            content_data = json_mod.loads(content_str)
        except (json_mod.JSONDecodeError, TypeError):
            return None
    
        if message_type == "text":
            text = content_data.get("text", "")
            if not text:
                return None
    
        elif message_type == "image":
            image_key = content_data.get("image_key", "")
            text = "[图片]"
            if image_key:
                attachments.append(Attachment(
                    kind="image",
                    file_id=raw.get("message_id", ""),
                    meta={"file_key": image_key},
                ))
    
        elif message_type == "file":
            file_key = content_data.get("file_key", "")
            file_name = content_data.get("file_name", "")
            text = f"[文件: {file_name}]" if file_name else "[文件]"
            if file_key:
                import mimetypes
                mime, _ = mimetypes.guess_type(file_name)
                attachments.append(Attachment(
                    kind="file",
                    name=file_name,
                    mime=mime,
                    file_id=raw.get("message_id", ""),
                    meta={"file_key": file_key},
                ))
    
        elif message_type == "audio":
            file_key = content_data.get("file_key", "")
            text = "[语音]"
            if file_key:
                attachments.append(Attachment(
                    kind="audio",
                    file_id=raw.get("message_id", ""),
                    meta={"file_key": file_key},
                ))
    
        elif message_type == "post":
            # Rich text (post) messages - may be either:
            # 1. Direct: {"title": "...", "content": [[...]]}
            # 2. Lang-wrapped: {"zh_cn": {"title": "...", "content": [[...]]}}
            texts_parts: list[str] = []
            post_body = None
            if "content" in content_data:
                # Direct structure
                post_body = content_data
            else:
                # Language-wrapped structure
                for lang_key in ("zh_cn", "en_us", "ja_jp"):
                    post_body = content_data.get(lang_key)
                    if post_body:
                        break
            if post_body:
                title = post_body.get("title", "")
                if title:
                    texts_parts.append(title)
                for paragraph in post_body.get("content", []):
                    for element in paragraph:
                        tag = element.get("tag", "")
                        if tag == "text":
                            t = element.get("text", "").strip()
                            if t:
                                texts_parts.append(t)
                        elif tag == "img":
                            image_key = element.get("image_key", "")
                            if image_key:
                                attachments.append(Attachment(
                                    kind="image",
                                    file_id=raw.get("message_id", ""),
                                    meta={"file_key": image_key},
                                ))
                        elif tag == "a":
                            link_text = element.get("text", "")
                            href = element.get("href", "")
                            if link_text:
                                texts_parts.append(f"{link_text}({href})" if href else link_text)
            text = "\n".join(texts_parts) if texts_parts else ""
            if not text and not attachments:
                return None
    
        else:
            # Unsupported message type
            if message_type:
                log.debug("feishu: skipping unsupported message_type=%s", message_type)
            return None
    
        # Determine conv_id
        chat_type = raw.get("chat_type", "")
        if chat_type == "group":
            conv_id = raw.get("chat_id", "")
        else:
            # P2P: use sender's open_id as conv_id
            conv_id = raw.get("sender_id", "")
    
        if not conv_id:
            return None
    
        msg_id = raw.get("message_id", "")
        sender_id = raw.get("sender_id", "")
    
        # Timestamp: create_time is milliseconds string
        try:
            ts = float(raw.get("create_time", "0")) / 1000.0
        except (TypeError, ValueError):
            ts = time.time()
    
        sender = Sender(id=sender_id, name=sender_id)
        content = Content(text=text, attachments=attachments)
        return ParsedInbound(
            conv_id=conv_id,
            msg_id=msg_id,
            ts=ts,
            sender=sender,
            content=content,
        )

    # ----------------------------------------------------------------- outbound

    async def send(
        self,
        conv_id: str,
        content: Content,
    ) -> tuple[str, dict[str, Any]]:
        if self._lark_client is None:
            raise SendError("feishu adapter not started; call start() first")

        import json as json_mod
        from lark_oapi.api.im.v1 import (
            CreateMessageRequest,
            CreateMessageRequestBody,
        )

        # Determine receive_id_type from conv_id prefix
        if conv_id.startswith("oc_"):
            receive_id_type = "chat_id"
        elif conv_id.startswith("ou_"):
            receive_id_type = "open_id"
        else:
            # Fallback: try as chat_id
            receive_id_type = "chat_id"

        text = content.text or ""
        # Use interactive card with markdown element for rich rendering
        msg_content = json_mod.dumps({
            "elements": [
                {
                    "tag": "markdown",
                    "content": text,
                }
            ],
        }, ensure_ascii=False)

        request = (
            CreateMessageRequest.builder()
            .receive_id_type(receive_id_type)
            .request_body(
                CreateMessageRequestBody.builder()
                .receive_id(conv_id)
                .content(msg_content)
                .msg_type("interactive")
                .uuid(str(uuid.uuid4()))
                .build()
            )
            .build()
        )

        try:
            response = await asyncio.to_thread(
                self._lark_client.im.v1.message.create, request
            )
        except Exception as e:
            raise SendError(f"feishu im.v1.message.create raised: {e}") from e

        if not response.success():
            raise SendError(
                f"feishu send failed: code={response.code}, msg={response.msg}"
            )

        resp_data = {
            "code": response.code,
            "msg": response.msg,
            "message_id": response.data.message_id if response.data else "",
        }
        platform_msg_id = resp_data.get("message_id", "")
        return str(platform_msg_id), resp_data
