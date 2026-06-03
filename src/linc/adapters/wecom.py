"""WeCom (企业微信) AI Bot adapter — WebSocket long connection mode.

Uses the official `wecom-aibot-sdk` (WecomTeam/wecom-aibot-python-sdk):
  - WebSocket persistent connection for receiving messages and events.
  - `send_message(chatid, body)` for proactive message sending.
  - Event-driven architecture via pyee (AsyncIOEventEmitter).

conv_id conventions:
  - Single chat: userid of the sender (from.userid)
  - Group chat: chatid of the group
  Both are directly usable as the `chatid` parameter in send_message.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import time
from typing import Any

from pydantic import BaseModel, Field

from ..core.adapter import Adapter, ParsedInbound
from ..core.errors import SendError
from ..core.models import Attachment, Content, Sender
from . import register

log = logging.getLogger(__name__)


class _WecomSdkLogger:
    """Bridge WeCom SDK logs into LINC logging and suppress noisy debug heartbeats."""

    def debug(self, message: str, *args: object) -> None:
        if message.startswith("Heartbeat ") or message.startswith("Received heartbeat ack"):
            return
        log.debug("wecom sdk: " + message, *args)

    def info(self, message: str, *args: object) -> None:
        log.info("wecom sdk: " + message, *args)

    def warn(self, message: str, *args: object) -> None:
        log.warning("wecom sdk: " + message, *args)

    def error(self, message: str, *args: object) -> None:
        log.error("wecom sdk: " + message, *args)


class WecomConfig(BaseModel):
    """WeCom AI Bot credentials.

    Create an AI Bot at https://developer.work.weixin.qq.com:
      - bot_id: The bot ID from the WeCom admin console
      - secret: The bot secret for authentication
    """

    bot_id: str = Field(..., description="Bot ID from WeCom developer console")
    secret: str = Field(..., description="Bot secret for WebSocket authentication")

def _patch_wecom_websocket_connect(wecom_ws: Any) -> None:
    """Make the official WeCom SDK work reliably with websockets>=15.

    `websockets.connect()` defaults to `proxy=True` since websockets 15. The
    current official WeCom SDK does not pass `proxy=None`, so local HTTP proxy
    environment variables may be applied on top of a system/TUN proxy and cause
    opening-handshake timeouts. Mirror the Feishu SDK's compatibility behavior.
    """
    connect = wecom_ws.websockets.connect
    if getattr(connect, "_linc_patched", False):
        return

    params = inspect.signature(connect).parameters
    supports_proxy = "proxy" in params
    supports_open_timeout = "open_timeout" in params

    def patched_connect(*args: Any, **kwargs: Any) -> Any:
        if supports_proxy:
            kwargs.setdefault("proxy", None)
        if supports_open_timeout:
            kwargs.setdefault("open_timeout", 30)
        return connect(*args, **kwargs)

    patched_connect._linc_patched = True  # type: ignore[attr-defined]
    wecom_ws.websockets.connect = patched_connect

async def _disconnect_wecom_client(client: Any) -> None:
    manager = getattr(client, "_ws_manager", None)
    if manager is None:
        try:
            client.disconnect()
        except Exception:
            log.debug("wecom adapter: disconnect failed", exc_info=True)
        await asyncio.sleep(0)
        return

    try:
        if hasattr(client, "_started"):
            client._started = False
        manager._is_manual_close = True
        stop_heartbeat = getattr(manager, "_stop_heartbeat", None)
        if stop_heartbeat is not None:
            stop_heartbeat()
        clear_pending = getattr(manager, "_clear_pending_messages", None)
        if clear_pending is not None:
            clear_pending("Connection manually closed")

        websocket = getattr(manager, "_ws", None)
        if websocket is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(
                    websocket.close(code=1000, reason="Manual disconnect"),
                    timeout=2,
                )
            manager._ws = None

        await _cancel_task(getattr(manager, "_receive_task", None), "receive")
        await _cancel_task(getattr(manager, "_heartbeat_task", None), "heartbeat")
        log.info("wecom adapter: websocket disconnected")
    except Exception:
        log.debug("wecom adapter: async disconnect failed", exc_info=True)


async def _cancel_task(task: asyncio.Task | None, name: str) -> None:
    if task is None or task.done():
        return
    task.cancel()
    try:
        await asyncio.wait_for(task, timeout=2)
    except (asyncio.CancelledError, TimeoutError):
        pass
    except Exception:
        log.debug("wecom adapter: %s task shutdown failed", name, exc_info=True)


@register
class WecomAdapter(Adapter):
    name = "wecom"
    Config = WecomConfig

    def __init__(self, config: WecomConfig, hub, store) -> None:  # type: ignore[override]
        super().__init__(config, hub, store)
        self._client: Any = None  # aibot.WSClient instance

    # ----------------------------------------------------------------- lifecycle

    async def start(self) -> None:
        from aibot.client import WSClient
        from aibot.types import WSClientOptions
        import aibot.ws as wecom_ws

        _patch_wecom_websocket_connect(wecom_ws)

        cfg: WecomConfig = self.config  # type: ignore[assignment]

        options = WSClientOptions(
            bot_id=cfg.bot_id,
            secret=cfg.secret,
            max_reconnect_attempts=-1,  # infinite reconnect
            logger=_WecomSdkLogger(),
        )
        self._client = WSClient(options)

        # Register message handler
        self._client.on("message", self._on_message)

        # Connect (establishes WebSocket in background)
        await self._client.connect()
        log.info("wecom adapter: websocket connected")

    async def stop(self) -> None:
        client = self._client
        self._client = None
        if client is not None:
            await _disconnect_wecom_client(client)

    # ----------------------------------------------------------------- inbound

    def _on_message(self, frame: dict[str, Any]) -> None:
        """Handler called by pyee when a message frame arrives (in asyncio context)."""
        try:
            body = frame.get("body", {})
            if not body:
                return

            raw = {
                "msgid": body.get("msgid", ""),
                "msgtype": body.get("msgtype", ""),
                "chattype": body.get("chattype", ""),  # "single" or "group"
                "chatid": body.get("chatid", ""),
                "chatname": body.get("chatname", ""),
                "from_userid": body.get("from", {}).get("userid", ""),
                "from_name": (
                    body.get("from", {}).get("alias", "")
                    or body.get("from", {}).get("name", "")
                ),
                "text_content": body.get("text", {}).get("content", ""),
                "req_id": frame.get("headers", {}).get("req_id", ""),
            }

            # Extract image / file / mixed attachments so parse_inbound
            # and on_event can process them.
            msgtype = raw["msgtype"]
            if msgtype == "image":
                img = body.get("image", {})
                raw["_image"] = {"url": img.get("url", ""), "aeskey": img.get("aeskey", "")}
            elif msgtype == "file":
                f = body.get("file", {})
                raw["_file"] = {
                    "url": f.get("url", ""),
                    "aeskey": f.get("aeskey", ""),
                    "name": f.get("file_name", ""),
                }
            elif msgtype == "mixed":
                raw["_mixed_items"] = body.get("mixed", {}).get("msg_item", [])

            # on_event is a coroutine — schedule it
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(self.on_event(raw))
            else:
                log.warning("wecom adapter: event loop not running, dropping message")
        except Exception:
            log.exception("wecom adapter: _on_message failed")

    async def on_event(self, raw: dict[str, Any]) -> int | None:
        """Override to download WeCom attachments before persisting."""
        parsed = self.parse_inbound(raw)
        if parsed is None:
            return None

        content = parsed.content
        if content.attachments and self._client is not None:
            attachments_dir = (self.store.db_path.parent / "attachments").resolve()
            attachments_dir.mkdir(parents=True, exist_ok=True)
            new_attachments: list[Attachment] = []
            for att in content.attachments:
                att_dict = att.model_dump()
                url = att.url
                aeskey = att.meta.get("aeskey", "") if att.meta else ""
                if url and aeskey:
                    try:
                        data, filename = await self._client.download_file(url, aeskey)
                        safe_name = filename or att.name or att.file_id or "file"
                        local_path = attachments_dir / f"{att.file_id}_{safe_name}"
                        local_path.write_bytes(data)
                        att_dict["path"] = str(local_path)
                        log.debug(
                            "wecom: downloaded %s -> %s (%d bytes)",
                            att.file_id, local_path, len(data),
                        )
                    except Exception:
                        log.exception("wecom: download failed for %s", att.file_id)
                new_attachments.append(Attachment(**att_dict))
            content = Content(text=content.text, attachments=new_attachments)

        sender = Sender(
            id=parsed.sender.id,
            name=parsed.sender.name or parsed.sender.id,
        )
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
                self.name, parsed.conv_id, sender.name, sender.id, text_preview,
            )
        return row_id

    def parse_inbound(self, raw: dict[str, Any]) -> ParsedInbound | None:
        msgtype = raw.get("msgtype", "")

        # Build text from the most relevant source for this message type.
        text = ""
        if msgtype == "text":
            text = raw.get("text_content", "")
        elif msgtype == "image":
            text = "[图片]"
        elif msgtype == "file":
            name = raw.get("_file", {}).get("name", "")
            text = f"[文件: {name}]" if name else "[文件]"
        elif msgtype == "mixed":
            for item in raw.get("_mixed_items", []):
                if item.get("msgtype") == "text":
                    text = item.get("text", {}).get("content", "")
                    break
        else:
            return None

        if not text and msgtype == "text":
            return None

        # Build attachment list for non-text media.
        attachments: list[Attachment] = []
        if msgtype == "image":
            img = raw.get("_image", {})
            if img.get("url"):
                attachments.append(Attachment(
                    kind="image",
                    url=img["url"],
                    file_id=raw.get("msgid", ""),
                    meta={"aeskey": img.get("aeskey", "")},
                ))
        elif msgtype == "file":
            f = raw.get("_file", {})
            if f.get("url"):
                attachments.append(Attachment(
                    kind="file",
                    url=f["url"],
                    name=f.get("name", ""),
                    file_id=raw.get("msgid", ""),
                    meta={"aeskey": f.get("aeskey", "")},
                ))
        elif msgtype == "mixed":
            for item in raw.get("_mixed_items", []):
                item_type = item.get("msgtype", "")
                if item_type == "image":
                    img = item.get("image", {})
                    if img.get("url"):
                        attachments.append(Attachment(
                            kind="image",
                            url=img["url"],
                            file_id=item.get("msgid", ""),
                            meta={"aeskey": img.get("aeskey", "")},
                        ))
                elif item_type == "file":
                    f = item.get("file", {})
                    if f.get("url"):
                        attachments.append(Attachment(
                            kind="file",
                            url=f["url"],
                            name=f.get("file_name", ""),
                            file_id=item.get("msgid", ""),
                            meta={"aeskey": f.get("aeskey", "")},
                        ))

        chattype = raw.get("chattype", "")
        from_userid = raw.get("from_userid", "")

        # Determine conv_id:
        # - Group chat: use chatid (group identifier)
        # - Single chat: use from_userid (the sender)
        if chattype == "group":
            conv_id = raw.get("chatid", "")
        else:
            conv_id = from_userid

        if not conv_id:
            return None

        msg_id = raw.get("msgid", "")
        sender = Sender(
            id=from_userid,
            name=raw.get("from_name", "") or from_userid,
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

    async def send(
        self,
        conv_id: str,
        content: Content,
    ) -> tuple[str, dict[str, Any]]:
        if self._client is None:
            raise SendError("wecom adapter not started; call start() first")

        text = content.text or ""
        body = {
            "msgtype": "markdown",
            "markdown": {"content": text},
        }

        try:
            result = await self._client.send_message(conv_id, body)
        except Exception as e:
            raise SendError(f"wecom send_message failed: {e}") from e

        # result is the ACK frame from WebSocket
        resp_data = dict(result) if result else {}
        msg_id = resp_data.get("headers", {}).get("req_id", "")
        return str(msg_id), resp_data
