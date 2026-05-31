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
from ..core.models import Content, Sender
from . import register

log = logging.getLogger(__name__)


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
                self._loop.create_task(self.on_event(raw))
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
            response = await self._lark_client.contact.v3.user.aget(request)
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

    async def on_event(self, raw: dict[str, Any]) -> int | None:
        """Override to resolve Feishu user display name before persisting."""
        parsed = self.parse_inbound(raw)
        if parsed is None:
            return None

        display_name = await self._resolve_user_name(parsed.sender.id)
        sender = Sender(id=parsed.sender.id, name=display_name)
        row_id = await self.store.insert_inbound(
            platform=self.name,
            conv_id=parsed.conv_id,
            msg_id=parsed.msg_id,
            ts=parsed.ts,
            sender=sender,
            content=parsed.content,
            raw=raw,
        )
        if row_id is not None:
            text_preview = (parsed.content.text or "")[:80]
            log.info(
                "⬇ [%s] %s | %s (%s): %s",
                self.name,
                parsed.conv_id,
                display_name,
                parsed.sender.id,
                text_preview,
            )
        return row_id

    def parse_inbound(self, raw: dict[str, Any]) -> ParsedInbound | None:
        # Filter out bot's own messages
        if raw.get("sender_type") == "app":
            return None

        # Only handle text messages in v0.1
        message_type = raw.get("message_type")
        content_str = raw.get("content") or ""

        # Parse content JSON — text messages are {"text": "hello"}
        text = ""
        if content_str:
            import json
            try:
                content_data = json.loads(content_str)
                text = content_data.get("text", "")
            except (json.JSONDecodeError, TypeError):
                text = content_str

        if not text:
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
        content = Content(text=text)
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
        msg_content = json_mod.dumps({"text": text})

        request = (
            CreateMessageRequest.builder()
            .receive_id_type(receive_id_type)
            .request_body(
                CreateMessageRequestBody.builder()
                .receive_id(conv_id)
                .content(msg_content)
                .msg_type("text")
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
