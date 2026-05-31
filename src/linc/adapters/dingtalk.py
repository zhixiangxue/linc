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
from ..core.models import Content, Sender
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

    def parse_inbound(self, raw: dict[str, Any]) -> ParsedInbound | None:
        text = raw.get("text", "")
        if not text:
            return None

        # Only handle text messages in v0.1
        msg_type = raw.get("message_type", "")
        if msg_type and msg_type != "text":
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
        content = Content(text=text)

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
        if self._stream_client is None:
            raise SendError("dingtalk adapter not started; call start() first")

        text = content.text or ""

        # Get access_token from SDK (manages expiry internally)
        access_token = self._stream_client.get_access_token()
        if not access_token:
            raise SendError("dingtalk adapter: failed to get access_token")

        headers = {
            "x-acs-dingtalk-access-token": access_token,
            "Content-Type": "application/json",
        }

        # Determine target type from conv_id prefix
        if conv_id.startswith("user:"):
            user_id = conv_id[5:]
            url = f"{DINGTALK_API}/v1.0/robot/oToMessages/batchSend"
            payload = {
                "robotCode": self._robot_code,
                "userIds": [user_id],
                "msgKey": "sampleText",
                "msgParam": f'{{"content": {_json_escape(text)}}}',
            }
        elif conv_id.startswith("group:"):
            group_id = conv_id[6:]
            url = f"{DINGTALK_API}/v1.0/robot/groupMessages/send"
            payload = {
                "robotCode": self._robot_code,
                "openConversationId": group_id,
                "msgKey": "sampleText",
                "msgParam": f'{{"content": {_json_escape(text)}}}',
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

        # The API returns processQueryKey for async delivery tracking
        msg_id = data.get("processQueryKey", "") or data.get("messageId", "")
        return str(msg_id), data


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
