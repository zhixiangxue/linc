"""Slack adapter — Socket Mode (no public webhook required).

Why Socket Mode for v0.1: it works behind NAT / on a laptop / in CI without any
ingress configuration. The trade-off is one persistent WebSocket per gateway
process; that's fine for v0.1 scale (small number of bot installations).

Wire-up (slack-sdk):
  - ``AsyncWebClient(bot_token)`` issues outbound REST calls (``chat.postMessage``).
  - ``SocketModeClient(app_token, web_client)`` opens the WebSocket for events
    and writes ack frames back through the same socket.

Event filtering: Slack delivers many things that are NOT user messages —
``message_changed``, ``message_deleted``, ``channel_join``, bot's own echo,
etc. We filter all of those in ``parse_inbound`` (return ``None``) so the
default ``on_event`` simply drops them. Without this filter the bot will
echo its own replies into an infinite loop.

Sender resolution: ``on_event`` resolves user IDs to display names via
``users.info`` with an in-memory cache (one API call per unique user).
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from ..core.adapter import Adapter, ParsedInbound
from ..core.errors import SendError
from ..core.models import Content, Sender
from . import register

log = logging.getLogger(__name__)


class SlackConfig(BaseModel):
    """Slack Socket Mode credentials.

    Both tokens are issued from a Slack App (api.slack.com/apps):
      - ``bot_token`` (xoxb-...): grants chat:write, channels:history, etc.
      - ``app_token`` (xapp-...): app-level token with ``connections:write``
        scope, used solely to open the Socket Mode WebSocket.
    """

    bot_token: str = Field(..., description="Bot User OAuth Token (xoxb-...)")
    app_token: str = Field(..., description="App-Level Token with connections:write (xapp-...)")


@register
class SlackAdapter(Adapter):
    name = "slack"
    Config = SlackConfig

    def __init__(self, config: SlackConfig, hub, store) -> None:  # type: ignore[override]
        super().__init__(config, hub, store)
        # Lazy-initialized in start() so that constructing the adapter never
        # opens any sockets (important for tests).
        self._web: Any = None
        self._sm: Any = None
        self._user_cache: dict[str, str] = {}  # user_id -> display_name

    # ----------------------------------------------------------------- lifecycle

    async def start(self) -> None:
        # Imports are local so that unit tests that monkeypatch ``self._web``
        # without ever calling ``start()`` don't pay the import cost — and so
        # that import-time failures here surface only when actually used.
        from slack_sdk.socket_mode.aiohttp import SocketModeClient
        from slack_sdk.web.async_client import AsyncWebClient

        cfg: SlackConfig = self.config  # type: ignore[assignment]
        self._web = AsyncWebClient(token=cfg.bot_token)
        self._sm = SocketModeClient(app_token=cfg.app_token, web_client=self._web)
        self._sm.socket_mode_request_listeners.append(self._listener)
        await self._sm.connect()
        log.info("slack adapter: socket mode connected")

    async def stop(self) -> None:
        # Best-effort teardown; nothing here should throw out.
        # Note: AsyncWebClient has no close() — it manages aiohttp sessions
        # per-request internally. Only the SocketModeClient owns the long-
        # lived WebSocket and needs explicit disconnect/close.
        sm = self._sm
        self._sm = None
        self._web = None
        if sm is not None:
            try:
                await sm.disconnect()
            except Exception:
                log.exception("slack adapter: disconnect failed")
            try:
                await sm.close()
            except Exception:
                log.exception("slack adapter: close failed")

    # ----------------------------------------------------------------- inbound

    async def _resolve_user_name(self, user_id: str) -> str:
        """Resolve a Slack user ID to display name, with in-memory cache."""
        if user_id in self._user_cache:
            return self._user_cache[user_id]
        if self._web is None:
            return user_id
        try:
            resp = await self._web.users_info(user=user_id)
            data = getattr(resp, "data", resp) or {}
            user_obj = data.get("user", {})
            profile = user_obj.get("profile", {})
            name = (
                profile.get("display_name_normalized")
                or profile.get("display_name")
                or profile.get("real_name_normalized")
                or user_obj.get("real_name")
                or profile.get("real_name")
                or user_id
            )
            log.debug("slack user resolve: %s -> %s (profile=%r)", user_id, name, profile)
            self._user_cache[user_id] = name
            return name
        except Exception as e:
            log.warning("slack adapter: users.info failed for %s: %s", user_id, e)
            self._user_cache[user_id] = user_id
            return user_id

    async def on_event(self, raw: dict[str, Any]) -> int | None:
        """Override to resolve user display name before persisting."""
        parsed = self.parse_inbound(raw)
        if parsed is None:
            return None
        # Resolve real display name asynchronously
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
                "\u2b07 [%s] %s | %s (%s): %s",
                self.name, parsed.conv_id, display_name, parsed.sender.id, text_preview,
            )
        return row_id

    async def _listener(self, client: Any, req: Any) -> None:
        """Single Socket Mode dispatch entry point.

        Per Slack docs we MUST ack every envelope; failure to do so causes the
        server to redeliver. We ack first, then process — if processing throws,
        Slack still considers the event delivered.
        """
        from slack_sdk.socket_mode.response import SocketModeResponse

        try:
            await client.send_socket_mode_response(
                SocketModeResponse(envelope_id=req.envelope_id)
            )
        except Exception:
            log.exception("slack adapter: failed to ack envelope")
            return

        if req.type != "events_api":
            return
        event = (req.payload or {}).get("event") or {}
        if not event:
            return
        try:
            await self.on_event(event)
        except Exception:
            log.exception("slack adapter: on_event failed for event %r", event.get("ts"))

    def parse_inbound(self, raw: dict[str, Any]) -> ParsedInbound | None:
        # Drop anything that isn't a fresh user message. The set of "weird"
        # message subtypes is open-ended, so we treat ANY ``subtype`` as
        # non-conversational. This matches what real Slack bots do.
        if raw.get("type") != "message":
            return None
        if raw.get("subtype"):
            return None
        # Bot's own messages echo through the events API; without this filter
        # an echo agent loops forever.
        if raw.get("bot_id"):
            return None

        channel = raw.get("channel")
        user = raw.get("user")
        text = raw.get("text") or ""
        ts_str = raw.get("ts")
        if not channel or not user or not ts_str:
            return None
        try:
            ts = float(ts_str)
        except (TypeError, ValueError):
            return None

        # Sender uses user_id as placeholder; on_event resolves the real name.
        sender = Sender(id=str(user), name=str(user))
        content = Content(text=str(text))
        return ParsedInbound(
            conv_id=str(channel),
            msg_id=str(ts_str),
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
        if self._web is None:
            raise SendError("slack adapter not started; call start() first")
        try:
            resp = await self._web.chat_postMessage(
                channel=conv_id,
                text=content.text or "",
            )
        except Exception as e:
            raise SendError(f"slack chat_postMessage raised: {e}") from e

        # AsyncWebClient returns SlackResponse; .data is the parsed JSON dict.
        data = dict(getattr(resp, "data", resp) or {})
        if not data.get("ok", False):
            raise SendError(f"slack chat_postMessage error: {data.get('error', '<unknown>')}")
        ts = data.get("ts")
        if not ts:
            raise SendError("slack chat_postMessage returned no ts")
        return str(ts), data
