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
import re
from typing import Any

from pydantic import BaseModel, Field

from ..core.adapter import Adapter, ParsedInbound
from ..core.errors import SendError
from ..core.models import Content, Sender
from . import register

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Markdown → Slack mrkdwn conversion
# ---------------------------------------------------------------------------

# Patterns that need special handling beyond what slackify_markdown provides.
_TABLE_RE = re.compile(r"(?m)^\|.*\|$(?:\n\|[\s:|-]*\|$)(?:\n\|.*\|$)*")
_CODE_FENCE_RE = re.compile(r"```[\s\S]*?```")
_INLINE_CODE_RE = re.compile(r"`[^`]+`")
_LEFTOVER_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_LEFTOVER_HEADER_RE = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)
_LATEX_BLOCK_RE = re.compile(r"\\\[[\s\S]*?\\\]")
_LATEX_INLINE_RE = re.compile(r"\\\([\s\S]*?\\\)")


def _table_to_text(match: re.Match) -> str:
    """Convert a Markdown table into Slack-friendly bullet-style text."""
    lines = [ln.strip() for ln in match.group(0).strip().splitlines() if ln.strip()]
    if len(lines) < 2:
        return match.group(0)
    headers = [h.strip() for h in lines[0].strip("|").split("|")]
    start = 2 if re.fullmatch(r"[|\s:\-]+", lines[1]) else 1
    rows = []
    for line in lines[start:]:
        cells = (line.strip("|").split("|") + [""] * len(headers))[:len(headers)]
        parts = [f"*{headers[i].strip()}*: {cells[i].strip()}" for i in range(len(headers)) if cells[i].strip()]
        if parts:
            rows.append(" · ".join(parts))
    return "\n".join(rows)


def _to_slack_mrkdwn(text: str) -> str:
    """Convert LLM-flavored Markdown to Slack mrkdwn format."""
    if not text:
        return ""

    # Strip LaTeX math — Slack mrkdwn has no math support.
    text = _LATEX_BLOCK_RE.sub(
        lambda m: m.group(0).replace("\\[", "").replace("\\]", "").strip(), text
    )
    text = _LATEX_INLINE_RE.sub(
        lambda m: m.group(0).replace("\\(", "").replace("\\)", "").strip(), text
    )

    # Run slackify_markdown first so it handles standard **bold**, *italic*,
    # - bullets, [links], > quotes, etc. before our custom table/fallback rules.
    try:
        from slackify_markdown import slackify_markdown
        text = slackify_markdown(text)
    except ImportError:
        # Basic built-in conversion fallback
        text = _LEFTOVER_BOLD_RE.sub(r"*\1*", text)
        text = _LEFTOVER_HEADER_RE.sub(r"*\1*", text)
        text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"<\2|\1>", text)

    # Convert Markdown tables AFTER slackify so header *bold* markers survive.
    text = _TABLE_RE.sub(_table_to_text, text)

    # Protect code blocks from further substitutions.
    code_blocks: list[str] = []

    def _stash(m: re.Match) -> str:
        code_blocks.append(m.group(0))
        return f"\x00CB{len(code_blocks) - 1}\x00"

    text = _CODE_FENCE_RE.sub(_stash, text)
    text = _INLINE_CODE_RE.sub(_stash, text)

    # Post-process: any **bold** that survived → *bold* (Slack mrkdwn format).
    text = _LEFTOVER_BOLD_RE.sub(r"*\1*", text)
    text = _LEFTOVER_HEADER_RE.sub(r"*\1*", text)

    # Restore code blocks.
    for i, block in enumerate(code_blocks):
        text = text.replace(f"\x00CB{i}\x00", block)

    return text


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

        # Download Slack file attachments via bot token and save locally.
        # ``url_private`` requires auth — the LLM can't access it directly,
        # so we download now and store a local ``path`` that the agent can
        # read and convert to base64 for the multimodal model.
        content = parsed.content
        if content.attachments and self._web is not None:
            import aiohttp

            cfg: SlackConfig = self.config  # type: ignore[assignment]
            attachments_dir = (self.store.db_path.parent / "attachments").resolve()
            attachments_dir.mkdir(parents=True, exist_ok=True)
            new_attachments: list[dict[str, Any]] = []
            async with aiohttp.ClientSession() as session:
                for att in content.attachments:
                    att_dict = att.model_dump()
                    file_url = att.url
                    if file_url and file_url.startswith("https://files.slack.com/"):
                        try:
                            headers = {"Authorization": f"Bearer {cfg.bot_token}"}
                            async with session.get(file_url, headers=headers) as resp:
                                if resp.status == 200:
                                    file_bytes = await resp.read()
                                    safe_name = att.name or att.file_id or "file"
                                    local_path = attachments_dir / f"{att.file_id}_{safe_name}"
                                    local_path.write_bytes(file_bytes)
                                    att_dict["path"] = str(local_path)
                                    log.debug(
                                        "slack: downloaded %s -> %s (%d bytes)",
                                        att.file_id, local_path, len(file_bytes),
                                    )
                                else:
                                    log.warning(
                                        "slack: download %s returned %d",
                                        file_url, resp.status,
                                    )
                        except Exception:
                            log.exception("slack: download failed for %s", att.file_id)
                    new_attachments.append(att_dict)
            content = Content(text=content.text, attachments=new_attachments)

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
        # non-conversational — EXCEPT ``file_share``, which represents a user
        # uploading an image / file alongside (optional) text.
        if raw.get("type") != "message":
            return None
        subtype = raw.get("subtype")
        if subtype and subtype != "file_share":
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

        # Build attachment list from Slack ``files`` array (present on
        # ``file_share`` events). Map Slack mimetype prefix to linc kind.
        _MIME_KIND: dict[str, str] = {
            "image/": "image",
            "video/": "video",
            "audio/": "audio",
        }
        attachments: list[Any] = []
        for f in raw.get("files") or []:
            kind = "file"
            mime = f.get("mimetype") or ""
            for prefix, k in _MIME_KIND.items():
                if mime.startswith(prefix):
                    kind = k
                    break
            attachments.append(
                dict(
                    kind=kind,
                    url=f.get("url_private") or f.get("url_private_download"),
                    file_id=f.get("id"),
                    mime=mime or None,
                    name=f.get("name"),
                    size=f.get("size"),
                )
            )

        sender = Sender(id=str(user), name=str(user))
        content = Content(text=str(text), attachments=attachments)
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
            mrkdwn_text = _to_slack_mrkdwn(content.text or "")
            resp = await self._web.chat_postMessage(
                channel=conv_id,
                text=mrkdwn_text,
                mrkdwn=True,
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
