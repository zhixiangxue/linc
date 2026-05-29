"""Adapter ABC — the only abstraction every IM platform implementation must obey.

Lifecycle (called by LincGateway):
    1. cls(config, hub, store)
    2. await adapter.start()    # connect / open polling / open socket
    3. ... incoming events trigger `await adapter.on_event(raw)` (default impl
       parses + persists). Outgoing messages are pulled from the outbox by the
       dispatcher, which calls `await adapter.send(conv_id, content)`.
    4. await adapter.stop()

Subclasses MUST set the class variable `name` (lowercase platform identifier,
e.g. "slack") and MUST implement: start, stop, send, parse_inbound. They MAY
override `on_event` if they need custom inbound bookkeeping.
"""

from __future__ import annotations

import abc
import logging
from typing import Any, ClassVar, NamedTuple

from pydantic import BaseModel

from .models import Content, Sender
from .hub import Hub
from .store import SqliteStore

log = logging.getLogger(__name__)


class ParsedInbound(NamedTuple):
    """Tuple returned by `Adapter.parse_inbound` for valid message events."""

    conv_id: str
    msg_id: str
    ts: float
    sender: Sender
    content: Content


class Adapter(abc.ABC):
    """Base class for IM platform adapters."""

    # Subclasses MUST set this to the platform identifier (matches REGISTRY key).
    name: ClassVar[str]
    # Subclasses MUST set this to the pydantic model used to validate the
    # adapter's section in linc.yaml. The server instantiates it via
    # `cls.Config.model_validate(raw_yaml_dict)` before passing it in.
    Config: ClassVar[type[BaseModel]]

    def __init__(
        self,
        config: BaseModel,
        hub: Hub,
        store: SqliteStore,
    ) -> None:
        self.config = config
        self.hub = hub
        self.store = store

    # -------------------------------------------------------------- lifecycle

    @abc.abstractmethod
    async def start(self) -> None:
        """Connect to the platform: open polling loop, websocket, or webhook routes."""

    @abc.abstractmethod
    async def stop(self) -> None:
        """Disconnect cleanly. Should be idempotent."""

    # -------------------------------------------------------------- outbound

    @abc.abstractmethod
    async def send(
        self,
        conv_id: str,
        content: Content,
    ) -> tuple[str, dict[str, Any]]:
        """Translate `content` into a platform API call.

        Returns `(platform_msg_id, platform_raw_response)`. Raises SendError
        (with the original exception chained) on failure; the dispatcher will
        persist the error string and mark the row 'failed'.
        """

    # -------------------------------------------------------------- inbound

    @abc.abstractmethod
    def parse_inbound(self, raw: dict[str, Any]) -> ParsedInbound | None:
        """Translate a platform event dict into linc's normalized form.

        Return `None` for events that are not user messages (presence updates,
        typing indicators, internal pings, etc.) — they will be silently dropped.
        """

    async def on_event(self, raw: dict[str, Any]) -> int | None:
        """Default inbound pipeline: parse_inbound -> store.insert_inbound.

        Subclasses normally do not need to override this. It returns the new
        row id (or None on duplicate / non-message event).
        """
        parsed = self.parse_inbound(raw)
        if parsed is None:
            return None
        row_id = await self.store.insert_inbound(
            platform=self.name,
            conv_id=parsed.conv_id,
            msg_id=parsed.msg_id,
            ts=parsed.ts,
            sender=parsed.sender,
            content=parsed.content,
            raw=raw,
        )
        if row_id is not None:
            text_preview = (parsed.content.text or "")[:80]
            log.info(
                "\u2b07 [%s] %s | %s: %s",
                self.name, parsed.conv_id, parsed.sender.name or parsed.sender.id, text_preview,
            )
        return row_id
