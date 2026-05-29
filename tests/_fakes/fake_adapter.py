"""Fake adapter — used only in tests for e2e flows that should not touch the
network. It loops outbound messages back into the inbox so tests can verify
the full client→outbox→adapter.send→store→client roundtrip.

NOT auto-registered: tests must call `register(FakeAdapter)` explicitly inside
a fixture and `unregister('fake')` on teardown.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from pydantic import BaseModel

from linc.core.adapter import Adapter, ParsedInbound
from linc.core.models import Content, Sender


class FakeConfig(BaseModel):
    """Minimal config for FakeAdapter. Empty by design."""


class FakeAdapter(Adapter):
    """In-memory adapter. Records every send; lets tests inject inbound events."""

    name = "fake"
    Config = FakeConfig

    def __init__(self, config, hub, store):
        super().__init__(config, hub, store)
        # Recorded outbound calls — list of (conv_id, content).
        self.sent: list[tuple[str, Content]] = []
        self._started = False

    async def start(self) -> None:
        self._started = True

    async def stop(self) -> None:
        self._started = False

    async def send(
        self,
        conv_id: str,
        content: Content,
    ) -> tuple[str, dict[str, Any]]:
        self.sent.append((conv_id, content))
        msg_id = f"fake-{uuid.uuid4().hex[:12]}"
        return msg_id, {"ok": True, "conv_id": conv_id, "msg_id": msg_id}

    def parse_inbound(self, raw: dict[str, Any]) -> ParsedInbound | None:
        """Expects raw shaped like:
        {"conv_id": str, "msg_id": str, "ts": float|None,
         "sender": {"id": str, "name": str|None},
         "text": str|None, "attachments": [...] (optional)}

        Returns None if 'msg_id' is missing (treated as a non-message event).
        """
        msg_id = raw.get("msg_id")
        if not msg_id:
            return None
        sender_raw = raw.get("sender") or {}
        sender = Sender(
            id=str(sender_raw.get("id", "anon")),
            name=sender_raw.get("name"),
        )
        content = Content(
            text=raw.get("text"),
            attachments=raw.get("attachments") or [],
        )
        return ParsedInbound(
            conv_id=str(raw["conv_id"]),
            msg_id=str(msg_id),
            ts=float(raw.get("ts") or time.time()),
            sender=sender,
            content=content,
        )

    # --- test helpers --------------------------------------------------------

    async def inject(self, raw: dict[str, Any]) -> int | None:
        """Simulate an incoming platform event. Returns the inserted row id."""
        return await self.on_event(raw)
