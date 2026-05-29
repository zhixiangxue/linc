"""linc unified data models. Pydantic v2 BaseModel everywhere — no dataclasses.

Conventions:
  - All models are `frozen=True` (immutable; hashable).
  - `Content` is the linc-internal normalized message body. Adapters translate
    both directions: platform-raw -> Content (parse_inbound) and Content -> platform
    payload (send). The original platform JSON is also stored in `raw` as a fallback.
  - Direction is encoded by *type*, not by a field: InboundMessage vs OutboundMessage.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Content building blocks
# ---------------------------------------------------------------------------


class Attachment(BaseModel):
    """Unified attachment.

    v0.1 stores references only (url / file_id). Local download (`path`) lands
    in v0.2 along with content-addressed storage under .linc/attachments/.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["image", "video", "audio", "file"]
    url: str | None = None
    path: str | None = None
    file_id: str | None = None
    mime: str | None = None
    name: str | None = None
    size: int | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class Content(BaseModel):
    """Normalized message body. Either text, attachments, or both.

    Higher-level structures (mention / quote / buttons / cards) are intentionally
    NOT modeled in v0.1 — they will be added in v0.2 as additional optional fields.
    """

    model_config = ConfigDict(frozen=True)

    text: str | None = None
    attachments: list[Attachment] = Field(default_factory=list)


class Sender(BaseModel):
    """Inbound message sender. Required on every InboundMessage."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str | None = None
    is_bot: bool = False
    meta: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Stored messages
# ---------------------------------------------------------------------------


class InboundMessage(BaseModel):
    """A message received FROM a platform. Persisted with dir='in'."""

    model_config = ConfigDict(frozen=True)

    id: int
    platform: str
    conv_id: str
    msg_id: str
    ts: float
    status: Literal["unread", "read"]
    sender: Sender
    content: Content
    raw: dict[str, Any]


class OutboundMessage(BaseModel):
    """A message queued/sent TO a platform. Persisted with dir='out'.

    `msg_id` and `raw` are populated only after the dispatcher receives the
    platform's response (status transitions pending -> sent).
    """

    model_config = ConfigDict(frozen=True)

    id: int
    platform: str
    conv_id: str
    msg_id: str | None = None
    ts: float
    status: Literal["pending", "sent", "failed"]
    error: str | None = None
    content: Content
    raw: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Client-side construction
# ---------------------------------------------------------------------------


class OutboundDraft(BaseModel):
    """Constructed by the agent client, handed to SqliteStore.enqueue_outbound.

    The `platform` field is supplied separately at enqueue time (it comes from
    the Client handle, not from the agent), so it lives outside the draft.
    """

    model_config = ConfigDict(frozen=True)

    conv_id: str
    content: Content
