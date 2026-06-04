"""linc unified data models. Pydantic v2 BaseModel everywhere — no dataclasses.

Conventions:
  - All models are `frozen=True` (immutable; hashable).
  - `Content` is the linc-internal normalized message body. Adapters translate
    both directions: platform-raw -> Content (parse_inbound) and Content -> platform
    payload (send). The original platform JSON is also stored in `raw` as a fallback.
  - Direction is encoded by *type*, not by a field: InboundMessage vs OutboundMessage.
"""

from __future__ import annotations

import base64 as _b64
import mimetypes as _mt
from pathlib import Path as _Path
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

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    _IMAGE_EXTS: set[str] = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg"}
    _AUDIO_EXTS: set[str] = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac"}
    _VIDEO_EXTS: set[str] = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

    @property
    def is_image(self) -> bool:
        """True if this attachment is an image (by kind, MIME, or extension)."""
        if self.kind == "image":
            return True
        if self.kind != "file":
            return False
        if self.mime and self.mime.startswith("image/"):
            return True
        return self.ext in self._IMAGE_EXTS

    @property
    def is_audio(self) -> bool:
        """True if this attachment is audio (by kind, MIME, or extension)."""
        if self.kind == "audio":
            return True
        if self.kind != "file":
            return False
        if self.mime and self.mime.startswith("audio/"):
            return True
        return self.ext in self._AUDIO_EXTS

    @property
    def is_video(self) -> bool:
        """True if this attachment is a video (by kind, MIME, or extension)."""
        if self.kind == "video":
            return True
        if self.kind != "file":
            return False
        if self.mime and self.mime.startswith("video/"):
            return True
        return self.ext in self._VIDEO_EXTS

    @property
    def is_media(self) -> bool:
        """True if image, audio, or video."""
        return self.is_image or self.is_audio or self.is_video

    @property
    def ext(self) -> str:
        """File extension including the dot (e.g. '.png'), or '' if unknown.

        Resolved from `name`, `path`, or guessed from `mime`.
        """
        # Try name first
        if self.name:
            suffix = _Path(self.name).suffix.lower()
            if suffix:
                return suffix
        # Then path
        if self.path:
            suffix = _Path(self.path).suffix.lower()
            if suffix:
                return suffix
        # Fallback: guess from MIME
        if self.mime:
            guessed = _mt.guess_extension(self.mime)
            if guessed:
                return guessed.lower()
        return ""

    @property
    def data_uri(self) -> str | None:
        """Base64-encoded data URI from local `path`, or None if unavailable.

        Returns e.g. 'data:image/png;base64,iVBOR...' ready for LLM consumption.
        """
        if not self.path:
            return None
        p = _Path(self.path)
        if not p.is_file():
            return None
        data = p.read_bytes()
        b64 = _b64.b64encode(data).decode("ascii")
        mime = self.mime or _mt.guess_type(p.name)[0] or "application/octet-stream"
        return f"data:{mime};base64,{b64}"

    def read_bytes(self) -> bytes | None:
        """Read raw file bytes from local `path`, or None if unavailable."""
        if not self.path:
            return None
        p = _Path(self.path)
        if not p.is_file():
            return None
        return p.read_bytes()


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
