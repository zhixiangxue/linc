"""linc — IM Gateway daemon for LLM Agents."""

from .client import Client, Linc
from .core.models import (
    Attachment,
    Content,
    InboundMessage,
    OutboundDraft,
    OutboundMessage,
    Sender,
)

__version__ = "0.1.0dev0"

__all__ = [
    "Linc",
    "Client",
    "Content",
    "Attachment",
    "Sender",
    "InboundMessage",
    "OutboundMessage",
    "OutboundDraft",
    "__version__",
]
