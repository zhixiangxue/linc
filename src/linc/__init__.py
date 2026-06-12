"""linc — IM Gateway daemon for LLM Agents."""

from .client import Client, Messenger
from .core.models import (
    Attachment,
    Content,
    InboundMessage,
    OutboundDraft,
    OutboundMessage,
    Sender,
)
from .launch import launch

__version__ = "0.1.0dev0"

__all__ = [
    "Client",
    "Messenger",
    "Content",
    "Attachment",
    "Sender",
    "InboundMessage",
    "OutboundMessage",
    "OutboundDraft",
    "launch",
    "__version__",
]
