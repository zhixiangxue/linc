"""linc exception hierarchy. Keep this module dependency-free."""

from __future__ import annotations


class LincError(Exception):
    """Base class for all linc-defined exceptions."""


class AlreadyRunning(LincError):
    """Raised when another linc-gateway or agent process holds the flock."""


class UnknownPlatform(LincError):
    """Raised when a requested platform name is not in SUPPORTED_PLATFORMS."""


class ConfigError(LincError):
    """Raised when linc.yaml fails to load or validate."""


class StoreError(LincError):
    """Raised on internal SqliteStore failures (schema mismatch, corruption, etc.)."""


class SendError(LincError):
    """Raised by an Adapter.send() implementation when the platform call fails.

    The dispatcher will catch this and persist `error` on the outbound row.
    Carry the original exception via `__cause__` (use `raise SendError(...) from e`).
    """
