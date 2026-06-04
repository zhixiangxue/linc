"""Agent-side SDK: ``Client`` context manager + ``Messenger`` handle.

Typical usage in an agent script (NOT inside the gateway process)::

    async with Client(".linc") as client:
        unread = await client.slack.read_unread()
        for m in unread:
            await client.slack.send(conv_id=m.conv_id, content=f"echo: {m.content.text}")

Design notes:
- ``Client.__aenter__`` acquires ``agent.lock`` flock so at most one agent process
  talks to a given data_dir at a time. The gateway holds its own ``linc.pid``
  lock; the two are independent — gateway and agent can (and must) coexist.
- ``client.messenger(platform)`` is the preferred explicit way to obtain a platform
  messenger. ``client.<platform>`` is resolved via ``__getattr__`` with adapter
  registry validation.
- The client does NOT instantiate adapters and does NOT need ``linc.yaml``. It
  only needs read/write access to the SQLite file at ``<data_dir>/linc.db``.
- All write operations go through the SAME ``SqliteStore`` API used by the
  gateway, so concurrent gateway + agent are safe (single-writer SQLite via
  asyncio.Lock + WAL).

Naming: ``Messenger`` is the per-platform SDK façade (a thin, stateless proxy over
the shared store). It does **not** open any network connection — that is the
gateway-side ``Adapter``'s job.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .adapters import is_supported, supported
from .core.locks import acquire_agent_lock, release
from .core.models import Content, InboundMessage, OutboundDraft, OutboundMessage
from .core.store import SqliteStore


class Client:
    """Async context manager — the agent-side entry point."""

    def __init__(self, data_dir: str | Path = ".linc") -> None:
        self._data_dir = Path(data_dir).expanduser()
        self._store: SqliteStore | None = None
        self._lock_fd: int | None = None

    # ------------------------------------------------------------------ lifecycle

    async def __aenter__(self) -> "Client":
        self._data_dir.mkdir(parents=True, exist_ok=True)
        # flock first — fails fast if another agent already holds it.
        self._lock_fd = acquire_agent_lock(self._data_dir)
        try:
            self._store = SqliteStore(self._data_dir / "linc.db")
            await self._store.open()
        except Exception:
            # Release the lock so a retry isn't blocked.
            release(self._lock_fd)
            self._lock_fd = None
            self._store = None
            raise
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._store is not None:
            try:
                await self._store.close()
            finally:
                self._store = None
        if self._lock_fd is not None:
            try:
                release(self._lock_fd)
            finally:
                self._lock_fd = None

    # ------------------------------------------------------------------ store access

    @property
    def store(self) -> SqliteStore:
        if self._store is None:
            raise RuntimeError("Client not entered; use 'async with Client(...) as client'")
        return self._store

    # ------------------------------------------------------------------ platform factory

    def messenger(self, platform: str, conv_id: str | None = None) -> "Messenger":
        """Return a Messenger for the given platform.

        Use this for dynamic/variable platform names. For static access,
        prefer the attribute form: ``client.slack``, ``client.feishu``.
        """
        if not is_supported(platform):
            raise ValueError(
                f"unknown IM platform {platform!r}; "
                f"registered platforms are {sorted(supported())}. "
                f"Did you forget to import the adapter, or mistype the name?"
            )
        return Messenger(name=platform, store=self.store, conv_id=conv_id)

    def __getattr__(self, name: str) -> "Messenger":
        """Attribute access: ``client.slack`` / ``client.feishu`` ...

        Returns a Messenger directly (no call needed). Raises ``AttributeError``
        for any name not in the adapter registry.
        """
        # Internal / dunder attributes must not be intercepted, else getattr
        # probes (hasattr, copy, pickle, ...) blow up with confusing errors.
        if name.startswith("_"):
            raise AttributeError(name)
        if not is_supported(name):
            raise AttributeError(
                f"unknown IM platform {name!r}; "
                f"registered platforms are {sorted(supported())}. "
                f"Did you forget to import the adapter, or mistype the name?"
            )
        return Messenger(name=name, store=self.store)

    # ------------------------------------------------------------------ cross-platform helpers

    async def read_unread(
        self, limit: int | None = None,
    ) -> list[InboundMessage]:
        """Atomically claim unread messages across **all** platforms."""
        return await self.store.claim_unread(platform=None, conv_id=None, limit=limit)

    async def history(
        self,
        platform: str | None = None,
        conv_id: str | None = None,
        since: float | None = None,
        limit: int = 100,
    ) -> list[InboundMessage | OutboundMessage]:
        return await self.store.history(
            platform=platform, conv_id=conv_id, since=since, limit=limit
        )


class Messenger:
    """Per-platform SDK handle bound to (optionally) a single conversation.

    A thin, stateless proxy over the shared ``SqliteStore`` — every method just
    translates to a store call with ``platform=self.name`` (and optional
    ``conv_id``) filled in. It does NOT open IM connections; that work belongs
    to the gateway-side ``Adapter``.
    """

    __slots__ = ("name", "_store", "_conv_id")

    def __init__(
        self,
        name: str,
        store: SqliteStore,
        conv_id: str | None = None,
    ) -> None:
        self.name = name
        self._store = store
        self._conv_id = conv_id

    # ------------------------------------------------------------------ scope helpers

    def conv(self, conv_id: str) -> "Messenger":
        """Return a new handle bound to ``conv_id`` (chainable)."""
        return Messenger(name=self.name, store=self._store, conv_id=conv_id)

    # ------------------------------------------------------------------ inbound

    async def read_unread(
        self, conv_id: str | None = None, limit: int | None = None,
    ) -> list[InboundMessage]:
        """Atomically claim unread inbound messages for this platform.

        ``conv_id`` precedence: method arg > handle binding > None (all convs).
        """
        cid = conv_id if conv_id is not None else self._conv_id
        return await self._store.claim_unread(
            platform=self.name, conv_id=cid, limit=limit
        )

    async def list_unread(
        self, conv_id: str | None = None, limit: int | None = None,
    ) -> list[InboundMessage]:
        """Peek at unread messages WITHOUT marking them read (debug/inspection)."""
        cid = conv_id if conv_id is not None else self._conv_id
        return await self._store.list_unread(
            platform=self.name, conv_id=cid, limit=limit
        )

    # ------------------------------------------------------------------ outbound

    async def send(
        self,
        content: str | Content = "",
        *,
        conv_id: str | None = None,
    ) -> int:
        """Enqueue an outbound message for the gateway to deliver.

        Args:
            content: str (becomes Content(text=...)) or a Content instance.
            conv_id: target conversation. Method arg > handle binding; one
                of them MUST be set, otherwise raises ValueError.

        Returns:
            The outbox row id (useful for tracing in `history`).
        """
        cid = conv_id if conv_id is not None else self._conv_id
        if cid is None:
            raise ValueError(
                f"send() requires conv_id (either as argument or bound via "
                f"client.{self.name}.conv(conv_id))"
            )
        if isinstance(content, str):
            payload = Content(text=content)
        else:
            payload = content
        draft = OutboundDraft(conv_id=cid, content=payload)
        return await self._store.enqueue_outbound(
            platform=self.name, draft=draft, ts=time.time()
        )

    # ------------------------------------------------------------------ history

    async def history(
        self,
        conv_id: str | None = None,
        since: float | None = None,
        limit: int = 100,
    ) -> list[InboundMessage | OutboundMessage]:
        cid = conv_id if conv_id is not None else self._conv_id
        return await self._store.history(
            platform=self.name, conv_id=cid, since=since, limit=limit
        )


__all__ = ["Client", "Messenger"]
