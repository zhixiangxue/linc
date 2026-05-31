"""LincGateway — long-running daemon: lifecycle + outbox dispatcher.

Despite the word "gateway", this is **not** an HTTP server. It is a Python
daemon process that owns the SQLite file, instantiates one adapter per IM
platform listed in `linc.yaml`, and runs a background dispatcher loop. The
name reflects PRD §1: linc is an "IM Gateway process" sitting between agents
and IM platforms.

Responsibilities (PRD §6.9):
  1. Acquire `linc.pid` flock so only one gateway runs per data_dir.
  2. Open SqliteStore (single connection, WAL, schema migrate).
  3. Bring up the shared Hub (HttpClient, future shared WebServer).
  4. Instantiate every adapter listed in `linc.yaml`, validate its config
     section against `Adapter.Config`, then call `start()`.
  5. Run the outbox dispatcher loop: poll for `pending` rows, dispatch via
     the right adapter, mark `sent` / `failed`.
  6. On stop signal: cancel dispatcher, stop adapters in reverse order, shut
     down hub, close store, release flock.

The dispatcher uses a simple polling tick (default 100ms). PRAGMA data_version
based wakeup is a v0.2 optimization — the spike confirmed it works, but a
100ms tick is already inaudible compared to network RTT and keeps the code
trivially correct.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path

from .adapters import get as get_adapter_cls
from .core.adapter import Adapter
from .core.config import LincConfig
from .core.errors import ConfigError, SendError
from .core.http import HttpxClient
from .core.hub import Hub
from .core.locks import acquire_gateway_lock, release
from .core.store import SqliteStore

log = logging.getLogger(__name__)
ADAPTER_STOP_TIMEOUT = 5.0


class LincGateway:
    """Daemon process owning the SQLite file and all live adapter connections."""

    def __init__(self, config: LincConfig) -> None:
        self.config = config
        self.store: SqliteStore | None = None
        self.hub: Hub | None = None
        self.adapters: dict[str, Adapter] = {}
        self._lock_fd: int | None = None
        self._dispatcher_task: asyncio.Task[None] | None = None
        self._stop_event: asyncio.Event = asyncio.Event()

    # ------------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        """Acquire lock, init everything, kick off the dispatcher.

        Raises AlreadyRunning if another linc-gateway holds the lock.
        Raises ConfigError if `linc.yaml` references unknown platforms or has
        invalid per-adapter config.
        """
        data_dir = self.config.data_dir.expanduser()
        data_dir.mkdir(parents=True, exist_ok=True)
        adapter_names = list(self.config.adapters)
        log.info(
            "🚀 linc-gateway starting: data_dir=%s adapters=%s pid=%d",
            data_dir,
            adapter_names,
            os.getpid(),
        )

        # 1. flock so only one gateway per data_dir
        self._lock_fd = acquire_gateway_lock(data_dir)

        # 2. store
        self.store = SqliteStore(data_dir / "linc.db")
        await self.store.open()

        # 3. hub (shared infrastructure for adapters)
        self.hub = Hub(http=HttpxClient())
        await self.hub.startup()

        # 4. adapters
        try:
            await self._build_adapters()
        except Exception:
            # Tear down anything we already brought up.
            await self._teardown_partial()
            raise

        # 5. dispatcher
        self._stop_event.clear()
        self._dispatcher_task = asyncio.create_task(
            self._dispatch_loop(), name="linc-dispatcher"
        )
        log.info(
            "🎉 linc-gateway ready: data_dir=%s adapters=%s pid=%d",
            data_dir, sorted(self.adapters), os.getpid(),
        )

    async def stop(self) -> None:
        """Idempotent shutdown."""
        log.info("linc-gateway stopping")
        self._stop_event.set()

        if self._dispatcher_task is not None:
            self._dispatcher_task.cancel()
            try:
                await self._dispatcher_task
            except (asyncio.CancelledError, Exception):
                pass
            self._dispatcher_task = None

        # Stop adapters in reverse start order. Best-effort.
        for name, adapter in reversed(list(self.adapters.items())):
            try:
                await asyncio.wait_for(adapter.stop(), timeout=ADAPTER_STOP_TIMEOUT)
            except TimeoutError:
                log.warning(
                    "adapter %s stop() timed out after %.1fs; continuing shutdown",
                    name,
                    ADAPTER_STOP_TIMEOUT,
                )
            except Exception:
                log.exception("adapter %s stop() failed", name)
        self.adapters.clear()

        if self.hub is not None:
            await self.hub.shutdown()
            self.hub = None

        if self.store is not None:
            await self.store.close()
            self.store = None

        if self._lock_fd is not None:
            release(self._lock_fd)
            self._lock_fd = None
        log.info("linc-gateway stopped")

    async def run_forever(self) -> None:
        """Block until `stop()` is called or the dispatcher dies."""
        if self._dispatcher_task is None:
            raise RuntimeError("gateway not started; call start() first")
        try:
            await self._dispatcher_task
        except asyncio.CancelledError:
            pass

    # ------------------------------------------------------------------ internals

    async def _build_adapters(self) -> None:
        """Instantiate adapters listed in linc.yaml. Validate each's config."""
        assert self.store is not None and self.hub is not None
        total = len(self.config.adapters)
        if total == 0:
            log.warning("⚠️ no adapters configured; gateway will only run the outbox dispatcher")
            return

        log.info("🧩 starting %d adapter(s): %s", total, list(self.config.adapters))
        for index, (platform, raw) in enumerate(self.config.adapters.items(), start=1):
            started_at = time.perf_counter()
            log.info("🔌 [%d/%d] starting adapter: %s", index, total, platform)
            try:
                adapter_cls = get_adapter_cls(platform)
            except Exception as e:
                log.error("❌ [%d/%d] adapter %s not found: %s", index, total, platform, e)
                raise ConfigError(
                    f"linc.yaml: adapters.{platform!r} -> {e}"
                ) from e
            cfg_cls = getattr(adapter_cls, "Config", None)
            if cfg_cls is None:
                log.error("❌ [%d/%d] adapter %s has no Config class", index, total, platform)
                raise ConfigError(
                    f"adapter {platform!r} ({adapter_cls.__qualname__}) "
                    f"is missing a Config class attribute"
                )
            try:
                cfg = cfg_cls.model_validate(raw or {})
            except Exception as e:
                log.error("❌ [%d/%d] adapter %s config invalid: %s", index, total, platform, e)
                raise ConfigError(
                    f"linc.yaml: adapters.{platform}: {e}"
                ) from e
            adapter = adapter_cls(cfg, self.hub, self.store)
            try:
                await adapter.start()
            except Exception:
                elapsed = time.perf_counter() - started_at
                log.exception(
                    "❌ [%d/%d] adapter %s failed after %.2fs",
                    index,
                    total,
                    platform,
                    elapsed,
                )
                # Roll back any adapters already started in this loop.
                for n, a in reversed(list(self.adapters.items())):
                    try:
                        await a.stop()
                    except Exception:
                        log.exception("adapter %s stop() failed during rollback", n)
                self.adapters.clear()
                raise
            self.adapters[platform] = adapter
            elapsed = time.perf_counter() - started_at
            log.info("✅ [%d/%d] adapter %s started in %.2fs", index, total, platform, elapsed)

    async def _teardown_partial(self) -> None:
        """Best-effort teardown when start() fails partway through."""
        for name, adapter in reversed(list(self.adapters.items())):
            try:
                await adapter.stop()
            except Exception:
                log.exception("adapter %s stop() failed during partial teardown", name)
        self.adapters.clear()
        if self.hub is not None:
            await self.hub.shutdown()
            self.hub = None
        if self.store is not None:
            await self.store.close()
            self.store = None
        if self._lock_fd is not None:
            release(self._lock_fd)
            self._lock_fd = None

    async def _dispatch_loop(self) -> None:
        """Poll the outbox; for each `pending` row, call adapter.send."""
        assert self.store is not None
        tick = max(self.config.poll_interval_ms, 10) / 1000.0
        while not self._stop_event.is_set():
            try:
                await self._dispatch_once()
            except Exception:
                log.exception("dispatcher iteration failed")
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=tick)
                return  # stop signaled
            except asyncio.TimeoutError:
                continue

    async def _dispatch_once(self) -> None:
        """Drain pending messages for every adapter, mark sent/failed."""
        assert self.store is not None
        for platform, adapter in self.adapters.items():
            pendings = await self.store.list_pending(platform)
            for msg in pendings:
                try:
                    plat_msg_id, plat_raw = await adapter.send(msg.conv_id, msg.content)
                    await self.store.mark_sent(msg.id, plat_msg_id, plat_raw)
                    text_preview = (msg.content.text or "")[:80]
                    log.info(
                        "\u2b06 [%s] %s | %s", platform, msg.conv_id, text_preview,
                    )
                except SendError as e:
                    log.warning("send failed (%s id=%d): %s", platform, msg.id, e)
                    await self.store.mark_failed(msg.id, f"{type(e).__name__}: {e}")
                except Exception as e:
                    log.exception("send raised unexpected exception (%s id=%d)", platform, msg.id)
                    await self.store.mark_failed(msg.id, f"{type(e).__name__}: {e}")

    # ------------------------------------------------------------------ factory

    @classmethod
    def from_yaml(cls, path: str | Path) -> "LincGateway":
        cfg = LincConfig.from_yaml(path)
        return cls(cfg)
