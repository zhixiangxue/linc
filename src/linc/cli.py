"""Command-line entry point: ``linc serve / unread / history / send / tail / status``.

Wired to the ``linc`` console script via ``[project.scripts]`` in pyproject.toml.

All async commands wrap a small ``_run`` coroutine in ``asyncio.run``. We do NOT
share an event loop across commands; each invocation is a fresh process.

Read-only commands (unread, history, status, tail) only need ``--data-dir`` and
talk to SQLite directly. ``serve`` needs ``--config`` (the YAML) since adapter
instantiation lives there. ``send`` goes through the agent SDK so it acquires
``agent.lock`` like any other agent process.
"""

from __future__ import annotations

import asyncio
import json as _json
import os
import signal
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import typer

from .client import Linc
from .core.errors import AlreadyRunning, LincError
from .core.models import InboundMessage, OutboundMessage
from .core.store import SqliteStore
from .gateway import LincGateway

app = typer.Typer(
    name="linc",
    help="Linc — IM Gateway daemon for LLM Agents.",
    no_args_is_help=True,
    add_completion=False,
)

DEFAULT_DATA_DIR = Path(".linc")
DEFAULT_CONFIG = Path("linc.yaml")


# ============================================================================
# helpers
# ============================================================================


def _err(msg: str) -> None:
    typer.echo(f"linc: {msg}", err=True)


def _die(msg: str, code: int = 1) -> None:
    _err(msg)
    raise typer.Exit(code)


def _fmt_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _fmt_inbound(m: InboundMessage) -> str:
    sender = m.sender.name or m.sender.id
    text = m.content.text or ""
    if m.content.attachments:
        text += f"  [+{len(m.content.attachments)} attach]"
    return f"[{_fmt_ts(m.ts)}] {m.platform}/{m.conv_id} <{sender}> {text}"


def _fmt_outbound(m: OutboundMessage) -> str:
    text = m.content.text or ""
    err = f"  ERR: {m.error}" if m.error else ""
    return f"[{_fmt_ts(m.ts)}] {m.platform}/{m.conv_id} → ({m.status}) {text}{err}"


def _fmt_msg(m: InboundMessage | OutboundMessage) -> str:
    return _fmt_inbound(m) if isinstance(m, InboundMessage) else _fmt_outbound(m)


def _emit(rows: list[Any], as_json: bool) -> None:
    if as_json:
        for r in rows:
            typer.echo(r.model_dump_json())
    else:
        for r in rows:
            typer.echo(_fmt_msg(r))


async def _open_store(data_dir: Path) -> SqliteStore:
    db_path = data_dir / "linc.db"
    if not db_path.exists():
        _die(f"no SQLite file at {db_path}; has linc serve ever been started here?")
    store = SqliteStore(db_path)
    await store.open()
    return store


# ============================================================================
# linc serve
# ============================================================================


def _check_yaml_perms(path: Path) -> None:
    """Warn (don't fail) if linc.yaml is world/group readable — it holds creds."""
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
    except OSError:
        return
    if mode & 0o077:
        _err(
            f"warning: {path} mode is {oct(mode)}; contains plaintext credentials. "
            f"Run: chmod 600 {path}"
        )


@app.command()
def serve(
    config: Path = typer.Option(DEFAULT_CONFIG, "--config", "-c", help="linc.yaml path"),
) -> None:
    """Start the gateway daemon (single-instance per data_dir)."""
    if not config.exists():
        _die(f"config not found: {config}")
    _check_yaml_perms(config)

    try:
        gateway = LincGateway.from_yaml(config)
    except LincError as e:
        _die(str(e))

    async def _run() -> None:
        loop = asyncio.get_running_loop()
        stop = asyncio.Event()

        def _stop_handler() -> None:
            stop.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _stop_handler)
            except NotImplementedError:
                pass  # Windows; Ctrl-C will raise KeyboardInterrupt instead

        try:
            await gateway.start()
        except AlreadyRunning as e:
            _die(str(e))
        except LincError as e:
            _die(str(e))

        typer.echo(f"linc serve: started (pid={os.getpid()}); Ctrl-C to stop")
        try:
            stop_task = asyncio.create_task(stop.wait())
            fwd_task = asyncio.create_task(gateway.run_forever())
            done, pending = await asyncio.wait(
                {stop_task, fwd_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
        finally:
            await gateway.stop()
            typer.echo("linc serve: stopped")

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


# ============================================================================
# linc unread / history / send
# ============================================================================


@app.command()
def unread(
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir", "-d"),
    platform: str | None = typer.Option(None, "--platform", "-p"),
    conv_id: str | None = typer.Option(None, "--conv", "-C"),
    limit: int | None = typer.Option(None, "--limit", "-n"),
    json_out: bool = typer.Option(False, "--json", help="Emit one JSON object per line"),
) -> None:
    """Peek at unread inbound messages (does NOT mark them read)."""

    async def _run() -> None:
        store = await _open_store(data_dir)
        try:
            rows = await store.list_unread(platform=platform, conv_id=conv_id, limit=limit)
        finally:
            await store.close()
        _emit(rows, json_out)

    asyncio.run(_run())


@app.command()
def history(
    platform: str = typer.Option(..., "--platform", "-p"),
    conv_id: str | None = typer.Option(None, "--conv", "-C"),
    n: int = typer.Option(50, "-n", "--limit", help="max rows"),
    since: float | None = typer.Option(None, "--since", help="unix epoch seconds"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir", "-d"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Show inbound + outbound history for a conversation."""

    async def _run() -> None:
        store = await _open_store(data_dir)
        try:
            rows = await store.history(
                platform=platform, conv_id=conv_id, since=since, limit=n,
            )
        finally:
            await store.close()
        _emit(rows, json_out)

    asyncio.run(_run())


@app.command()
def send(
    platform: str = typer.Argument(..., help="IM platform name (must be registered)"),
    conv_id: str = typer.Argument(..., help="conversation id"),
    text: str = typer.Argument(..., help="message text"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir", "-d"),
) -> None:
    """Enqueue a single outbound message (acquires agent.lock; gateway will deliver)."""

    async def _run() -> None:
        try:
            async with Linc(data_dir) as linc:
                client = getattr(linc, platform)()
                row_id = await client.send(text, conv_id=conv_id)
        except AlreadyRunning as e:
            _die(str(e))
        except AttributeError as e:
            _die(str(e))
        typer.echo(f"queued outbound id={row_id} platform={platform} conv={conv_id}")

    asyncio.run(_run())


# ============================================================================
# linc tail
# ============================================================================


@app.command()
def tail(
    platform: str | None = typer.Option(None, "--platform", "-p"),
    conv_id: str | None = typer.Option(None, "--conv", "-C"),
    backfill: int = typer.Option(0, "-n", help="show last N rows before tailing"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir", "-d"),
    interval_ms: int = typer.Option(200, "--interval", help="poll interval (ms)"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Stream new messages as they arrive (polls every --interval ms)."""

    async def _run() -> None:
        store = await _open_store(data_dir)
        try:
            seen_ids: set[int] = set()
            if backfill > 0:
                rows = await store.history(
                    platform=platform, conv_id=conv_id, limit=backfill,
                )
                _emit(rows, json_out)
                seen_ids.update(r.id for r in rows)
            else:
                # Seed seen_ids with whatever's already there so we only print NEW rows.
                rows = await store.history(
                    platform=platform, conv_id=conv_id, limit=10_000,
                )
                seen_ids.update(r.id for r in rows)

            tick = max(interval_ms, 50) / 1000.0
            while True:
                await asyncio.sleep(tick)
                rows = await store.history(
                    platform=platform, conv_id=conv_id, limit=500,
                )
                new = [r for r in rows if r.id not in seen_ids]
                if new:
                    _emit(new, json_out)
                    seen_ids.update(r.id for r in new)
        finally:
            await store.close()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


# ============================================================================
# linc status
# ============================================================================


@app.command()
def status(
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir", "-d"),
) -> None:
    """Report whether a gateway is currently running on the given data_dir."""
    pid_path = data_dir / "linc.pid"
    db_path = data_dir / "linc.db"
    if not data_dir.exists():
        typer.echo(f"data_dir not found: {data_dir}")
        raise typer.Exit(2)

    typer.echo(f"data_dir: {data_dir.resolve()}")
    typer.echo(f"db:       {'present' if db_path.exists() else 'missing'}")

    if not pid_path.exists():
        typer.echo("gateway:  not running (no linc.pid)")
        return

    # Try to grab the lock non-blocking. If it fails, someone holds it.
    import fcntl

    try:
        fd = os.open(pid_path, os.O_RDWR)
    except OSError as e:
        typer.echo(f"gateway:  ?? could not open linc.pid: {e}")
        return
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pid_text = ""
            try:
                pid_text = pid_path.read_text().strip()
            except OSError:
                pass
            typer.echo(f"gateway:  RUNNING (pid={pid_text or '?'})")
            return
        # We got the lock — nobody else holds it.
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        typer.echo("gateway:  not running (linc.pid stale)")
    finally:
        os.close(fd)


if __name__ == "__main__":
    app()
