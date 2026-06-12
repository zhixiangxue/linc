"""Programmatic launch API — start gateway and return a real Client.

Usage::

    from linc import launch

    async def main():
        client = await launch("linc.yaml")
        try:
            unread = await client.pull()
            for m in unread:
                await client.send(
                    f"echo: {m.content.text}", platform=m.platform, conv_id=m.conv_id
                )
        finally:
            await client.close()

``launch()`` is only a convenience initializer. It starts ``linc serve`` as a
background subprocess, waits until the gateway reports full readiness, opens a
``Client`` connected to the shared SQLite store, and returns that Client. The
returned object is the same public Client type used when the gateway is started
manually via CLI.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

from .client import Client

STARTUP_TIMEOUT = 30.0       # seconds to wait for gateway ready
READY_POLL_INTERVAL = 0.1    # seconds between readiness checks
SHUTDOWN_TIMEOUT = 5.0       # seconds to wait after SIGTERM before SIGKILL


async def launch(config_path: str | Path) -> Client:
    """Start linc gateway as a subprocess and return a connected Client.

    The returned object is a real ``Client`` instance. It has the same message
    API as ``Client(data_dir)`` used with a manually started gateway, and also
    owns the gateway subprocess for cleanup via ``await client.close()``.

    Args:
        config_path: Path to a ``linc.yaml`` config file.

    Returns:
        An entered ``Client`` instance connected to the configured data_dir.

    Raises:
        FileNotFoundError: if ``config_path`` does not exist.
        RuntimeError: if the gateway subprocess exits during startup.
        TimeoutError: if the gateway does not become ready within 30s.
    """
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")

    data_dir = _resolve_data_dir(config_path)
    proc = _spawn_gateway(config_path)
    client = Client(
        data_dir,
        gateway_process=proc,
        gateway_shutdown_timeout=SHUTDOWN_TIMEOUT,
    )
    # Register immediately so Ctrl+C during startup still cleans the gateway.
    client._register_cleanup_handlers()

    try:
        await _wait_ready(proc, data_dir)
        await client.__aenter__()
    except Exception:
        client._unregister_cleanup_handlers()
        client._force_kill_gateway_process()
        raise
    return client


def _resolve_data_dir(config_path: Path) -> Path:
    """Resolve data_dir from linc.yaml using the same absolute-path semantics."""
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    data_dir_str: str = raw.get("data_dir", ".linc")
    if isinstance(data_dir_str, str):
        data_dir_str = os.path.expanduser(data_dir_str)
    return Path(data_dir_str).resolve()


def _spawn_gateway(config_path: Path) -> subprocess.Popen[str]:
    """Spawn `linc serve` in its own process group for clean signal routing."""
    env = os.environ.copy()
    env["LINC_LAUNCH_PROGRESS"] = "1"
    env["LINC_CONFIG_PATH"] = str(config_path)
    return subprocess.Popen(
        [sys.executable, "-m", "linc.cli", "serve", "-c", str(config_path)],
        stdout=None,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,
    )


async def _wait_ready(proc: subprocess.Popen[str], data_dir: Path) -> None:
    """Wait until gateway writes linc.ready with its PID."""
    deadline = time.monotonic() + STARTUP_TIMEOUT
    ready_path = data_dir / "linc.ready"

    while time.monotonic() < deadline:
        if proc.poll() is not None:
            stderr_tail = ""
            if proc.stderr is not None:
                stderr_tail = proc.stderr.read() or ""
            raise RuntimeError(
                f"linc gateway exited prematurely (exit code {proc.returncode}). "
                f"Stderr:\n{stderr_tail[-1000:]}"
            )

        if ready_path.exists():
            try:
                ready_pid = ready_path.read_text(encoding="utf-8").strip()
            except OSError:
                ready_pid = ""
            if ready_pid == str(proc.pid):
                return

        await asyncio.sleep(READY_POLL_INTERVAL)

    raise TimeoutError(
        f"linc gateway did not become ready within {STARTUP_TIMEOUT:.0f}s"
    )


__all__ = ["launch"]
