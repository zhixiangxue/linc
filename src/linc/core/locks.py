"""Single-instance flock utilities (POSIX: Linux/macOS).

Two locks live under data_dir:
  - linc.pid    : held by `linc serve` for as long as the daemon (LincGateway) runs.
  - agent.lock  : held by the agent process inside `async with Linc(...)`.

Both use fcntl.flock(LOCK_EX | LOCK_NB). Acquisition is non-blocking; if another
process holds the lock we raise AlreadyRunning immediately rather than queueing.

Windows is intentionally not supported in v0.1 (see PRD §18, item 5).
"""

from __future__ import annotations

import fcntl
import os
from pathlib import Path

from .errors import AlreadyRunning


def _acquire(lock_path: Path, label: str) -> int:
    """Acquire an exclusive non-blocking flock on `lock_path`.

    Returns the open file descriptor on success. The fd MUST be retained by the
    caller until release(); closing the fd implicitly releases the lock, so we
    do NOT use a context manager here.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as e:
        os.close(fd)
        raise AlreadyRunning(
            f"{label} lock at {lock_path} is held by another process"
        ) from e
    # Best-effort: write our PID into the lock file for human inspection.
    # Truncate first so a smaller PID does not leave trailing bytes from a prior run.
    try:
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
    except OSError:
        # PID write is purely informational; the lock itself is what matters.
        pass
    return fd


def release(fd: int) -> None:
    """Release the flock and close the descriptor. Idempotent-safe at shutdown."""
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


def acquire_gateway_lock(data_dir: Path) -> int:
    """Acquire the linc-gateway single-instance lock at <data_dir>/linc.pid."""
    return _acquire(Path(data_dir) / "linc.pid", "linc-gateway")


def acquire_agent_lock(data_dir: Path) -> int:
    """Acquire the agent single-instance lock at <data_dir>/agent.lock."""
    return _acquire(Path(data_dir) / "agent.lock", "linc-agent")
