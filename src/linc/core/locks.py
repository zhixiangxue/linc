"""Single-instance lock utilities (cross-platform: Linux/macOS/Windows).

Two locks live under data_dir:
  - linc.pid    : held by `linc serve` for as long as the daemon (LincGateway) runs.
  - agent.lock  : held by the agent process inside `async with Linc(...)`.

On POSIX systems we use fcntl.flock(LOCK_EX | LOCK_NB).
On Windows we use msvcrt.locking(LK_NBLCK).
Acquisition is non-blocking; if another process holds the lock we raise
AlreadyRunning immediately rather than queueing.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from .errors import AlreadyRunning

# --- Platform-specific lock/unlock primitives ---

if sys.platform == "win32":
    import msvcrt

    _LOCK_LEN = 1  # lock the first byte

    def _platform_lock(fd: int) -> None:
        """Non-blocking exclusive lock (Windows)."""
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, _LOCK_LEN)
        except (OSError, IOError) as e:
            raise BlockingIOError(str(e)) from e

    def _platform_unlock(fd: int) -> None:
        """Unlock (Windows). Seek to 0 first since msvcrt locks by position."""
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, _LOCK_LEN)
        except OSError:
            pass

else:
    import fcntl

    def _platform_lock(fd: int) -> None:
        """Non-blocking exclusive lock (POSIX)."""
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _platform_unlock(fd: int) -> None:
        """Unlock (POSIX)."""
        fcntl.flock(fd, fcntl.LOCK_UN)


# --- Public API ---


def _acquire(lock_path: Path, label: str) -> int:
    """Acquire an exclusive non-blocking lock on `lock_path`.

    Returns the open file descriptor on success. The fd MUST be retained by the
    caller until release(); closing the fd implicitly releases the lock, so we
    do NOT use a context manager here.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        _platform_lock(fd)
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
    """Release the lock and close the descriptor. Idempotent-safe at shutdown."""
    try:
        _platform_unlock(fd)
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
