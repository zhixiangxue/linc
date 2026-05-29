"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest_asyncio

from linc.core.store import SqliteStore


@pytest_asyncio.fixture
async def store(tmp_path: Path) -> SqliteStore:
    """A fresh SqliteStore backed by a tmp_path file. Auto-closed at teardown."""
    s = SqliteStore(tmp_path / "messages.db")
    await s.open()
    try:
        yield s
    finally:
        await s.close()
