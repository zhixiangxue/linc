"""CLI smoke tests via typer.testing.CliRunner.

We don't test ``serve`` (long-running daemon — covered by tests/test_gateway.py)
or ``tail`` (infinite loop). The remaining commands are short-lived.

NOTE: every test in this file is **synchronous** on purpose. ``CliRunner.invoke``
starts its own ``asyncio.run`` internally; nesting that inside pytest-asyncio's
loop would crash with "asyncio.run() cannot be called from a running event loop".
We use small ``asyncio.run(...)`` helpers for store seeding/inspection.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from linc.adapters import register, unregister
from linc.cli import app
from linc.core.models import Content, Sender
from linc.core.store import SqliteStore

from _fakes.fake_adapter import FakeAdapter


runner = CliRunner()


@pytest.fixture
def fake_registered():
    register(FakeAdapter)
    yield
    unregister("fake")


# ------------------------------------------------------------------ store helpers


async def _seed_async(data_dir: Path) -> None:
    store = SqliteStore(data_dir / "linc.db")
    await store.open()
    try:
        await store.insert_inbound(
            platform="fake", conv_id="C1", msg_id="m1", ts=1000.0,
            sender=Sender(id="U1", name="Alice"),
            content=Content(text="hello"),
            raw={"text": "hello"},
        )
        await store.insert_inbound(
            platform="fake", conv_id="C1", msg_id="m2", ts=2000.0,
            sender=Sender(id="U2", name="Bob"),
            content=Content(text="world"),
            raw={"text": "world"},
        )
    finally:
        await store.close()


def _seed(data_dir: Path) -> None:
    asyncio.run(_seed_async(data_dir))


async def _list_pending_async(data_dir: Path):
    store = SqliteStore(data_dir / "linc.db")
    await store.open()
    try:
        return await store.list_pending("fake")
    finally:
        await store.close()


# ------------------------------------------------------------------ help


def test_top_level_help_lists_all_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("serve", "unread", "history", "send", "tail", "status"):
        assert cmd in result.stdout


# ------------------------------------------------------------------ unread


def test_unread_lists_messages(tmp_path: Path):
    _seed(tmp_path)
    result = runner.invoke(app, ["unread", "-d", str(tmp_path)])
    assert result.exit_code == 0, result.stdout
    assert "Alice" in result.stdout and "hello" in result.stdout
    assert "Bob" in result.stdout and "world" in result.stdout


def test_unread_does_not_consume(tmp_path: Path):
    _seed(tmp_path)
    runner.invoke(app, ["unread", "-d", str(tmp_path)])
    # Second call still sees them — peek must not mark read.
    result = runner.invoke(app, ["unread", "-d", str(tmp_path)])
    assert "Alice" in result.stdout


def test_unread_json_emits_one_per_line(tmp_path: Path):
    _seed(tmp_path)
    result = runner.invoke(app, ["unread", "-d", str(tmp_path), "--json"])
    assert result.exit_code == 0
    lines = [l for l in result.stdout.splitlines() if l.strip()]
    assert len(lines) == 2
    for l in lines:
        obj = json.loads(l)
        assert obj["platform"] == "fake"


def test_unread_missing_db_errors(tmp_path: Path):
    result = runner.invoke(app, ["unread", "-d", str(tmp_path)])
    assert result.exit_code == 1
    assert "no SQLite file" in result.stderr or "no SQLite file" in result.stdout


# ------------------------------------------------------------------ history


def test_history_filters_by_conv(tmp_path: Path):
    _seed(tmp_path)
    result = runner.invoke(
        app, ["history", "-p", "fake", "-C", "C1", "-d", str(tmp_path)]
    )
    assert result.exit_code == 0
    assert "hello" in result.stdout and "world" in result.stdout


def test_history_unknown_conv_returns_empty(tmp_path: Path):
    _seed(tmp_path)
    result = runner.invoke(
        app, ["history", "-p", "fake", "-C", "nope", "-d", str(tmp_path)]
    )
    assert result.exit_code == 0
    assert result.stdout.strip() == ""


# ------------------------------------------------------------------ send


def test_send_enqueues_pending_outbound(tmp_path: Path, fake_registered):
    _seed(tmp_path)
    result = runner.invoke(
        app, ["send", "fake", "C1", "echo: hi", "-d", str(tmp_path)]
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "queued outbound id=" in result.stdout

    rows = asyncio.run(_list_pending_async(tmp_path))
    assert len(rows) == 1
    assert rows[0].content.text == "echo: hi"


def test_send_unknown_platform_errors(tmp_path: Path):
    # No adapter registered; CLI should surface AttributeError as a clean exit.
    result = runner.invoke(
        app, ["send", "wxchat", "C1", "hi", "-d", str(tmp_path)]
    )
    assert result.exit_code == 1
    out = result.stderr + result.stdout
    assert "unknown IM platform" in out


# ------------------------------------------------------------------ status


def test_status_reports_not_running_when_clean(tmp_path: Path):
    # data_dir exists, no linc.pid → not running.
    result = runner.invoke(app, ["status", "-d", str(tmp_path)])
    assert result.exit_code == 0
    assert "not running" in result.stdout


def test_status_reports_missing_data_dir(tmp_path: Path):
    missing = tmp_path / "doesnotexist"
    result = runner.invoke(app, ["status", "-d", str(missing)])
    assert result.exit_code == 2
    assert "not found" in result.stdout
