"""End-to-end test for LincGateway using FakeAdapter.

Covers the full gateway-side cycle:
  - flock acquired and released
  - adapter instantiated, started, stopped
  - dispatcher picks up `pending` rows and calls adapter.send
  - successful send marks row 'sent' with platform msg_id + raw
  - send raising an exception marks row 'failed' with error message
  - second start in same data_dir blocked by AlreadyRunning
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from linc.adapters import register, unregister
from linc.core.config import LincConfig
from linc.core.errors import AlreadyRunning, SendError
from linc.core.models import Content, OutboundDraft
from linc.gateway import LincGateway

from _fakes.fake_adapter import FakeAdapter


@pytest.fixture
def fake_registered():
    register(FakeAdapter)
    yield
    unregister("fake")


@pytest.fixture
def cfg(tmp_path: Path) -> LincConfig:
    return LincConfig(
        data_dir=tmp_path,
        poll_interval_ms=20,  # tighten for fast test
        adapters={"fake": {}},
    )


async def _wait_until(predicate, timeout: float = 2.0, tick: float = 0.02) -> None:
    """Poll `predicate` (sync or async) until it returns truthy or timeout."""
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        result = predicate()
        if asyncio.iscoroutine(result):
            result = await result
        if result:
            return
        if asyncio.get_event_loop().time() > deadline:
            raise AssertionError(f"timeout waiting for {predicate}")
        await asyncio.sleep(tick)


# ---------------------------------------------------------------- happy path


async def test_dispatcher_sends_pending_outbound(cfg, fake_registered):
    gateway = LincGateway(cfg)
    await gateway.start()
    try:
        # Enqueue an outbound message via store (simulating client side).
        store = gateway.store
        assert store is not None
        msg_id = await store.enqueue_outbound(
            platform="fake",
            draft=OutboundDraft(conv_id="C123", content=Content(text="hi")),
            ts=1000.0,
        )

        adapter = gateway.adapters["fake"]
        await _wait_until(lambda: len(adapter.sent) == 1)

        # The fake adapter recorded one send.
        conv_id, content = adapter.sent[0]
        assert conv_id == "C123"
        assert content.text == "hi"

        # The row is now 'sent' with msg_id + raw populated.
        rows = await store.history(platform="fake")
        assert len(rows) == 1
        sent_row = rows[0]
        assert sent_row.id == msg_id
        assert sent_row.status == "sent"
        assert sent_row.msg_id and sent_row.msg_id.startswith("fake-")
        assert sent_row.raw == {"ok": True, "conv_id": "C123", "msg_id": sent_row.msg_id}
    finally:
        await gateway.stop()


# ---------------------------------------------------------------- failure path


async def test_dispatcher_marks_failed_on_send_exception(cfg, fake_registered, monkeypatch):
    gateway = LincGateway(cfg)
    await gateway.start()
    try:
        async def boom(self, conv_id, content):
            raise SendError("boom")

        # Patch the live instance, not the class, to keep other tests untouched.
        adapter = gateway.adapters["fake"]
        monkeypatch.setattr(adapter, "send", boom.__get__(adapter))

        store = gateway.store
        assert store is not None
        await store.enqueue_outbound(
            platform="fake",
            draft=OutboundDraft(conv_id="C9", content=Content(text="x")),
            ts=2000.0,
        )

        async def is_failed():
            rows = await store.history(platform="fake")
            return rows and rows[0].status == "failed"

        await _wait_until(is_failed)

        rows = await store.history(platform="fake")
        assert rows[0].status == "failed"
        assert rows[0].error and "boom" in rows[0].error
    finally:
        await gateway.stop()


# ---------------------------------------------------------------- inbound via adapter helper


async def test_inbound_event_persisted_and_claimable(cfg, fake_registered):
    gateway = LincGateway(cfg)
    await gateway.start()
    try:
        adapter = gateway.adapters["fake"]
        await adapter.inject(
            {
                "conv_id": "C1",
                "msg_id": "m-1",
                "ts": 3000.0,
                "sender": {"id": "U1", "name": "Alice"},
                "text": "hello",
            }
        )
        store = gateway.store
        assert store is not None
        unread = await store.list_unread(platform="fake")
        assert len(unread) == 1
        assert unread[0].sender.name == "Alice"
        assert unread[0].content.text == "hello"
    finally:
        await gateway.stop()


# ---------------------------------------------------------------- single-instance lock


async def test_second_gateway_blocked_by_flock(cfg, fake_registered):
    a = LincGateway(cfg)
    await a.start()
    try:
        b = LincGateway(cfg)
        with pytest.raises(AlreadyRunning):
            await b.start()
    finally:
        await a.stop()


# ---------------------------------------------------------------- bad config


async def test_unknown_platform_raises_config_error(tmp_path):
    from linc.core.errors import ConfigError

    cfg = LincConfig(
        data_dir=tmp_path,
        poll_interval_ms=20,
        adapters={"never-registered": {}},
    )
    gateway = LincGateway(cfg)
    with pytest.raises(ConfigError):
        await gateway.start()
    # Lock must have been released so a second attempt would not be blocked.
    assert (tmp_path / "linc.pid").exists()
    # And we can re-acquire (simulate a second gateway attempt).
    gateway2 = LincGateway(LincConfig(data_dir=tmp_path, adapters={}))
    await gateway2.start()
    await gateway2.stop()
