"""End-to-end tests: a real ``LincGateway`` running alongside a real ``Client``
agent client in the same process, mediated only by SQLite + the two flocks.

These tests exist to verify the **architectural promise**:
  - linc.pid (gateway) and client.lock (client) are independent — the two
    processes are designed to coexist on the same data_dir.
  - The full inbound path works: adapter.inject -> store -> client.read_unread.
  - The full outbound path works: client.send -> outbox -> dispatcher ->
    adapter.send -> store.mark_sent.
  - An echo loop completes within a couple of dispatcher ticks.

We use ``FakeAdapter`` so no real network is involved. Real adapter (Slack)
correctness is the responsibility of t11.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from linc import Client
from linc.adapters import register, unregister
from linc.core.config import LincConfig
from linc.gateway import LincGateway

from _fakes.fake_adapter import FakeAdapter


# ------------------------------------------------------------------ fixtures


@pytest.fixture
def fake_registered():
    register(FakeAdapter)
    yield
    unregister("fake")


@pytest.fixture
def cfg(tmp_path: Path) -> LincConfig:
    return LincConfig(
        data_dir=tmp_path,
        poll_interval_ms=20,  # tighten for fast e2e
        adapters={"fake": {}},
    )


async def _wait_until(predicate, timeout: float = 2.0, tick: float = 0.02) -> None:
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while True:
        result = predicate()
        if asyncio.iscoroutine(result):
            result = await result
        if result:
            return
        if loop.time() > deadline:
            raise AssertionError(f"timeout waiting for {predicate}")
        await asyncio.sleep(tick)


# ------------------------------------------------------------------ coexistence


async def test_gateway_and_client_locks_coexist(cfg, fake_registered):
    """linc.pid and client.lock are independent flocks on different files."""
    gateway = LincGateway(cfg)
    await gateway.start()
    try:
        # Agent client must be able to enter while gateway runs.
        async with Client(cfg.data_dir):
            assert (cfg.data_dir / "linc.pid").exists()
            assert (cfg.data_dir / "client.lock").exists()
    finally:
        await gateway.stop()


# ------------------------------------------------------------------ inbound path: adapter -> client


async def test_inbound_adapter_to_client(cfg, fake_registered):
    """adapter.inject seeds the store; client.read_unread claims it."""
    gateway = LincGateway(cfg)
    await gateway.start()
    try:
        adapter = gateway.adapters["fake"]
        await adapter.inject(
            {
                "conv_id": "C1",
                "msg_id": "m-1",
                "ts": 1000.0,
                "sender": {"id": "U1", "name": "Alice"},
                "text": "ping",
            }
        )
        async with Client(cfg.data_dir) as client:
            msgs = await client.fake.read_unread()
            assert [m.content.text for m in msgs] == ["ping"]
            # Claimed exactly once.
            again = await client.fake.read_unread()
            assert again == []
    finally:
        await gateway.stop()


# ------------------------------------------------------------------ outbound path: client -> adapter


async def test_outbound_client_to_adapter(cfg, fake_registered):
    """client.send -> outbox -> dispatcher -> adapter.send."""
    gateway = LincGateway(cfg)
    await gateway.start()
    try:
        adapter = gateway.adapters["fake"]
        async with Client(cfg.data_dir) as client:
            row_id = await client.fake.send("hello there", conv_id="C42")
            assert row_id > 0

        # The dispatcher (poll_interval_ms=20) should pick it up almost instantly.
        await _wait_until(lambda: len(adapter.sent) == 1, timeout=2.0)
        conv_id, content = adapter.sent[0]
        assert conv_id == "C42"
        assert content.text == "hello there"

        # The store row should be marked 'sent' with msg_id + raw populated.
        rows = await gateway.store.history(platform="fake")
        assert len(rows) == 1
        sent = rows[0]
        assert sent.id == row_id
        assert sent.status == "sent"
        assert sent.msg_id and sent.msg_id.startswith("fake-")
    finally:
        await gateway.stop()


# ------------------------------------------------------------------ full echo loop


async def test_echo_agent_loop(cfg, fake_registered):
    """The canonical demo: an agent that echoes every inbound back to the same conv.

    Verifies the entire two-way roundtrip in a single test:
        adapter.inject -> store.in -> agent.read_unread -> agent.send
                       -> store.out -> dispatcher -> adapter.send
    """
    gateway = LincGateway(cfg)
    await gateway.start()
    try:
        adapter = gateway.adapters["fake"]
        stop = asyncio.Event()

        async def echo_agent() -> None:
            async with Client(cfg.data_dir) as client:
                fake = client.fake
                # Tight loop for the test — production agents would back off.
                while not stop.is_set():
                    msgs = await fake.read_unread()
                    for m in msgs:
                        await fake.send(
                            f"echo: {m.content.text}", conv_id=m.conv_id
                        )
                    try:
                        await asyncio.wait_for(stop.wait(), timeout=0.02)
                        return
                    except asyncio.TimeoutError:
                        continue

        agent_task = asyncio.create_task(echo_agent())
        try:
            # Inject two messages on different conversations.
            await adapter.inject(
                {
                    "conv_id": "C1", "msg_id": "in-1", "ts": 1.0,
                    "sender": {"id": "U1", "name": "Alice"},
                    "text": "hello",
                }
            )
            await adapter.inject(
                {
                    "conv_id": "C2", "msg_id": "in-2", "ts": 2.0,
                    "sender": {"id": "U2", "name": "Bob"},
                    "text": "world",
                }
            )

            await _wait_until(lambda: len(adapter.sent) == 2, timeout=3.0)

            sent_by_conv = {conv: content.text for conv, content in adapter.sent}
            assert sent_by_conv == {
                "C1": "echo: hello",
                "C2": "echo: world",
            }
        finally:
            stop.set()
            await asyncio.wait_for(agent_task, timeout=2.0)
    finally:
        await gateway.stop()


# ------------------------------------------------------------------ history reflects both sides


async def test_history_contains_both_directions_after_loop(cfg, fake_registered):
    gateway = LincGateway(cfg)
    await gateway.start()
    try:
        adapter = gateway.adapters["fake"]
        await adapter.inject(
            {
                "conv_id": "C1", "msg_id": "in-1", "ts": 1.0,
                "sender": {"id": "U1", "name": "Alice"},
                "text": "ping",
            }
        )
        async with Client(cfg.data_dir) as client:
            msgs = await client.fake.read_unread()
            assert msgs[0].content.text == "ping"
            await client.fake.send("pong", conv_id="C1")

        await _wait_until(lambda: len(adapter.sent) == 1, timeout=2.0)

        rows = await gateway.store.history(platform="fake", conv_id="C1")
        # Two rows: one inbound (status=read), one outbound (status=sent).
        assert len(rows) == 2
        statuses = sorted(r.status for r in rows)
        assert statuses == ["read", "sent"]
        texts = [r.content.text for r in rows]
        assert texts == ["ping", "pong"]
    finally:
        await gateway.stop()
