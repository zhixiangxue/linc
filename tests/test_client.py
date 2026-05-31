"""Tests for the agent-side ``Linc`` / ``Client`` SDK.

The client does NOT spin up adapters — it only talks to SQLite. So these tests
seed inbound rows directly via the store (simulating what the gateway would do
when an adapter receives a real IM event), and read the outbox after `send()`
to verify what the gateway would dispatch.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from linc import Attachment, Content, Linc
from linc.adapters import register, unregister
from linc.core.errors import AlreadyRunning
from linc.core.models import Sender
from linc.core.store import SqliteStore

from _fakes.fake_adapter import FakeAdapter


# ------------------------------------------------------------------ fixtures


@pytest.fixture
def fake_registered():
    register(FakeAdapter)
    yield
    unregister("fake")


async def _seed_inbound(
    data_dir: Path,
    *,
    platform: str = "fake",
    conv_id: str = "C1",
    msg_id: str = "m-1",
    sender_name: str = "Alice",
    text: str = "hello",
    ts: float = 1000.0,
) -> int:
    """Insert one inbound row directly via the store (simulating the adapter)."""
    store = SqliteStore(data_dir / "linc.db")
    await store.open()
    try:
        return await store.insert_inbound(
            platform=platform,
            conv_id=conv_id,
            msg_id=msg_id,
            ts=ts,
            sender=Sender(id="U1", name=sender_name),
            content=Content(text=text),
            raw={"text": text},
        )
    finally:
        await store.close()


# ------------------------------------------------------------------ lifecycle / lock


async def test_enter_exit_acquires_and_releases_agent_lock(tmp_path: Path):
    async with Linc(tmp_path) as linc:
        assert linc.store is not None
        assert (tmp_path / "agent.lock").exists()
    # After exit, a fresh Linc must be able to re-acquire.
    async with Linc(tmp_path):
        pass


async def test_second_linc_blocked_by_agent_lock(tmp_path: Path):
    async with Linc(tmp_path):
        with pytest.raises(AlreadyRunning):
            async with Linc(tmp_path):
                pass


async def test_store_property_raises_when_not_entered(tmp_path: Path):
    linc = Linc(tmp_path)
    with pytest.raises(RuntimeError):
        _ = linc.store


# ------------------------------------------------------------------ platform factory


async def test_unknown_platform_raises_attribute_error(tmp_path: Path, fake_registered):
    async with Linc(tmp_path) as linc:
        with pytest.raises(AttributeError, match="unknown IM platform"):
            linc.wxchat()


async def test_dunder_attributes_not_intercepted(tmp_path: Path):
    """`hasattr(linc, '__copy__')` must NOT explode in __getattr__."""
    async with Linc(tmp_path) as linc:
        assert not hasattr(linc, "__copy__")
        assert not hasattr(linc, "_definitely_private")


async def test_registered_platform_returns_handle(tmp_path: Path, fake_registered):
    async with Linc(tmp_path) as linc:
        fake = linc.get("fake")
        assert fake.name == "fake"
        # Bind conv at factory time.
        chat = linc.get("fake", conv_id="C42")
        assert chat.name == "fake"
        assert chat._conv_id == "C42"


async def test_dynamic_platform_factory_remains_compat_alias(tmp_path: Path, fake_registered):
    async with Linc(tmp_path) as linc:
        fake = linc.fake()
        assert fake.name == "fake"
        chat = linc.fake(conv_id="C42")
        assert chat._conv_id == "C42"


async def test_get_unknown_platform_raises_value_error(tmp_path: Path, fake_registered):
    async with Linc(tmp_path) as linc:
        with pytest.raises(ValueError, match="unknown IM platform"):
            linc.get("wxchat")


# ------------------------------------------------------------------ send


async def test_send_string_enqueues_pending_outbound(tmp_path: Path, fake_registered):
    async with Linc(tmp_path) as linc:
        row_id = await linc.get("fake").send("hello world", conv_id="C1")
        assert row_id > 0
        rows = await linc.store.list_pending("fake")
        assert len(rows) == 1
        assert rows[0].id == row_id
        assert rows[0].conv_id == "C1"
        assert rows[0].content.text == "hello world"


async def test_send_content_with_attachments(tmp_path: Path, fake_registered):
    payload = Content(
        text="see image",
        attachments=[Attachment(kind="image", url="https://x/y.png")],
    )
    async with Linc(tmp_path) as linc:
        await linc.get("fake").send(payload, conv_id="C1")
        rows = await linc.store.list_pending("fake")
        assert len(rows) == 1
        assert rows[0].content.text == "see image"
        assert len(rows[0].content.attachments) == 1
        assert rows[0].content.attachments[0].url == "https://x/y.png"


async def test_send_uses_bound_conv_id(tmp_path: Path, fake_registered):
    async with Linc(tmp_path) as linc:
        chat = linc.get("fake", conv_id="C-bound")
        await chat.send("hi")
        rows = await linc.store.list_pending("fake")
        assert rows[0].conv_id == "C-bound"


async def test_send_method_arg_overrides_bound_conv_id(tmp_path: Path, fake_registered):
    async with Linc(tmp_path) as linc:
        chat = linc.get("fake", conv_id="C-bound")
        await chat.send("hi", conv_id="C-override")
        rows = await linc.store.list_pending("fake")
        assert rows[0].conv_id == "C-override"


async def test_send_without_conv_id_raises(tmp_path: Path, fake_registered):
    async with Linc(tmp_path) as linc:
        with pytest.raises(ValueError, match="conv_id"):
            await linc.get("fake").send("hi")


async def test_conv_chain_returns_new_handle(tmp_path: Path, fake_registered):
    async with Linc(tmp_path) as linc:
        base = linc.get("fake")
        chat = base.conv("C-chain")
        assert base._conv_id is None
        assert chat._conv_id == "C-chain"
        await chat.send("hi")
        rows = await linc.store.list_pending("fake")
        assert rows[0].conv_id == "C-chain"


# ------------------------------------------------------------------ read_unread


async def test_read_unread_claims_messages(tmp_path: Path, fake_registered):
    await _seed_inbound(tmp_path, conv_id="C1", msg_id="m1", text="hi", ts=1.0)
    await _seed_inbound(tmp_path, conv_id="C1", msg_id="m2", text="ho", ts=2.0)
    async with Linc(tmp_path) as linc:
        msgs = await linc.fake().read_unread()
        assert [m.content.text for m in msgs] == ["hi", "ho"]
        # Second call returns nothing — they were claimed.
        again = await linc.fake().read_unread()
        assert again == []


async def test_read_unread_respects_bound_conv_id(tmp_path: Path, fake_registered):
    await _seed_inbound(tmp_path, conv_id="C1", msg_id="m1", text="a", ts=1.0)
    await _seed_inbound(tmp_path, conv_id="C2", msg_id="m2", text="b", ts=2.0)
    async with Linc(tmp_path) as linc:
        msgs = await linc.fake(conv_id="C2").read_unread()
        assert [m.content.text for m in msgs] == ["b"]
        # C1 still claimable.
        leftover = await linc.fake().read_unread()
        assert [m.content.text for m in leftover] == ["a"]


async def test_list_unread_does_not_claim(tmp_path: Path, fake_registered):
    await _seed_inbound(tmp_path, msg_id="m1", text="peek", ts=1.0)
    async with Linc(tmp_path) as linc:
        peek = await linc.fake().list_unread()
        assert len(peek) == 1
        # Still claimable afterwards.
        msgs = await linc.fake().read_unread()
        assert len(msgs) == 1


# ------------------------------------------------------------------ cross-platform helpers


async def test_read_unread_all_spans_platforms(tmp_path: Path, fake_registered):
    # Register a second fake platform under a different name to verify the
    # cross-platform claim is truly cross-platform.
    class FakeTwo(FakeAdapter):
        name = "fake2"

    register(FakeTwo)
    try:
        await _seed_inbound(tmp_path, platform="fake", msg_id="m1", text="a", ts=1.0)
        await _seed_inbound(tmp_path, platform="fake2", msg_id="m2", text="b", ts=2.0)
        async with Linc(tmp_path) as linc:
            msgs = await linc.read_unread_all()
            assert [m.platform for m in msgs] == ["fake", "fake2"]
    finally:
        unregister("fake2")


async def test_history_returns_inbound_and_outbound(tmp_path: Path, fake_registered):
    await _seed_inbound(tmp_path, msg_id="m1", text="in", ts=1.0)
    async with Linc(tmp_path) as linc:
        await linc.fake().send("out", conv_id="C1")
        rows = await linc.fake().history()
        assert len(rows) == 2
        # Sorted by ts asc — inbound (ts=1.0) comes before outbound (now).
        assert rows[0].content.text == "in"
        assert rows[1].content.text == "out"
