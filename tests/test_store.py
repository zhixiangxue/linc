"""SqliteStore unit tests: CRUD, dedup, claim_unread atomicity, data_version."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import aiosqlite
import pytest

from linc.core.models import Attachment, Content, OutboundDraft, Sender
from linc.core.store import SqliteStore


def _make_sender(uid: str = "U1", name: str = "alice") -> Sender:
    return Sender(id=uid, name=name)


def _make_content(text: str = "hi") -> Content:
    return Content(text=text)


# ---------------------------------------------------------------------------
# inbound
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_insert_inbound_returns_id(store: SqliteStore) -> None:
    rid = await store.insert_inbound(
        platform="slack",
        conv_id="C1",
        msg_id="m1",
        ts=time.time(),
        sender=_make_sender(),
        content=_make_content("hello"),
        raw={"type": "message"},
    )
    assert rid is not None
    assert rid > 0


@pytest.mark.asyncio
async def test_insert_inbound_dedup_returns_none(store: SqliteStore) -> None:
    """Duplicate (platform, msg_id) -> INSERT OR IGNORE -> None."""
    args = dict(
        platform="slack",
        conv_id="C1",
        msg_id="m1",
        ts=time.time(),
        sender=_make_sender(),
        content=_make_content("hi"),
        raw={"x": 1},
    )
    first = await store.insert_inbound(**args)
    second = await store.insert_inbound(**args)
    assert first is not None
    assert second is None


@pytest.mark.asyncio
async def test_claim_unread_returns_and_marks_read(store: SqliteStore) -> None:
    now = time.time()
    for i in range(3):
        await store.insert_inbound(
            platform="slack",
            conv_id="C1",
            msg_id=f"m{i}",
            ts=now + i,
            sender=_make_sender(),
            content=_make_content(f"msg{i}"),
            raw={},
        )
    msgs = await store.claim_unread(platform="slack")
    assert len(msgs) == 3
    assert all(m.status == "read" for m in msgs)
    assert [m.content.text for m in msgs] == ["msg0", "msg1", "msg2"]

    # Second claim returns nothing — they were marked read atomically.
    again = await store.claim_unread(platform="slack")
    assert again == []


@pytest.mark.asyncio
async def test_claim_unread_filters_by_conv_and_limit(store: SqliteStore) -> None:
    now = time.time()
    for i, conv in enumerate(["C1", "C2", "C1", "C2"]):
        await store.insert_inbound(
            platform="slack",
            conv_id=conv,
            msg_id=f"m{i}",
            ts=now + i,
            sender=_make_sender(),
            content=_make_content(f"x{i}"),
            raw={},
        )
    only_c1 = await store.claim_unread(platform="slack", conv_id="C1")
    assert [m.conv_id for m in only_c1] == ["C1", "C1"]

    # Remaining C2 messages are still unread; limit=1 should claim just one.
    one = await store.claim_unread(platform="slack", limit=1)
    assert len(one) == 1
    assert one[0].conv_id == "C2"


@pytest.mark.asyncio
async def test_claim_unread_atomic_under_contention(store: SqliteStore) -> None:
    """Two concurrent claim_unread calls on the SAME connection must split
    the rows (no message claimed twice). With BEGIN IMMEDIATE the second call
    will queue behind the first and see no unread rows."""
    now = time.time()
    for i in range(10):
        await store.insert_inbound(
            platform="slack",
            conv_id="C1",
            msg_id=f"m{i}",
            ts=now + i,
            sender=_make_sender(),
            content=_make_content(f"x{i}"),
            raw={},
        )
    a, b = await asyncio.gather(
        store.claim_unread(platform="slack"),
        store.claim_unread(platform="slack"),
    )
    ids_a = {m.id for m in a}
    ids_b = {m.id for m in b}
    assert ids_a.isdisjoint(ids_b)
    assert len(ids_a) + len(ids_b) == 10


@pytest.mark.asyncio
async def test_claim_unread_round_trips_attachments(store: SqliteStore) -> None:
    content = Content(
        text="see this",
        attachments=[Attachment(kind="image", url="https://x/y.png", mime="image/png")],
    )
    await store.insert_inbound(
        platform="slack",
        conv_id="C1",
        msg_id="m1",
        ts=time.time(),
        sender=_make_sender(),
        content=content,
        raw={"raw_key": "raw_val"},
    )
    msgs = await store.claim_unread()
    assert len(msgs) == 1
    m = msgs[0]
    assert m.content == content
    assert m.raw == {"raw_key": "raw_val"}
    assert m.sender.name == "alice"


# ---------------------------------------------------------------------------
# outbound
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enqueue_and_list_pending(store: SqliteStore) -> None:
    draft = OutboundDraft(conv_id="C1", content=_make_content("send me"))
    rid = await store.enqueue_outbound("slack", draft, ts=time.time())
    assert rid > 0

    pending = await store.list_pending("slack")
    assert len(pending) == 1
    assert pending[0].id == rid
    assert pending[0].status == "pending"
    assert pending[0].msg_id is None
    assert pending[0].raw is None
    assert pending[0].content.text == "send me"


@pytest.mark.asyncio
async def test_mark_sent_transitions_state(store: SqliteStore) -> None:
    draft = OutboundDraft(conv_id="C1", content=_make_content("hi"))
    rid = await store.enqueue_outbound("slack", draft, ts=time.time())
    await store.mark_sent(rid, msg_id="ts.123", raw={"ok": True, "ts": "ts.123"})

    pending = await store.list_pending("slack")
    assert pending == []
    hist = await store.history(platform="slack")
    assert len(hist) == 1
    sent = hist[0]
    assert sent.status == "sent"
    assert sent.msg_id == "ts.123"
    assert sent.raw == {"ok": True, "ts": "ts.123"}


@pytest.mark.asyncio
async def test_mark_failed_records_error(store: SqliteStore) -> None:
    draft = OutboundDraft(conv_id="C1", content=_make_content("hi"))
    rid = await store.enqueue_outbound("slack", draft, ts=time.time())
    await store.mark_failed(rid, error="rate_limited")

    pending = await store.list_pending("slack")
    assert pending == []
    hist = await store.history(platform="slack")
    assert hist[0].status == "failed"
    assert hist[0].error == "rate_limited"


@pytest.mark.asyncio
async def test_multiple_pending_no_unique_conflict(store: SqliteStore) -> None:
    """Two pending outbound rows have msg_id=NULL each; UNIQUE(platform,msg_id)
    must not reject them (SQLite treats multiple NULLs as distinct)."""
    draft = OutboundDraft(conv_id="C1", content=_make_content("hi"))
    a = await store.enqueue_outbound("slack", draft, ts=time.time())
    b = await store.enqueue_outbound("slack", draft, ts=time.time() + 0.01)
    assert a != b
    pending = await store.list_pending("slack")
    assert {m.id for m in pending} == {a, b}


# ---------------------------------------------------------------------------
# data_version (cross-connection wakeup signal)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_data_version_increments_on_other_connection_write(
    store: SqliteStore, tmp_path: Path
) -> None:
    """The fixture's connection should observe data_version increments triggered
    by writes from a *separate* aiosqlite connection — the basis of the outbox
    dispatcher's wakeup mechanism."""
    v0 = await store.data_version()

    async with aiosqlite.connect(store.db_path) as other:
        await other.execute("PRAGMA journal_mode = WAL")
        await other.execute(
            """INSERT INTO messages
               (platform, conv_id, msg_id, ts, dir, status, sender, content, raw)
               VALUES ('slack','C1','x',1.0,'in','unread','{}','{}','{}')"""
        )
        await other.commit()

    v1 = await store.data_version()
    assert v1 != v0


# ---------------------------------------------------------------------------
# history
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_history_returns_both_directions_in_order(store: SqliteStore) -> None:
    base = 1000.0
    await store.insert_inbound(
        platform="slack", conv_id="C1", msg_id="m1", ts=base + 1,
        sender=_make_sender(), content=_make_content("in1"), raw={},
    )
    await store.enqueue_outbound(
        "slack", OutboundDraft(conv_id="C1", content=_make_content("out1")), ts=base + 2,
    )
    await store.insert_inbound(
        platform="slack", conv_id="C1", msg_id="m2", ts=base + 3,
        sender=_make_sender(), content=_make_content("in2"), raw={},
    )

    hist = await store.history(platform="slack", conv_id="C1")
    assert [m.content.text for m in hist] == ["in1", "out1", "in2"]
