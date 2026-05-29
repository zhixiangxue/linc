"""SqliteStore: every SQL statement linc executes lives here.

Design notes:
  - One persistent aiosqlite connection per process. WAL mode allows the agent
    process and the linc-serve process to keep their own connections in
    parallel without blocking each other.
  - JSON columns (`sender`, `content`, `raw`) are serialized via pydantic's
    `model_dump_json` (for Content/Sender) and json.dumps (for arbitrary
    platform raw dicts). Reads round-trip through pydantic for validation.
  - `claim_unread` runs inside a single `BEGIN IMMEDIATE` transaction so the
    SELECT and UPDATE are atomic — no read-then-mark window. This is the
    primary defense against accidental double-handling even before the agent
    flock kicks in.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import aiosqlite

from .errors import StoreError
from .models import (
    Content,
    InboundMessage,
    OutboundDraft,
    OutboundMessage,
    Sender,
)
from .schema import apply_pragmas, init_schema, migrate

# Column order used by SELECT * — must match DDL_V1.
_COLS = (
    "id",
    "platform",
    "conv_id",
    "msg_id",
    "ts",
    "dir",
    "status",
    "error",
    "sender",
    "content",
    "raw",
)


class SqliteStore:
    """Async wrapper around a single aiosqlite.Connection."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self._db: aiosqlite.Connection | None = None
        # Serializes write operations on the single underlying connection.
        # aiosqlite already runs SQL on one worker thread, but multi-statement
        # blocks (BEGIN/SELECT/UPDATE/COMMIT in claim_unread) need their own
        # mutual exclusion to prevent two coroutines from interleaving a
        # second BEGIN inside an open transaction.
        self._write_lock: asyncio.Lock | None = None

    # ------------------------------------------------------------------ lifecycle

    async def open(self) -> None:
        """Open the connection, apply pragmas, ensure schema is up to date."""
        if self._db is not None:
            return
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        db = await aiosqlite.connect(self.db_path)
        # Row factory gives us name-based access (row["platform"]).
        db.row_factory = aiosqlite.Row
        await apply_pragmas(db)
        await init_schema(db)
        await migrate(db)
        self._db = db
        self._write_lock = asyncio.Lock()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None
        self._write_lock = None

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise StoreError("SqliteStore is not open; call open() first")
        return self._db

    @property
    def _lock(self) -> asyncio.Lock:
        if self._write_lock is None:
            raise StoreError("SqliteStore is not open; call open() first")
        return self._write_lock

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _row_to_inbound(row: aiosqlite.Row, *, override_status: str | None = None) -> InboundMessage:
        if row["sender"] is None:
            raise StoreError(f"inbound row id={row['id']} has NULL sender")
        return InboundMessage(
            id=row["id"],
            platform=row["platform"],
            conv_id=row["conv_id"],
            msg_id=row["msg_id"],
            ts=row["ts"],
            status=override_status or row["status"],  # type: ignore[arg-type]
            sender=Sender.model_validate_json(row["sender"]),
            content=Content.model_validate_json(row["content"]),
            raw=json.loads(row["raw"]) if row["raw"] is not None else {},
        )

    @staticmethod
    def _row_to_outbound(row: aiosqlite.Row) -> OutboundMessage:
        return OutboundMessage(
            id=row["id"],
            platform=row["platform"],
            conv_id=row["conv_id"],
            msg_id=row["msg_id"],
            ts=row["ts"],
            status=row["status"],  # type: ignore[arg-type]
            error=row["error"],
            content=Content.model_validate_json(row["content"]),
            raw=json.loads(row["raw"]) if row["raw"] is not None else None,
        )

    @staticmethod
    def _row_to_either(row: aiosqlite.Row) -> InboundMessage | OutboundMessage:
        if row["dir"] == "in":
            return SqliteStore._row_to_inbound(row)
        return SqliteStore._row_to_outbound(row)

    # ------------------------------------------------------------------ inbound (server)

    async def insert_inbound(
        self,
        platform: str,
        conv_id: str,
        msg_id: str,
        ts: float,
        sender: Sender,
        content: Content,
        raw: dict[str, Any],
    ) -> int | None:
        """Insert an inbound message. Returns the new row id, or None on duplicate.

        Duplicates are detected via UNIQUE(platform, msg_id) and silently ignored
        (PRD §10) — IM platforms occasionally redeliver the same event.
        """
        async with self._lock:
            cur = await self.db.execute(
                """
                INSERT OR IGNORE INTO messages
                    (platform, conv_id, msg_id, ts, dir, status, sender, content, raw)
                VALUES (?, ?, ?, ?, 'in', 'unread', ?, ?, ?)
                """,
                (
                    platform,
                    conv_id,
                    msg_id,
                    ts,
                    sender.model_dump_json(),
                    content.model_dump_json(),
                    json.dumps(raw, ensure_ascii=False),
                ),
            )
            await self.db.commit()
            # cur.lastrowid is undefined when IGNORE swallowed the insert; detect via rowcount.
            row_id = cur.lastrowid if cur.rowcount > 0 else None
            await cur.close()
            return row_id

    # ------------------------------------------------------------------ inbound (agent)

    async def claim_unread(
        self,
        platform: str | None = None,
        conv_id: str | None = None,
        limit: int | None = None,
    ) -> list[InboundMessage]:
        """Atomically read and mark-as-read in a single transaction.

        Returns the claimed messages with status='read' (already updated).
        """
        where = ["dir='in'", "status='unread'"]
        params: list[Any] = []
        if platform is not None:
            where.append("platform=?")
            params.append(platform)
        if conv_id is not None:
            where.append("conv_id=?")
            params.append(conv_id)
        where_sql = " AND ".join(where)
        limit_sql = f" LIMIT {int(limit)}" if limit else ""

        db = self.db
        async with self._lock:
            # BEGIN IMMEDIATE acquires the write lock right away, so a concurrent
            # writer cannot sneak in between our SELECT and UPDATE.
            await db.execute("BEGIN IMMEDIATE")
            try:
                cur = await db.execute(
                    f"SELECT {', '.join(_COLS)} FROM messages "
                    f"WHERE {where_sql} ORDER BY ts ASC, id ASC{limit_sql}",
                    params,
                )
                rows = await cur.fetchall()
                await cur.close()
                if rows:
                    ids = [r["id"] for r in rows]
                    placeholders = ",".join("?" * len(ids))
                    await db.execute(
                        f"UPDATE messages SET status='read' WHERE id IN ({placeholders})",
                        ids,
                    )
                await db.commit()
            except Exception:
                await db.rollback()
                raise

        return [self._row_to_inbound(r, override_status="read") for r in rows]

    # ------------------------------------------------------------------ outbound (agent)

    async def enqueue_outbound(
        self,
        platform: str,
        draft: OutboundDraft,
        ts: float,
    ) -> int:
        """Persist a pending outbound message; returns the new row id."""
        async with self._lock:
            cur = await self.db.execute(
                """
                INSERT INTO messages
                    (platform, conv_id, msg_id, ts, dir, status, content, raw)
                VALUES (?, ?, NULL, ?, 'out', 'pending', ?, NULL)
                """,
                (platform, draft.conv_id, ts, draft.content.model_dump_json()),
            )
            await self.db.commit()
            row_id = cur.lastrowid
            await cur.close()
        if row_id is None:
            raise StoreError("INSERT did not return a lastrowid")
        return row_id

    # ------------------------------------------------------------------ outbound (server)

    async def list_pending(self, platform: str) -> list[OutboundMessage]:
        cur = await self.db.execute(
            f"SELECT {', '.join(_COLS)} FROM messages "
            "WHERE dir='out' AND status='pending' AND platform=? "
            "ORDER BY ts ASC, id ASC",
            (platform,),
        )
        rows = await cur.fetchall()
        await cur.close()
        return [self._row_to_outbound(r) for r in rows]

    async def mark_sent(self, id: int, msg_id: str, raw: dict[str, Any]) -> None:
        async with self._lock:
            await self.db.execute(
                "UPDATE messages SET status='sent', msg_id=?, raw=? WHERE id=? AND dir='out'",
                (msg_id, json.dumps(raw, ensure_ascii=False), id),
            )
            await self.db.commit()

    async def mark_failed(self, id: int, error: str) -> None:
        async with self._lock:
            await self.db.execute(
                "UPDATE messages SET status='failed', error=? WHERE id=? AND dir='out'",
                (error, id),
            )
            await self.db.commit()

    # ------------------------------------------------------------------ generic

    async def history(
        self,
        platform: str | None = None,
        conv_id: str | None = None,
        since: float | None = None,
        until: float | None = None,
        limit: int = 100,
    ) -> list[InboundMessage | OutboundMessage]:
        clauses: list[str] = []
        params: list[Any] = []
        if platform is not None:
            clauses.append("platform=?")
            params.append(platform)
        if conv_id is not None:
            clauses.append("conv_id=?")
            params.append(conv_id)
        if since is not None:
            clauses.append("ts >= ?")
            params.append(since)
        if until is not None:
            clauses.append("ts < ?")
            params.append(until)
        where_sql = ("WHERE " + " AND ".join(clauses)) if clauses else ""

        cur = await self.db.execute(
            f"SELECT {', '.join(_COLS)} FROM messages {where_sql} "
            "ORDER BY ts ASC, id ASC LIMIT ?",
            (*params, int(limit)),
        )
        rows = await cur.fetchall()
        await cur.close()
        return [self._row_to_either(r) for r in rows]

    async def list_unread(
        self,
        platform: str | None = None,
        conv_id: str | None = None,
        limit: int | None = None,
    ) -> list[InboundMessage]:
        """Read-only peek at unread messages; does NOT mark them read.

        Used by `linc unread` CLI for human inspection. Agents should use
        `claim_unread` instead.
        """
        where = ["dir='in'", "status='unread'"]
        params: list[Any] = []
        if platform is not None:
            where.append("platform=?")
            params.append(platform)
        if conv_id is not None:
            where.append("conv_id=?")
            params.append(conv_id)
        where_sql = " AND ".join(where)
        limit_sql = f" LIMIT {int(limit)}" if limit else ""
        cur = await self.db.execute(
            f"SELECT {', '.join(_COLS)} FROM messages "
            f"WHERE {where_sql} ORDER BY ts ASC, id ASC{limit_sql}",
            params,
        )
        rows = await cur.fetchall()
        await cur.close()
        return [self._row_to_inbound(r) for r in rows]

    # ------------------------------------------------------------------ wakeup

    async def data_version(self) -> int:
        """Return PRAGMA data_version. Spike-verified for cross-process change detection.

        On a long-lived connection in WAL mode, this value increases whenever
        ANY OTHER connection commits a write — exactly what the outbox dispatcher
        needs as a near-zero-cost wakeup signal.
        """
        cur = await self.db.execute("PRAGMA data_version")
        row = await cur.fetchone()
        await cur.close()
        if row is None:
            raise StoreError("PRAGMA data_version returned no row")
        return int(row[0])
