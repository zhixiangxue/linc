"""SQLite schema DDL + migration helpers.

Schema version is tracked in the `meta` table. Migrations only ADD COLUMN;
columns are never dropped (PRD §5.5).
"""

from __future__ import annotations

import aiosqlite

CURRENT_SCHEMA_VERSION = 1

# Pragmas applied on every connection open. WAL is required for the
# multi-process model (linc serve writes from one process while the agent
# writes/reads from another).
PRAGMAS: tuple[str, ...] = (
    "PRAGMA journal_mode = WAL",
    "PRAGMA synchronous = NORMAL",
    "PRAGMA foreign_keys = ON",
    "PRAGMA temp_store = MEMORY",
)


# Schema v1: see PRD §5.3.
# Note: SQLite UNIQUE treats multiple NULLs as distinct, so multiple outbound
# rows with msg_id=NULL (still pending) coexist without conflict — exactly what
# we need.
DDL_V1 = """
CREATE TABLE IF NOT EXISTS messages (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    platform  TEXT NOT NULL,
    conv_id   TEXT NOT NULL,
    msg_id    TEXT,
    ts        REAL NOT NULL,
    dir       TEXT NOT NULL CHECK(dir IN ('in','out')),
    status    TEXT NOT NULL,
    error     TEXT,
    sender    TEXT,
    content   TEXT NOT NULL,
    raw       TEXT,
    UNIQUE(platform, msg_id),
    CHECK(
        (dir='in'  AND status IN ('unread','read')) OR
        (dir='out' AND status IN ('pending','sent','failed'))
    )
);

CREATE INDEX IF NOT EXISTS idx_unread
    ON messages(platform, conv_id) WHERE dir='in'  AND status='unread';
CREATE INDEX IF NOT EXISTS idx_pending
    ON messages(platform, conv_id) WHERE dir='out' AND status='pending';
CREATE INDEX IF NOT EXISTS idx_conv_ts
    ON messages(platform, conv_id, ts);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


async def apply_pragmas(db: aiosqlite.Connection) -> None:
    """Apply PRAGMAs on a freshly opened connection."""
    for stmt in PRAGMAS:
        await db.execute(stmt)


async def init_schema(db: aiosqlite.Connection) -> None:
    """Create tables if absent and stamp the schema version.

    Idempotent: safe to call on every server startup.
    """
    await db.executescript(DDL_V1)
    await db.execute(
        "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)",
        (str(CURRENT_SCHEMA_VERSION),),
    )
    await db.commit()


async def get_schema_version(db: aiosqlite.Connection) -> int:
    """Return the persisted schema_version, or 0 if the meta row is absent."""
    cur = await db.execute("SELECT value FROM meta WHERE key='schema_version'")
    row = await cur.fetchone()
    await cur.close()
    return int(row[0]) if row else 0


async def migrate(db: aiosqlite.Connection) -> None:
    """Run any pending migrations from get_schema_version() to CURRENT_SCHEMA_VERSION.

    v0.1: only one schema version exists; this is a no-op past init_schema.
    Future versions append `if version < N: <ADD COLUMN ...>` blocks here.
    """
    version = await get_schema_version(db)
    if version > CURRENT_SCHEMA_VERSION:
        from .errors import StoreError

        raise StoreError(
            f"db schema_version={version} is newer than supported "
            f"({CURRENT_SCHEMA_VERSION}); refusing to downgrade"
        )
    # No upward migrations yet for v0.1.
