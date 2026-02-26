from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from langchain.schema import Document
from langchain.docstore.base import AddableMixin


class SqliteDocstore(AddableMixin):
    """Thread-safe, WAL-mode SQLite docstore — drop-in for InMemoryDocstore.

    Each thread gets its own connection so concurrent inserts from
    the producer/embed/insert pipeline never block each other.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._local = threading.local()
        # Initialise schema on the calling thread
        conn = self._conn()
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-32768")  # ~32 MB page cache
        conn.execute(
            "CREATE TABLE IF NOT EXISTS docs "
            "(uid TEXT PRIMARY KEY, content TEXT, metadata TEXT)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS id_map "
            "(pos INTEGER PRIMARY KEY, uid TEXT NOT NULL)"
        )
        conn.commit()

    # ── Internal ─────────────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        """Return (or create) a per-thread SQLite connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            c = sqlite3.connect(self._db_path, check_same_thread=False)
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=NORMAL")
            c.execute("PRAGMA cache_size=-32768")
            self._local.conn = c
        return self._local.conn

    # ── Public API ───────────────────────────────────────────────────────

    def add(self, docs: dict[str, Document]) -> None:
        conn = self._conn()
        conn.executemany(
            "INSERT OR REPLACE INTO docs VALUES (?, ?, ?)",
            [
                (uid, doc.page_content, json.dumps(doc.metadata or {}))
                for uid, doc in docs.items()
            ],
        )
        conn.commit()

    def search(self, uid: str) -> Document | str:
        row = self._conn().execute(
            "SELECT content, metadata FROM docs WHERE uid=?", (uid,)
        ).fetchone()
        if row is None:
            return f"ID {uid} not found"
        return Document(page_content=row[0], metadata=json.loads(row[1]))

    def __len__(self) -> int:
        return self._conn().execute("SELECT COUNT(*) FROM docs").fetchone()[0]

    # ── ID map (integer FAISS position ↔ docstore UUID) ──────────────────

    def save_id_map(self, id_map: dict[int, str]) -> None:
        """Persist the FAISS integer→UUID mapping into the id_map table."""
        conn = self._conn()
        conn.execute("DELETE FROM id_map")
        conn.executemany(
            "INSERT INTO id_map (pos, uid) VALUES (?, ?)",
            id_map.items(),
        )
        conn.commit()

    def load_id_map(self) -> dict[int, str]:
        """Return the stored integer→UUID mapping (empty dict if not yet saved)."""
        rows = self._conn().execute("SELECT pos, uid FROM id_map ORDER BY pos").fetchall()
        return {pos: uid for pos, uid in rows}