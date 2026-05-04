"""Thread-safe SQLite docstore for FAISS — drop-in for InMemoryDocstore."""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Iterator

from langchain.schema import Document
from langchain.docstore.base import AddableMixin


class SqliteDocstore(AddableMixin):
    """Thread-safe, WAL-mode SQLite docstore — drop-in for InMemoryDocstore.

    Optimized for memory-efficient batch operations:
    - WAL mode for concurrent reads/writes
    - Per-thread connections (no locking overhead)
    - Batch inserts with configurable commit frequency
    - Streaming iteration (no full load into RAM)
    """

    def __init__(
        self,
        db_path: str | Path,
        *,
        batch_commit_size: int = 1000,
    ) -> None:
        """Initialize the SQLite docstore.

        Args:
            db_path: Path to the SQLite database file.
            batch_commit_size: Commit after this many inserts in batch mode.
        """
        self._db_path = str(db_path)
        self._local = threading.local()
        self._batch_commit_size = batch_commit_size
        self._pending_count = 0

        # Initialise schema on the calling thread
        conn = self._conn()
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-32768")  # ~32 MB page cache
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA mmap_size=268435456")  # 256MB mmap
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
            c.execute("PRAGMA temp_store=MEMORY")
            c.execute("PRAGMA mmap_size=268435456")
            self._local.conn = c
        return self._local.conn

    # ── Public API ───────────────────────────────────────────────────────

    def add(self, docs: dict[str, Document]) -> None:
        """Add documents to the store (batch-optimized)."""
        conn = self._conn()
        conn.executemany(
            "INSERT OR REPLACE INTO docs VALUES (?, ?, ?)",
            [
                (uid, doc.page_content, json.dumps(doc.metadata or {}))
                for uid, doc in docs.items()
            ],
        )
        self._pending_count += len(docs)

        # Commit in batches for better performance
        if self._pending_count >= self._batch_commit_size:
            conn.commit()
            self._pending_count = 0
        # Note: Final commit happens in save_id_map or flush()

    def flush(self) -> None:
        """Force commit any pending inserts."""
        if self._pending_count > 0:
            self._conn().commit()
            self._pending_count = 0

    def search(self, uid: str) -> Document | str:
        """Search for a document by UID."""
        row = self._conn().execute(
            "SELECT content, metadata FROM docs WHERE uid=?", (uid,)
        ).fetchone()
        if row is None:
            return f"ID {uid} not found"
        return Document(page_content=row[0], metadata=json.loads(row[1]))

    def __len__(self) -> int:
        return self._conn().execute("SELECT COUNT(*) FROM docs").fetchone()[0]

    def iter_documents(self, batch_size: int = 1000) -> Iterator[Document]:
        """Iterate over all documents without loading everything into RAM.

        Args:
            batch_size: Number of documents to fetch per database query.

        Yields:
            Document objects one at a time.
        """
        conn = self._conn()
        offset = 0

        while True:
            rows = conn.execute(
                "SELECT content, metadata FROM docs LIMIT ? OFFSET ?",
                (batch_size, offset),
            ).fetchall()

            if not rows:
                break

            for content, metadata in rows:
                yield Document(
                    page_content=content,
                    metadata=json.loads(metadata),
                )

            offset += len(rows)

            # If we got fewer rows than batch_size, we're done
            if len(rows) < batch_size:
                break

    # ── ID map (integer FAISS position ↔ docstore UUID) ──────────────────

    def save_id_map(self, id_map: dict[int, str]) -> None:
        """Persist the FAISS integer→UUID mapping into the id_map table.

        Replaces the entire table — use ``append_id_map`` for incremental
        inserts during a running migration.
        """
        self.flush()
        conn = self._conn()
        conn.execute("DELETE FROM id_map")
        conn.executemany(
            "INSERT INTO id_map (pos, uid) VALUES (?, ?)",
            id_map.items(),
        )
        conn.commit()

    def append_id_map(self, entries: dict[int, str]) -> None:
        """Append new FAISS position→UID entries without rewriting the table.

        This is the preferred method during active indexing. Call
        ``save_id_map`` only when you need a full replace (e.g. after a
        training phase that reassigns positions).

        Args:
            entries: Dict mapping FAISS integer positions to docstore UIDs.
        """
        conn = self._conn()
        conn.executemany(
            "INSERT OR REPLACE INTO id_map (pos, uid) VALUES (?, ?)",
            entries.items(),
        )
        conn.commit()

    def load_id_map(self) -> dict[int, str]:
        """Return the stored integer→UUID mapping (empty dict if not yet saved)."""
        rows = self._conn().execute("SELECT pos, uid FROM id_map ORDER BY pos").fetchall()
        return {pos: uid for pos, uid in rows}

    def get_stats(self) -> dict[str, int]:
        """Return database statistics for debugging."""
        conn = self._conn()
        doc_count = conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
        id_map_count = conn.execute("SELECT COUNT(*) FROM id_map").fetchone()[0]
        page_count = conn.execute("PRAGMA page_count").fetchone()[0]
        page_size = conn.execute("PRAGMA page_size").fetchone()[0]

        return {
            "doc_count": doc_count,
            "id_map_count": id_map_count,
            "db_size_bytes": page_count * page_size,
            "db_size_mb": (page_count * page_size) / (1024 * 1024),
        }
