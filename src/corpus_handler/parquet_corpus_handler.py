"""Parquet-backed corpus handler.

Wraps a corpus stored as a single parquet file and implements all three
:class:`~src.corpus_handler.base.CorpusHandler` capabilities:

* :meth:`get_documents` — point-lookup of rows by ``wikipedia_id``,
  returned as LangChain :class:`~langchain.schema.Document` objects.
* :meth:`get_boundaries` — returns ``(boundaries_uw, boundaries_cw)``
  arrays, loading from a ``metadata.json`` cache when available and
  falling back to a full corpus scan otherwise.
* :meth:`create_metadata` — scans the corpus, computes both boundary
  flavours plus per-decile stats, and writes a ``metadata.json`` file
  compatible with :func:`~src.metrics.decile_utils.load_boundaries_from_metadata`.
"""

from __future__ import annotations

import gc
import json
import logging
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from langchain.schema import Document

from src.corpus_handler.base import CorpusHandler
from src.metrics.decile_utils import (
    boundaries_to_metadata,
    compute_corpus_boundaries,
    load_boundaries_from_metadata,
    COL_POPULARITY,
)

logger = logging.getLogger(__name__)

_BATCH_SIZE = 50_000


class ParquetCorpusHandler(CorpusHandler):
    """Corpus handler backed by a single parquet file.

    The parquet must contain at least the columns ``wikipedia_id`` (int),
    ``text`` (str), and ``popularity_avg`` (float).  All other columns are
    surfaced as document metadata.

    Args:
        corpus_path: Path to the corpus ``.parquet`` file.
        metadata_path: Optional path to a pre-computed ``metadata.json``
            file.  When provided and the file exists, :meth:`get_boundaries`
            skips the corpus scan entirely.  If *not* provided, defaults to
            ``corpus_path.parent / "metadata.json"``.
        chunk_size: Token/char chunk size forwarded to
            :func:`~src.metrics.decile_utils.compute_corpus_boundaries` when
            computing chunk-weighted boundaries.
        chunk_overlap: Overlap forwarded to the same function.
    """

    def __init__(
        self,
        corpus_path: str | Path,
        *,
        metadata_path: str | Path | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
    ) -> None:
        self.corpus_path: Path = Path(corpus_path)
        self.metadata_path: Path = (
            Path(metadata_path)
            if metadata_path is not None
            else self.corpus_path.parent / "metadata.json"
        )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Cache so repeated calls to get_boundaries() skip re-loading
        self._boundaries_uw: np.ndarray | None = None
        self._boundaries_cw: np.ndarray | None = None

    # ── Document retrieval ─────────────────────────────────────────────────

    def get_documents(self, wikipedia_ids: int | list[int]) -> list[Document]:
        """Return :class:`~langchain.schema.Document` objects for the given IDs.

        Streams the corpus in batches and collects all rows whose
        ``wikipedia_id`` matches one of the requested IDs.  The document
        ``page_content`` is taken from the ``text`` column; all other
        columns are placed in ``metadata`` alongside ``wikipedia_id`` and
        ``popularity_avg``.

        Args:
            wikipedia_ids: A single ``wikipedia_id`` or a list of them.

        Returns:
            List of matching :class:`~langchain.schema.Document` instances.
            Empty list if none are found.
        """
        if isinstance(wikipedia_ids, int):
            wikipedia_ids = [wikipedia_ids]

        target_ids: set[int] = set(wikipedia_ids)
        results: list[Document] = []

        pf = pq.ParquetFile(self.corpus_path)
        for batch in pf.iter_batches(batch_size=_BATCH_SIZE):
            wid_col = batch.column("wikipedia_id").to_pylist()
            for row_idx, wid in enumerate(wid_col):
                if wid is None:
                    continue
                if int(wid) not in target_ids:
                    continue

                # Build metadata from all columns
                meta: dict = {
                    name: batch.column(name)[row_idx].as_py()
                    for name in batch.schema.names
                    if name != "text"
                }
                meta["wikipedia_id"] = int(wid)

                text_col = batch.column("text") if "text" in batch.schema.names else None
                page_content = (
                    str(text_col[row_idx].as_py() or "")
                    if text_col is not None
                    else ""
                )

                results.append(Document(page_content=page_content, metadata=meta))

            # Early exit once all requested IDs have been found
            if len(results) >= len(target_ids):
                break

        return results

    # ── Decile boundaries ──────────────────────────────────────────────────

    def get_boundaries(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(boundaries_uw, boundaries_cw)`` arrays.

        Loads from :attr:`metadata_path` when the file exists; falls back
        to a full corpus scan via :meth:`create_metadata` (which also
        persists the result for future calls).

        Returns:
            ``(boundaries_uw, boundaries_cw)`` — each an ``np.ndarray``
            of shape ``(n_deciles + 1,)``.
        """
        if self._boundaries_uw is not None and self._boundaries_cw is not None:
            return self._boundaries_uw, self._boundaries_cw

        if self.metadata_path.exists():
            logger.info("Loading decile boundaries from %s", self.metadata_path)
            uw, cw, _ = load_boundaries_from_metadata(self.metadata_path)
        else:
            logger.info(
                "metadata.json not found at %s — running corpus scan",
                self.metadata_path,
            )
            self.create_metadata(self.metadata_path)
            uw, cw, _ = load_boundaries_from_metadata(self.metadata_path)

        self._boundaries_uw = uw
        self._boundaries_cw = cw
        return uw, cw

    # ── Metadata persistence ───────────────────────────────────────────────

    def create_metadata(self, output_path: Path) -> None:
        """Scan the corpus, compute boundaries, and write ``metadata.json``.

        Uses :func:`~src.metrics.decile_utils.compute_corpus_boundaries`
        to compute both unweighted and chunk-weighted boundary arrays and
        per-decile statistics, then merges the result into a
        ``metadata.json`` at *output_path* (creating or updating the file).

        Args:
            output_path: Destination path for the ``metadata.json`` file.
                The parent directory is created if it does not exist.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Scanning corpus for decile boundaries: %s", self.corpus_path)

        boundaries_uw, boundaries_cw, stats, _ = compute_corpus_boundaries(
            self.corpus_path,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        gc.collect()

        fragment = boundaries_to_metadata(
            boundaries_uw,
            boundaries_cw,
            stats,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        # Merge into existing file if present, otherwise start fresh
        existing: dict = {}
        if output_path.exists():
            with open(output_path) as f:
                existing = json.load(f)

        existing.update(fragment)
        with open(output_path, "w") as f:
            json.dump(existing, f, indent=2)

        logger.info("Wrote metadata to %s", output_path)

        # Invalidate in-memory cache so next get_boundaries() reloads
        self._boundaries_uw = None
        self._boundaries_cw = None
