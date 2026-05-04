"""BM25-based RAG service for local keyword retrieval.

BM25 (Best Match 25) ranks documents by term frequency and inverse document
frequency — no embedding model or GPU required.  It is useful as:

- A fast, zero-cost baseline against dense retrieval.
- A component in hybrid retrieval (BM25 + vector).
- An exact-keyword retriever for domain-specific terminology.

Index lifecycle
---------------
The loaded ``BM25Index`` is held on ``self._index`` after any indexing or
``load_index`` call, so retrieval methods require no ``index`` argument.

Scores
------
``rank_bm25`` does not expose its raw BM25 scores through the LangChain
``BM25Retriever`` interface, so ``retrieve_documents_with_scores`` returns
rank-based reciprocal scores ``1 / (rank + 1)`` (best result = 1.0).
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import pyarrow.parquet as pq
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from tqdm import tqdm

from .base import IndexResult, RagService
from .base import documents_from_dataframe

logger = logging.getLogger(__name__)


# ── BM25Index wrapper ─────────────────────────────────────────────────────────

class BM25Index:
    """Thin wrapper around ``BM25Retriever`` with persistence support.

    Attributes:
        retriever: The underlying LangChain ``BM25Retriever``.
    """

    def __init__(self, retriever: BM25Retriever) -> None:
        """Initialise from an existing retriever.

        Args:
            retriever: Configured ``BM25Retriever`` instance.
        """
        self.retriever = retriever

    # ── Search ────────────────────────────────────────────────────────────

    def similarity_search(self, query: str, k: int = 5) -> list[Document]:
        """Return the top-k documents matching *query*.

        Args:
            query: Query string.
            k: Number of documents to return.

        Returns:
            Ranked list of matching ``Document`` objects.
        """
        self.retriever.k = k
        return self.retriever.invoke(query)

    def similarity_search_with_score(
        self, query: str, k: int = 5
    ) -> list[tuple[Document, float]]:
        """Return the top-k documents with rank-based reciprocal scores.

        ``rank_bm25`` does not expose raw scores through the LangChain
        retriever interface, so scores are ``1 / (rank + 1)``:
        rank 0 → 1.0, rank 1 → 0.5, rank 2 → 0.333, …

        Args:
            query: Query string.
            k: Number of results to return.

        Returns:
            List of ``(Document, score)`` tuples, best-first.
        """
        docs = self.similarity_search(query, k=k)
        return [(doc, 1.0 / (i + 1)) for i, doc in enumerate(docs)]

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def doc_count(self) -> int:
        """Number of documents in the index."""
        return len(self.retriever.docs)

    @property
    def docs(self) -> list[Document]:
        """All documents stored in the index."""
        return list(self.retriever.docs)

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Pickle the retriever to *path*.

        Args:
            path: Destination file path (typically ``*.pkl``).
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self.retriever, fh)
        logger.info(f"Saved BM25 index to {path}")

    @classmethod
    def load(cls, path: Path) -> BM25Index:
        """Load a pickled retriever from *path*.

        Args:
            path: Path to the ``.pkl`` file.

        Returns:
            Loaded ``BM25Index`` instance.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        if not path.exists():
            raise FileNotFoundError(f"BM25 index not found at {path}")
        with open(path, "rb") as fh:
            retriever = pickle.load(fh)
        logger.info(f"Loaded BM25 index from {path} ({len(retriever.docs):,} docs)")
        return cls(retriever)


# ── BM25RagService ────────────────────────────────────────────────────────────

class BM25RagService(RagService):
    """RAG service using local BM25 keyword retrieval.

    The active index is stored internally on ``self._index`` after any
    indexing or ``load_index`` call — retrieval methods do not take an
    ``index`` argument.

    Args:
        default_top_k: Default number of results for retrieval methods.
        b: BM25 length-normalisation parameter (0 = off, 1 = full).
            Defaults to ``0.75``.
        k1: BM25 term-saturation parameter (typical range 1.2–2.0).
            Defaults to ``1.5``.

    Example::

        service = BM25RagService()
        service.index_from_parquet(
            Path("data/corpus.parquet"),
            text_field="text",
            metadata_fields=["wikipedia_id", "wikipedia_title"],
            collection_name="wiki_bm25",
        )
        docs = service.retrieve_documents("Who is Reza Pahlavi?", top_k=10)
    """

    def __init__(
        self,
        *,
        default_top_k: int = 5,
        b: float = 0.75,
        k1: float = 1.5,
    ) -> None:
        self.default_top_k = default_top_k
        self.b = b
        self.k1 = k1
        self._index: BM25Index | None = None
        self._index_path: Path | None = None

    # ── Internal helpers ──────────────────────────────────────────────────

    def _require_index(self) -> BM25Index:
        """Return the current index or raise if none is loaded.

        Returns:
            The active ``BM25Index``.

        Raises:
            ValueError: If no index has been built or loaded yet.
        """
        if self._index is None:
            raise ValueError(
                "No BM25 index loaded. Call index_from_parquet(), "
                "index_from_dataframe(), or load_index() first."
            )
        return self._index

    def _build_index(
        self,
        documents: list[Document],
        output_dir: Path | None,
        collection_name: str,
    ) -> BM25Index:
        """Create and optionally persist a ``BM25Index`` from *documents*.

        Args:
            documents: Pre-built document list.
            output_dir: If provided, the index is saved here as a ``.pkl``.
            collection_name: File stem for the saved index file.

        Returns:
            The constructed ``BM25Index``.
        """
        retriever = BM25Retriever.from_documents(
            documents,
            bm25_params={"b": self.b, "k1": self.k1},
        )
        retriever.k = self.default_top_k
        index = BM25Index(retriever)

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            pkl_path = output_dir / f"{collection_name}.pkl"
            index.save(pkl_path)
            self._index_path = pkl_path

        return index

    # ── Indexing ──────────────────────────────────────────────────────────

    def index_from_dataframe(
        self,
        df: pd.DataFrame,
        text_field: str,
        *,
        metadata_fields: Sequence[str] | None = None,
        collection_name: str = "bm25",
        output_dir: Path | None = None,
        **kwargs: Any,
    ) -> IndexResult:
        """Build a BM25 index from a pandas DataFrame.

        Args:
            df: Source DataFrame.
            text_field: Column containing document text.
            metadata_fields: Extra columns to store as document metadata.
            collection_name: Index file stem (used when ``output_dir`` is set).
            output_dir: Optional directory to persist the index.
            **kwargs: Ignored (for API compatibility).

        Returns:
            ``IndexResult`` with the ``BM25Index`` and document count.
        """
        logger.info(f"Building BM25 index from DataFrame ({len(df):,} rows)")
        documents = documents_from_dataframe(df, text_field, metadata_fields)
        self._index = self._build_index(documents, output_dir, collection_name)
        logger.info(f"BM25 index ready — {self._index.doc_count:,} documents")
        return IndexResult(self._index, self._index.doc_count)

    def index_from_parquet(
        self,
        parquet_path: Path,
        *,
        text_field: str = "text",
        metadata_fields: Sequence[str] | None = None,
        collection_name: str = "bm25",
        output_dir: Path | None = None,
        **kwargs: Any,
    ) -> IndexResult:
        """Build a BM25 index by loading an entire Parquet file.

        For very large files consider ``index_from_parquet_batches`` to
        keep memory usage bounded.

        Args:
            parquet_path: Path to the ``.parquet`` file.
            text_field: Column containing document text.
            metadata_fields: Extra columns to store as document metadata.
            collection_name: Index file stem (used when ``output_dir`` is set).
            output_dir: Optional directory to persist the index.
            **kwargs: Ignored.

        Returns:
            ``IndexResult`` with the ``BM25Index`` and document count.

        Raises:
            FileNotFoundError: If ``parquet_path`` does not exist.
        """
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        columns = [text_field] + list(metadata_fields or [])
        logger.info(f"Reading {parquet_path}")
        df = pq.read_table(parquet_path, columns=columns).to_pandas()
        logger.info(f"Loaded {len(df):,} rows")
        return self.index_from_dataframe(
            df,
            text_field,
            metadata_fields=metadata_fields,
            collection_name=collection_name,
            output_dir=output_dir,
        )

    def index_from_parquet_batches(
        self,
        parquet_path: Path,
        *,
        text_field: str = "text",
        metadata_fields: Sequence[str] | None = None,
        collection_name: str = "bm25",
        output_dir: Path | None = None,
        batch_size: int = 100_000,
        skip_rows: int = 0,
        **kwargs: Any,
    ) -> IndexResult:
        """Build a BM25 index by streaming a Parquet file in batches.

        All batches are accumulated in memory before building the index —
        BM25 requires the full corpus for IDF computation.  Use
        ``batch_size`` to control how many rows are held in each Arrow
        batch; the full document list is still assembled in RAM.

        Args:
            parquet_path: Path to the ``.parquet`` file.
            text_field: Column containing document text.
            metadata_fields: Extra columns to store as metadata.
            collection_name: Index file stem (used when ``output_dir`` is set).
            output_dir: Optional directory to persist the index.
            batch_size: Rows per Arrow batch (controls peak memory per read).
            skip_rows: Skip this many leading rows.
            **kwargs: Ignored.

        Returns:
            ``IndexResult`` with the ``BM25Index`` and total document count.

        Raises:
            FileNotFoundError: If ``parquet_path`` does not exist.
        """
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        columns = [text_field] + list(metadata_fields or [])
        pf = pq.ParquetFile(parquet_path)
        total_rows = pf.metadata.num_rows
        all_documents: list[Document] = []
        rows_seen = 0

        logger.info(
            f"Streaming {parquet_path.name} — "
            f"{total_rows - skip_rows:,} rows (batch={batch_size:,})"
        )

        for batch in tqdm(
            pf.iter_batches(batch_size=batch_size, columns=columns),
            desc="Reading batches",
            unit="batch",
        ):
            batch_len = batch.num_rows
            if rows_seen + batch_len <= skip_rows:
                rows_seen += batch_len
                continue
            df = batch.to_pandas()
            if rows_seen < skip_rows:
                df = df.iloc[skip_rows - rows_seen :]
            rows_seen += batch_len

            docs = documents_from_dataframe(df, text_field, metadata_fields)
            all_documents.extend(docs)

        logger.info(f"Building BM25 index on {len(all_documents):,} documents…")
        self._index = self._build_index(all_documents, output_dir, collection_name)
        logger.info(f"BM25 index ready — {self._index.doc_count:,} documents")
        return IndexResult(self._index, self._index.doc_count)

    # ── Index lifecycle ───────────────────────────────────────────────────

    def load_index(
        self,
        path_or_name: str | Path,
        collection_name: str = "bm25",
        **kwargs: Any,
    ) -> BM25Index:
        """Load a persisted BM25 index from disk.

        Accepts either:
        - A directory path — loads ``<dir>/<collection_name>.pkl``.
        - A direct ``.pkl`` file path.

        Args:
            path_or_name: Directory or ``.pkl`` file path.
            collection_name: Index file stem (used when a directory is given).
            **kwargs: Ignored.

        Returns:
            The loaded ``BM25Index``.

        Raises:
            FileNotFoundError: If the index file cannot be found.
        """
        path = Path(path_or_name)
        pkl_path = path if path.suffix == ".pkl" else path / f"{collection_name}.pkl"
        self._index = BM25Index.load(pkl_path)
        self._index_path = pkl_path
        return self._index

    def save_index(self, path: str | Path, collection_name: str = "bm25", **kwargs: Any) -> None:
        """Persist the current index to *path*.

        Args:
            path: Destination directory or ``.pkl`` file path.
            collection_name: File stem (used when *path* is a directory).
            **kwargs: Ignored.

        Raises:
            ValueError: If no index is loaded.
        """
        index = self._require_index()
        p = Path(path)
        pkl_path = p if p.suffix == ".pkl" else p / f"{collection_name}.pkl"
        index.save(pkl_path)
        self._index_path = pkl_path

    def delete_index(self, *, delete_files: bool = False, **kwargs: Any) -> None:
        """Drop the current index from memory and optionally from disk.

        Args:
            delete_files: If ``True`` and the index was loaded from disk,
                delete the ``.pkl`` file.
            **kwargs: Ignored.
        """
        if delete_files and self._index_path and self._index_path.exists():
            self._index_path.unlink()
            logger.info(f"Deleted BM25 index file at {self._index_path}")
        self._index = None
        self._index_path = None
        logger.info("BM25 index cleared from memory")

    # ── Retrieval ─────────────────────────────────────────────────────────

    def retrieve_documents(
        self,
        query: str,
        *,
        top_k: int = 5,
        **kwargs: Any,
    ) -> list[Document]:
        """Return the top-k documents matching *query* via BM25.

        Args:
            query: Free-text query string.
            top_k: Number of documents to return.
            **kwargs: Ignored.

        Returns:
            Ranked list of ``Document`` objects.

        Raises:
            ValueError: If no index is loaded.
        """
        return self._require_index().similarity_search(query, k=top_k)

    def retrieve_documents_with_scores(
        self,
        query: str,
        *,
        top_k: int = 5,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return the top-k documents with rank-based reciprocal scores.

        Args:
            query: Free-text query string.
            top_k: Number of results to return.
            **kwargs: Ignored.

        Returns:
            List of ``(Document, score)`` tuples, best-first.
            Scores are ``1 / (rank + 1)``.

        Raises:
            ValueError: If no index is loaded.
        """
        return self._require_index().similarity_search_with_score(query, k=top_k)

    def batch_retrieve(
        self,
        queries: list[str],
        *,
        top_k: int = 5,
        progress_bar: bool = True,
        **kwargs: Any,
    ) -> list[list[Document]]:
        """Retrieve documents for multiple queries.

        Args:
            queries: List of query strings.
            top_k: Results per query.
            progress_bar: Show tqdm progress bar.
            **kwargs: Ignored.

        Returns:
            One list of Documents per query.

        Raises:
            ValueError: If no index is loaded.
        """
        self._require_index()
        return [
            self.retrieve_documents(q, top_k=top_k)
            for q in tqdm(queries, desc="Retrieving (bm25)", disable=not progress_bar)
        ]

    def batch_retrieve_with_scores(
        self,
        queries: list[str],
        *,
        top_k: int = 5,
        progress_bar: bool = True,
        **kwargs: Any,
    ) -> list[list[tuple[Document, float]]]:
        """Retrieve scored documents for multiple queries.

        Args:
            queries: List of query strings.
            top_k: Results per query.
            progress_bar: Show tqdm progress bar.
            **kwargs: Ignored.

        Returns:
            One list of ``(Document, score)`` tuples per query.

        Raises:
            ValueError: If no index is loaded.
        """
        self._require_index()
        return [
            self.retrieve_documents_with_scores(q, top_k=top_k)
            for q in tqdm(queries, desc="Retrieving (bm25)", disable=not progress_bar)
        ]

    # ── Inspection ────────────────────────────────────────────────────────

    def get_doc_count(self) -> int:
        """Return the number of documents in the current index.

        Returns:
            Document count, or 0 if no index is loaded.
        """
        return self._index.doc_count if self._index is not None else 0

    def get_all_documents(
        self,
        *,
        batch_size: int = 1_000,
        progress_bar: bool = True,
    ) -> list[Document]:
        """Return every document in the BM25 index.

        Args:
            batch_size: Ignored (BM25 documents are all in memory).
            progress_bar: Ignored.

        Returns:
            List of all stored ``Document`` objects.

        Raises:
            ValueError: If no index is loaded.
        """
        return self._require_index().docs

    def get_index_stats(self) -> dict[str, Any]:
        """Return statistics about the current BM25 index.

        Returns:
            Dict with keys: ``loaded``, ``doc_count``, ``b``, ``k1``,
            and optionally ``index_path``.
        """
        if self._index is None:
            return {"loaded": False}
        return {
            "loaded": True,
            "doc_count": self._index.doc_count,
            "b": self.b,
            "k1": self.k1,
            "index_path": str(self._index_path) if self._index_path else None,
        }

    # ── Embedding helpers ─────────────────────────────────────────────────

    def embed_prompt(self, text: str) -> str:
        """BM25 uses no prompt templates — returns *text* unchanged.

        Args:
            text: Query string.

        Returns:
            The original *text* unmodified.
        """
        return text

    def embed_passage(self, text: str) -> str:
        """BM25 uses no prompt templates — returns *text* unchanged.

        Args:
            text: Passage string.

        Returns:
            The original *text* unmodified.
        """
        return text
