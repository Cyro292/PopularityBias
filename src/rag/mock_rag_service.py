"""Lightweight mock RAG service for testing and prototyping.

Stores documents in memory without any embedding or indexing.
Retrieval returns random subsets of stored documents with a score of 0.0.

This backend is useful for:
- Unit tests that need a ``RagService`` without external dependencies.
- Prototyping pipeline code before a real backend is wired in.
- Verifying that calling code handles the unified interface correctly.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from random import sample
from typing import Any, Sequence

import pandas as pd
import pyarrow.parquet as pq
from langchain.schema import Document
from tqdm import tqdm

from .base import IndexResult, RagService
from .base import documents_from_dataframe

logger = logging.getLogger(__name__)


class MockRagService(RagService):
    """Minimal RAG service that stores documents without embeddings.

    All retrieval methods return a random subset of the stored documents,
    each with a score of ``0.0``.  The index is held entirely in memory.

    Args:
        default_top_k: Default number of results for retrieval methods.

    Example::

        service = MockRagService()
        service.index_from_parquet(
            Path("data/corpus.parquet"),
            text_field="text",
        )
        docs = service.retrieve_documents("any query", top_k=3)
    """

    def __init__(self, *, default_top_k: int = 5) -> None:
        self.default_top_k = default_top_k
        self._documents: list[Document] = []
        self._index_path: Path | None = None

    # ── Internal helpers ──────────────────────────────────────────────────

    def _require_documents(self) -> list[Document]:
        """Return stored documents or raise if none are loaded.

        Returns:
            The stored document list.

        Raises:
            ValueError: If no documents have been indexed yet.
        """
        if not self._documents:
            raise ValueError(
                "No documents loaded. Call index_from_parquet(), "
                "index_from_dataframe(), or load_index() first."
            )
        return self._documents

    # ── Indexing ──────────────────────────────────────────────────────────

    def index_from_dataframe(
        self,
        df: pd.DataFrame,
        text_field: str,
        *,
        metadata_fields: Sequence[str] | None = None,
        collection_name: str = "mock",
        output_dir: Path | None = None,
        **kwargs: Any,
    ) -> IndexResult:
        """Store documents from a DataFrame in memory.

        Args:
            df: Source DataFrame.
            text_field: Column containing document text.
            metadata_fields: Extra columns to store as metadata.
            collection_name: Ignored (no persistence in mock mode unless
                ``output_dir`` is set).
            output_dir: Optional directory to pickle the document list.
            **kwargs: Ignored.

        Returns:
            ``IndexResult`` with ``None`` as the index object and
            the total document count.
        """
        self._documents = documents_from_dataframe(df, text_field, metadata_fields)
        logger.info(f"Mock RAG stored {len(self._documents):,} documents")

        if output_dir is not None:
            self.save_index(output_dir, collection_name=collection_name)

        return IndexResult(None, len(self._documents))

    def index_from_parquet(
        self,
        parquet_path: Path,
        *,
        text_field: str = "text",
        metadata_fields: Sequence[str] | None = None,
        collection_name: str = "mock",
        output_dir: Path | None = None,
        **kwargs: Any,
    ) -> IndexResult:
        """Store documents from a Parquet file in memory.

        Args:
            parquet_path: Path to the ``.parquet`` file.
            text_field: Column containing document text.
            metadata_fields: Extra columns to store as metadata.
            collection_name: File stem for optional persistence.
            output_dir: Optional directory to pickle the document list.
            **kwargs: Ignored.

        Returns:
            ``IndexResult`` with ``None`` as the index object and
            the total document count.

        Raises:
            FileNotFoundError: If ``parquet_path`` does not exist.
        """
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        columns = [text_field] + list(metadata_fields or [])
        df = pq.read_table(parquet_path, columns=columns).to_pandas()
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
        collection_name: str = "mock",
        output_dir: Path | None = None,
        batch_size: int = 100_000,
        skip_rows: int = 0,
        **kwargs: Any,
    ) -> IndexResult:
        """Stream a Parquet file into memory in batches.

        Args:
            parquet_path: Path to the ``.parquet`` file.
            text_field: Column containing document text.
            metadata_fields: Extra columns to store as metadata.
            collection_name: File stem for optional persistence.
            output_dir: Optional directory to pickle the document list.
            batch_size: Rows per Arrow batch.
            skip_rows: Skip this many leading rows.
            **kwargs: Ignored.

        Returns:
            ``IndexResult`` with ``None`` as the index object and total count.

        Raises:
            FileNotFoundError: If ``parquet_path`` does not exist.
        """
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        columns = [text_field] + list(metadata_fields or [])
        pf = pq.ParquetFile(parquet_path)
        all_documents: list[Document] = []
        rows_seen = 0

        for batch in tqdm(
            pf.iter_batches(batch_size=batch_size, columns=columns),
            desc="Reading (mock)",
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
            all_documents.extend(documents_from_dataframe(df, text_field, metadata_fields))

        self._documents = all_documents
        logger.info(f"Mock RAG stored {len(self._documents):,} documents")

        if output_dir is not None:
            self.save_index(output_dir, collection_name=collection_name)

        return IndexResult(None, len(self._documents))

    # ── Index lifecycle ───────────────────────────────────────────────────

    def load_index(
        self,
        path_or_name: str | Path,
        collection_name: str = "mock",
        **kwargs: Any,
    ) -> None:
        """Load a previously pickled document list from disk.

        Args:
            path_or_name: Directory or ``.pkl`` file path.
            collection_name: File stem (used when a directory is given).
            **kwargs: Ignored.

        Returns:
            ``None`` (Mock has no index object).

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        path = Path(path_or_name)
        pkl_path = path if path.suffix == ".pkl" else path / f"{collection_name}.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(f"Mock index not found at {pkl_path}")
        with open(pkl_path, "rb") as fh:
            self._documents = pickle.load(fh)
        self._index_path = pkl_path
        logger.info(f"Mock RAG loaded {len(self._documents):,} documents from {pkl_path}")
        return None

    def save_index(
        self,
        path: str | Path,
        collection_name: str = "mock",
        **kwargs: Any,
    ) -> None:
        """Pickle the document list to *path*.

        Args:
            path: Destination directory or ``.pkl`` file path.
            collection_name: File stem (used when *path* is a directory).
            **kwargs: Ignored.

        Raises:
            ValueError: If no documents are loaded.
        """
        self._require_documents()
        p = Path(path)
        pkl_path = p if p.suffix == ".pkl" else p / f"{collection_name}.pkl"
        pkl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pkl_path, "wb") as fh:
            pickle.dump(self._documents, fh)
        self._index_path = pkl_path
        logger.info(f"Mock RAG saved {len(self._documents):,} documents to {pkl_path}")

    def delete_index(self, *, delete_files: bool = False, **kwargs: Any) -> None:
        """Clear the stored documents from memory.

        Args:
            delete_files: If ``True`` and the index was loaded from / saved to
                disk, delete the ``.pkl`` file.
            **kwargs: Ignored.
        """
        if delete_files and self._index_path and self._index_path.exists():
            self._index_path.unlink()
            logger.info(f"Deleted mock index file at {self._index_path}")
        self._documents = []
        self._index_path = None
        logger.info("Mock RAG index cleared")

    # ── Retrieval ─────────────────────────────────────────────────────────

    def retrieve_documents(
        self,
        query: str,
        *,
        top_k: int = 5,
        **kwargs: Any,
    ) -> list[Document]:
        """Return a random subset of stored documents (query is ignored).

        Args:
            query: Ignored.
            top_k: Maximum number of documents to return.
            **kwargs: Ignored.

        Returns:
            Random list of up to ``top_k`` Documents.

        Raises:
            ValueError: If no documents are loaded.
        """
        docs = self._require_documents()
        if top_k <= 0:
            return []
        return sample(docs, k=min(top_k, len(docs)))

    def retrieve_documents_with_scores(
        self,
        query: str,
        *,
        top_k: int = 5,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return a random subset of documents, each with score ``0.0``.

        Args:
            query: Ignored.
            top_k: Maximum number of results to return.
            **kwargs: Ignored.

        Returns:
            List of ``(Document, 0.0)`` tuples.

        Raises:
            ValueError: If no documents are loaded.
        """
        docs = self.retrieve_documents(query, top_k=top_k)
        return [(doc, 0.0) for doc in docs]

    def batch_retrieve(
        self,
        queries: list[str],
        *,
        top_k: int = 5,
        progress_bar: bool = True,
        **kwargs: Any,
    ) -> list[list[Document]]:
        """Retrieve random documents for multiple queries.

        Args:
            queries: List of query strings (all ignored).
            top_k: Results per query.
            progress_bar: Show tqdm progress bar.
            **kwargs: Ignored.

        Returns:
            One list of random Documents per query.

        Raises:
            ValueError: If no documents are loaded.
        """
        self._require_documents()
        return [
            self.retrieve_documents(q, top_k=top_k)
            for q in tqdm(queries, desc="Retrieving (mock)", disable=not progress_bar)
        ]

    def batch_retrieve_with_scores(
        self,
        queries: list[str],
        *,
        top_k: int = 5,
        progress_bar: bool = True,
        **kwargs: Any,
    ) -> list[list[tuple[Document, float]]]:
        """Retrieve random scored documents for multiple queries.

        Args:
            queries: List of query strings (all ignored).
            top_k: Results per query.
            progress_bar: Show tqdm progress bar.
            **kwargs: Ignored.

        Returns:
            One list of ``(Document, 0.0)`` tuples per query.

        Raises:
            ValueError: If no documents are loaded.
        """
        self._require_documents()
        return [
            self.retrieve_documents_with_scores(q, top_k=top_k)
            for q in tqdm(queries, desc="Retrieving (mock)", disable=not progress_bar)
        ]

    # ── Inspection ────────────────────────────────────────────────────────

    def get_doc_count(self) -> int:
        """Return the number of stored documents.

        Returns:
            Document count, or 0 if none are loaded.
        """
        return len(self._documents)

    def get_all_documents(
        self,
        *,
        batch_size: int = 1_000,
        progress_bar: bool = True,
    ) -> list[Document]:
        """Return all stored documents.

        Args:
            batch_size: Ignored (all documents are in memory).
            progress_bar: Ignored.

        Returns:
            Copy of the stored document list.

        Raises:
            ValueError: If no documents are loaded.
        """
        return list(self._require_documents())

    def get_index_stats(self) -> dict[str, Any]:
        """Return statistics about the mock index.

        Returns:
            Dict with keys: ``loaded``, ``doc_count``, and optionally
            ``index_path``.
        """
        return {
            "loaded": bool(self._documents),
            "doc_count": len(self._documents),
            "index_path": str(self._index_path) if self._index_path else None,
        }

    # ── Embedding helpers ─────────────────────────────────────────────────

    def embed_prompt(self, text: str) -> str:
        """Mock has no prompt templates — returns *text* unchanged.

        Args:
            text: Query string.

        Returns:
            The original *text* unmodified.
        """
        return text

    def embed_passage(self, text: str) -> str:
        """Mock has no prompt templates — returns *text* unchanged.

        Args:
            text: Passage string.

        Returns:
            The original *text* unmodified.
        """
        return text
