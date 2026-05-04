"""Abstract base class and shared types for all RAG service implementations.

Every concrete backend (FAISS, Elasticsearch, BM25, Mock) inherits from
``RagService`` and must implement the methods marked ``@abstractmethod``.
Optional capabilities (e.g. ``save_index``, ``get_all_documents``) have
default implementations that raise ``NotImplementedError`` with a clear
message — backends that cannot support them should leave the default.

Shared types
------------
``VectorStoreLike``   Protocol satisfied by any object exposing
                      ``similarity_search`` and
                      ``similarity_search_with_score``.

``IndexResult``       Named tuple returned by all indexing methods:
                      ``(index, indexed_count)``.

Shared helpers
--------------
``documents_from_dataframe``  Build LangChain Documents from a DataFrame.
``split_documents``           Chunk documents via RecursiveCharacterTextSplitter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

import pandas as pd
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm


# ── Shared types ──────────────────────────────────────────────────────────────

@runtime_checkable
class VectorStoreLike(Protocol):
    """Structural protocol for vector store objects.

    Any object that exposes ``similarity_search`` and
    ``similarity_search_with_score`` satisfies this protocol — no explicit
    inheritance required.
    """

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[Document]:
        ...

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        ...


class IndexResult:
    """Return value of all indexing methods.

    Attributes:
        index: The constructed index object (type varies by backend).
        indexed_count: Number of documents / chunks indexed.
    """

    __slots__ = ("index", "indexed_count")

    def __init__(self, index: Any, indexed_count: int) -> None:
        self.index = index
        self.indexed_count = indexed_count

    def __iter__(self):
        """Allow tuple-style unpacking: ``index, count = result``."""
        yield self.index
        yield self.indexed_count

    def __repr__(self) -> str:
        return f"IndexResult(indexed_count={self.indexed_count:,})"


# ── Abstract base class ───────────────────────────────────────────────────────

class RagService(ABC):
    """Abstract base for retrieval-augmented generation backends.

    Unified interface across FAISS, Elasticsearch, BM25, and Mock backends.
    All backends share the same method names and signatures so that calling
    code never needs to branch on backend type for core operations.

    Indexing
    --------
    ``index_from_dataframe``       Build index from an in-memory DataFrame.
    ``index_from_parquet``         Build index from a Parquet file (whole file).
    ``index_from_parquet_batches`` Memory-efficient streaming indexing from Parquet.

    Index lifecycle
    ---------------
    ``load_index``   Attach a previously-built index to this service instance.
    ``save_index``   Persist the current index to disk (where applicable).
    ``delete_index`` Drop the current index from memory (and optionally disk).

    Retrieval
    ---------
    ``retrieve_documents``            Single query → list of Documents.
    ``retrieve_documents_with_scores`` Single query → list of (Document, score).
    ``batch_retrieve``                Many queries → list of result lists.
    ``batch_retrieve_with_scores``    Many queries → list of scored result lists.

    Inspection
    ----------
    ``get_doc_count``     Number of documents currently indexed.
    ``get_all_documents`` Iterate every stored document (where feasible).
    ``get_index_stats``   Backend-specific statistics dict.

    Embedding helpers
    -----------------
    ``embed_prompt``   Apply the query prompt template to a string.
    ``embed_passage``  Apply the passage prompt template to a string.
    """

    # ── Indexing ──────────────────────────────────────────────────────────

    @abstractmethod
    def index_from_dataframe(
        self,
        df: pd.DataFrame,
        text_field: str,
        *,
        metadata_fields: Sequence[str] | None = None,
        collection_name: str = "rag",
        **kwargs: Any,
    ) -> IndexResult:
        """Build an index from a pandas DataFrame.

        Args:
            df: Source DataFrame.
            text_field: Column containing document text.
            metadata_fields: Extra columns to store as document metadata.
            collection_name: Logical name / output path for the index.
            **kwargs: Backend-specific options.

        Returns:
            ``IndexResult`` with the index object and document count.
        """
        raise NotImplementedError

    @abstractmethod
    def index_from_parquet(
        self,
        parquet_path: Path,
        *,
        text_field: str = "text",
        metadata_fields: Sequence[str] | None = None,
        collection_name: str = "rag",
        **kwargs: Any,
    ) -> IndexResult:
        """Build an index from a Parquet file (loads the whole file).

        For large files prefer ``index_from_parquet_batches``.

        Args:
            parquet_path: Path to the ``.parquet`` file.
            text_field: Column containing document text.
            metadata_fields: Extra columns to store as metadata.
            collection_name: Logical name / output path for the index.
            **kwargs: Backend-specific options.

        Returns:
            ``IndexResult`` with the index object and document count.

        Raises:
            FileNotFoundError: If ``parquet_path`` does not exist.
        """
        raise NotImplementedError

    def index_from_parquet_batches(
        self,
        parquet_path: Path,
        *,
        text_field: str = "text",
        metadata_fields: Sequence[str] | None = None,
        collection_name: str = "rag",
        batch_size: int = 5_000,
        skip_rows: int = 0,
        **kwargs: Any,
    ) -> IndexResult:
        """Memory-efficient streaming indexing from a Parquet file.

        Reads the file in batches to keep RAM usage bounded.  Backends
        that do not support streaming should override and call
        ``index_from_parquet`` instead (or raise ``NotImplementedError``).

        Args:
            parquet_path: Path to the ``.parquet`` file.
            text_field: Column containing document text.
            metadata_fields: Extra columns to store as metadata.
            collection_name: Logical name / output path for the index.
            batch_size: Rows per batch.
            skip_rows: Skip this many leading rows (for resuming).
            **kwargs: Backend-specific options.

        Returns:
            ``IndexResult`` with the index object and total chunks indexed.

        Raises:
            FileNotFoundError: If ``parquet_path`` does not exist.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support streaming parquet indexing. "
            "Use index_from_parquet() instead."
        )

    # ── Index lifecycle ───────────────────────────────────────────────────

    @abstractmethod
    def load_index(self, path_or_name: str | Path, **kwargs: Any) -> Any:
        """Attach a previously-built index to this service instance.

        Args:
            path_or_name: File path (FAISS, BM25) or index name (Elasticsearch).
            **kwargs: Backend-specific options.

        Returns:
            The loaded index object.

        Raises:
            FileNotFoundError: If the index does not exist at the given path.
        """
        raise NotImplementedError

    def save_index(self, path: str | Path, **kwargs: Any) -> None:
        """Persist the current index to disk.

        Args:
            path: Destination directory or file path.
            **kwargs: Backend-specific options.

        Raises:
            NotImplementedError: If the backend does not support persistence
                (e.g. Elasticsearch stores its index server-side).
            ValueError: If no index is currently loaded.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support save_index(). "
            "The index is managed externally (e.g. on the Elasticsearch server)."
        )

    def delete_index(self, **kwargs: Any) -> None:
        """Drop the current index from memory (and optionally disk).

        Args:
            **kwargs: Backend-specific options (e.g. ``delete_files=True``).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement delete_index()."
        )

    # ── Retrieval ─────────────────────────────────────────────────────────

    @abstractmethod
    def retrieve_documents(
        self,
        query: str,
        *,
        top_k: int = 5,
        **kwargs: Any,
    ) -> list[Document]:
        """Return the top-k documents most relevant to *query*.

        Args:
            query: Free-text query string.
            top_k: Maximum number of documents to return.
            **kwargs: Backend-specific options (e.g. ``strategy``).

        Returns:
            List of ``Document`` objects, ranked by relevance (best first).

        Raises:
            ValueError: If no index is loaded.
        """
        raise NotImplementedError

    @abstractmethod
    def retrieve_documents_with_scores(
        self,
        query: str,
        *,
        top_k: int = 5,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return the top-k documents together with their relevance scores.

        Score semantics vary by backend:
        - Dense vector: cosine similarity (higher = more relevant).
        - BM25: rank-based reciprocal score ``1/(rank+1)`` (higher = better).
        - Elasticsearch: ES internal score (higher = more relevant).

        Args:
            query: Free-text query string.
            top_k: Maximum number of results to return.
            **kwargs: Backend-specific options.

        Returns:
            List of ``(Document, score)`` tuples, ranked best-first.

        Raises:
            ValueError: If no index is loaded.
        """
        raise NotImplementedError

    @abstractmethod
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
            top_k: Maximum results per query.
            progress_bar: Display a tqdm progress bar.
            **kwargs: Backend-specific options.

        Returns:
            List of result lists — one per query, each a list of Documents.
        """
        raise NotImplementedError

    @abstractmethod
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
            top_k: Maximum results per query.
            progress_bar: Display a tqdm progress bar.
            **kwargs: Backend-specific options.

        Returns:
            List of scored result lists — one per query, each a list of
            ``(Document, score)`` tuples ranked best-first.
        """
        raise NotImplementedError

    # ── Inspection ────────────────────────────────────────────────────────

    @abstractmethod
    def get_doc_count(self) -> int:
        """Return the number of documents currently indexed.

        Returns:
            Document count, or 0 if no index is loaded.
        """
        raise NotImplementedError

    def get_all_documents(
        self,
        *,
        batch_size: int = 1_000,
        progress_bar: bool = True,
    ) -> list[Document]:
        """Return every document stored in the index.

        This operation can be very expensive for large indices.  Backends
        where full iteration is not feasible (e.g. a remote Elasticsearch
        cluster with millions of documents) should raise
        ``NotImplementedError``.

        Args:
            batch_size: Number of documents to fetch per internal batch.
            progress_bar: Display a tqdm progress bar.

        Returns:
            List of all stored ``Document`` objects.

        Raises:
            NotImplementedError: If the backend does not support full iteration.
            ValueError: If no index is loaded.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support get_all_documents(). "
            "The index may be too large or stored remotely."
        )

    def get_index_stats(self) -> dict[str, Any]:
        """Return backend-specific statistics about the current index.

        Returns:
            Dict with at minimum ``{"loaded": bool}``.  Additional keys
            are backend-specific (e.g. ``n_vectors``, ``strategy``).
        """
        return {"loaded": False}

    # ── Embedding helpers ─────────────────────────────────────────────────

    def embed_prompt(self, text: str) -> str:
        """Apply the query prompt template to *text*.

        Backends that use prompt-prefixed embeddings (e.g. E5 models) wrap
        the query text with the configured template.  Backends without prompt
        templates return *text* unchanged.

        Args:
            text: Raw query string.

        Returns:
            Prompt-wrapped query string ready for embedding.
        """
        return text

    def embed_passage(self, text: str) -> str:
        """Apply the passage/document prompt template to *text*.

        Args:
            text: Raw passage string.

        Returns:
            Prompt-wrapped passage string ready for embedding.
        """
        return text


# ── Shared helpers ────────────────────────────────────────────────────────────

def documents_from_dataframe(
    df: pd.DataFrame,
    text_field: str,
    metadata_fields: Sequence[str] | None = None,
    *,
    progress_bar: bool = False,
) -> list[Document]:
    """Transform a DataFrame into LangChain Documents.

    Args:
        df: DataFrame to convert.
        text_field: Column name to use as document content.
        metadata_fields: Column names to include as metadata.
        progress_bar: Display a tqdm progress bar.

    Returns:
        List of ``Document`` objects.

    Raises:
        ValueError: If ``text_field`` or any ``metadata_fields`` are not
            present in the DataFrame.
    """
    if text_field not in df.columns:
        raise ValueError(f"text_field '{text_field}' not found in DataFrame columns")

    if metadata_fields is None:
        metadata_fields = []

    missing = set(metadata_fields) - set(df.columns)
    if missing:
        raise ValueError(f"metadata_fields {missing} not found in DataFrame columns")

    documents: list[Document] = []
    rows_iter = tqdm(
        df.iterrows(),
        total=len(df),
        desc="Building docs",
        unit="row",
        disable=not progress_bar,
    )
    for _, row in rows_iter:
        metadata = {field: row[field] for field in metadata_fields}
        documents.append(Document(page_content=str(row[text_field]), metadata=metadata))

    return documents


def split_documents(
    documents: Sequence[Document],
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[Document]:
    """Split documents into smaller chunks for better retrieval.

    Args:
        documents: Documents to split.
        chunk_size: Maximum characters per chunk.
        overlap: Number of characters to overlap between chunks.

    Returns:
        List of document chunks with preserved metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    return splitter.split_documents(list(documents))
