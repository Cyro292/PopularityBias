"""RAG (Retrieval-Augmented Generation) Service for Document Indexing.

This module provides a comprehensive service for indexing and querying documents using:
- Chroma vector store for efficient similarity search
- Multiple data sources: Parquet files, pandas DataFrames, HuggingFace Datasets
- Text chunking for optimal retrieval
- Multiple embedding providers (OpenAI, Google, HuggingFace)
- Rate limiting to prevent API throttling
- Memory-efficient batch processing for large datasets
- Resume capability for interrupted indexing jobs
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_core.rate_limiters import BaseRateLimiter, InMemoryRateLimiter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from .base import RagService, VectorStoreLike
from .document_utils import documents_from_text_dataframe, documents_from_html_dataframe, documents_from_text_arrow

import gc

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

_EMBEDDING_CLASSES: dict[str, type] = {}

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_EMBEDDING_PROVIDER = "openai"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
DEFAULT_RATE_LIMIT_CHECK_INTERVAL = 0.1
DEFAULT_RATE_LIMIT_BUCKET_SIZE = 1.0


@dataclass
class IndexingConfig:
    """Configuration for document indexing and embedding generation.

    This class controls all aspects of the indexing pipeline including:
    - Text chunking strategy
    - Embedding model selection
    - Rate limiting to prevent API throttling
    - Progress display options
    - Index storage distance metric (can use different metrics at query time)

    Attributes:
        chunk_size: Maximum characters per document chunk. Larger chunks provide more context
                   but may reduce retrieval precision.
        chunk_overlap: Number of characters to overlap between consecutive chunks. Helps maintain
                      context at chunk boundaries.
        embedding_provider: Which embedding service to use ("openai", "google", "huggingface").
        embedding_model: Specific model name (e.g., "text-embedding-3-large").
        use_progress: Whether to show progress bars during indexing.
        rate_limiter: Custom rate limiter instance. If provided, overrides requests_per_second.
        requests_per_second: Automatic rate limiting (requests/sec). Creates an in-memory limiter.
        rate_limit_check_interval: How often to check rate limits (seconds).
        rate_limit_bucket_size: Token bucket size for burst handling.
        distance_function: Distance function for index storage ("cosine", "l2", "ip").
                          Default "cosine" works well for most cases. You can query with any
                          metric at retrieval time - the index will re-rank results accordingly.
                          Using the same metric at query time as at index time is faster.
    """

    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER
    embedding_model: str = DEFAULT_OPENAI_EMBEDDING_MODEL
    use_progress: bool = True
    rate_limiter: BaseRateLimiter | None = None
    requests_per_second: float | None = None
    rate_limit_check_interval: float = DEFAULT_RATE_LIMIT_CHECK_INTERVAL
    rate_limit_bucket_size: float = DEFAULT_RATE_LIMIT_BUCKET_SIZE
    distance_function: str = "cosine"


def _get_embedding_class(provider: str):
    """Lazy-load embedding classes to avoid import overhead.

    This function implements lazy loading: embedding providers are only imported
    when actually needed, which reduces startup time and avoids unnecessary dependencies.
    Loaded classes are cached for reuse.

    Args:
        provider: Name of the embedding provider ("openai", "google", "huggingface").

    Returns:
        The embedding class for the specified provider.

    Raises:
        ValueError: If the provider is not supported.
    """
    provider = provider.lower()
    if provider not in _EMBEDDING_CLASSES:
        if provider == "openai":
            from langchain_openai import OpenAIEmbeddings
            _EMBEDDING_CLASSES[provider] = OpenAIEmbeddings
        elif provider == "google":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            _EMBEDDING_CLASSES[provider] = GoogleGenerativeAIEmbeddings
        elif provider == "huggingface":
            from langchain_huggingface import HuggingFaceEmbeddings
            _EMBEDDING_CLASSES[provider] = HuggingFaceEmbeddings
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")
    return _EMBEDDING_CLASSES[provider]


def _build_embeddings(config: IndexingConfig):
    """Build embeddings instance with optional rate limiting.

    Creates an embedding model based on the config and wraps it with rate limiting
    if configured. This prevents hitting API rate limits during bulk operations.

    Args:
        config: IndexingConfig specifying the provider, model, and rate limits.

    Returns:
        Embeddings instance, optionally wrapped with rate limiting.
    """
    embedding_cls = _get_embedding_class(config.embedding_provider)

    if config.embedding_provider.lower() == "huggingface":
        base = embedding_cls(model_name=config.embedding_model)
    else:
        base = embedding_cls(model=config.embedding_model)

    limiter = _resolve_rate_limiter(config)
    return _LangChainRateLimitedEmbeddings(base, limiter) if limiter else base


def _resolve_rate_limiter(config: IndexingConfig) -> BaseRateLimiter | None:
    """Create a rate limiter from configuration settings.

    Supports two modes:
    1. Custom limiter: Use the provided rate_limiter instance
    2. Auto limiter: Create an in-memory limiter from requests_per_second

    Args:
        config: IndexingConfig with rate limiting settings.

    Returns:
        A rate limiter instance, or None if rate limiting is disabled.
    """
    if config.rate_limiter:
        return config.rate_limiter

    if config.requests_per_second and config.requests_per_second > 0:
        return InMemoryRateLimiter(
            requests_per_second=config.requests_per_second,
            check_every_n_seconds=config.rate_limit_check_interval,
            max_bucket_size=config.rate_limit_bucket_size,
        )

    return None


class _LangChainRateLimitedEmbeddings:
    """Wrapper that adds rate limiting to any embeddings instance.

    This class intercepts embed_documents() and embed_query() calls to enforce
    rate limits before forwarding to the underlying embeddings model. This prevents
    API throttling errors during bulk indexing operations.

    The rate limiter uses a token bucket algorithm:
    - Tokens (representing allowed requests) accumulate over time
    - Each embedding call consumes a token
    - If no tokens available, the call blocks until one is available
    """

    def __init__(self, embeddings: Any, rate_limiter: BaseRateLimiter):
        """Initialize the rate-limited embeddings wrapper.

        Args:
            embeddings: The underlying embeddings instance to wrap.
            rate_limiter: LangChain rate limiter to enforce request limits.
        """
        self._embeddings = embeddings
        self._rate_limiter = rate_limiter

    def _acquire(self) -> None:
        """Block until a rate limit token is available."""
        self._rate_limiter.acquire(blocking=True)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed multiple documents with rate limiting.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (one per text).
        """
        self._acquire()
        return self._embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query with rate limiting.

        Args:
            text: Query text to embed.

        Returns:
            Single embedding vector.
        """
        self._acquire()
        return self._embeddings.embed_query(text)

    def __getattr__(self, item: str) -> Any:
        """Forward all other attribute access to the underlying embeddings."""
        return getattr(self._embeddings, item)


def _split_documents(
    documents: Sequence[Document],
    chunk_size: int,
    overlap: int,
) -> list[Document]:
    """Split documents into smaller chunks for better retrieval.

    Long documents are split into overlapping chunks. Chunking improves retrieval
    by allowing more precise matching to specific parts of documents. The overlap
    helps maintain context at chunk boundaries.

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


def _create_chroma_index(
    documents: list[Document],
    embeddings: Any,
    collection_name: str,
    persist_dir: str | None = None,
    batch_size: int = 500,
    show_progress: bool = False,
    distance_function: str = "cosine",
) -> Chroma:
    """Create Chroma vector index with batched insertion and progress tracking.

    For small datasets, creates the index in one operation. For large datasets,
    uses batched insertion with a progress bar to provide feedback and better
    memory management.

    Args:
        documents: Documents to index (must be already chunked if desired).
        embeddings: Embeddings instance to use for vectorization.
        collection_name: Name for the Chroma collection.
        persist_dir: Optional directory to save the index to disk.
        batch_size: Number of documents to process in each batch.
        show_progress: Whether to display a progress bar.
        distance_function: Distance function to use ("cosine", "l2", "ip").

    Returns:
        Initialized Chroma vector store.

    Raises:
        ValueError: If no documents provided or invalid distance function.
    """
    if not documents:
        raise ValueError("No documents to index")

    valid_functions = ["cosine", "l2", "ip"]
    if distance_function not in valid_functions:
        raise ValueError(f"Invalid distance function: {distance_function}. Must be one of {valid_functions}")

    collection_metadata = {"hnsw:space": distance_function}

    if len(documents) <= batch_size or not show_progress:
        kwargs = {
            "documents": documents,
            "embedding": embeddings,
            "collection_name": collection_name,
            "collection_metadata": collection_metadata,
        }
        if persist_dir:
            kwargs["persist_directory"] = persist_dir
        return Chroma.from_documents(**kwargs)

    first_batch = documents[:batch_size]
    kwargs = {
        "documents": first_batch,
        "embedding": embeddings,
        "collection_name": collection_name,
        "collection_metadata": collection_metadata,
    }
    if persist_dir:
        kwargs["persist_directory"] = persist_dir

    vectorstore = Chroma.from_documents(**kwargs)

    remaining = documents[batch_size:]
    with tqdm(total=len(remaining), desc="Indexing documents", unit="docs") as pbar:
        for i in range(0, len(remaining), batch_size):
            batch = remaining[i:i + batch_size]
            vectorstore.add_documents(batch)
            pbar.update(len(batch))

    return vectorstore


def _prepare_persist_dir(output_dir: Path | None) -> Path | None:
    """Prepare and validate the persistence directory for Chroma.

    This function handles the full lifecycle of directory preparation:
    1. Creates the output directory if it doesn't exist
    2. Removes any existing index to ensure a clean state
    3. Creates a fresh chroma subdirectory
    4. Validates write permissions
    5. Tests actual write capability

    Args:
        output_dir: Parent directory for the index. If None, returns None (in-memory index).

    Returns:
        Path to the chroma subdirectory, or None for in-memory operation.

    Raises:
        PermissionError: If the directory is not writable or cannot be accessed.
    """
    if not output_dir:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    persist_dir = output_dir / "chroma"

    if persist_dir.exists():
        logger.info(f"Deleting existing index at {persist_dir}")
        try:
            shutil.rmtree(persist_dir)
            time.sleep(0.1)
        except PermissionError as e:
            logger.error(f"Permission denied deleting {persist_dir}: {e}")
            raise

    persist_dir.mkdir(parents=True, exist_ok=True)
    if not os.access(persist_dir, os.W_OK):
        raise PermissionError(f"Directory {persist_dir} is not writable")

    test_file = persist_dir / ".write_test"
    try:
        test_file.touch()
        test_file.unlink()
    except Exception as e:
        raise PermissionError(f"Cannot write to {persist_dir}: {e}")

    return persist_dir


class nativeRagService(RagService):
    """RAG Service for indexing and querying documents with Chroma.

    This service provides a high-level interface for:
    - Creating vector indices from multiple data sources (Parquet, DataFrame, Dataset)
    - Chunking documents for optimal retrieval
    - Managing embeddings with rate limiting
    - Querying indices for similar documents
    - Resuming interrupted indexing jobs

    The service uses Chroma as the vector store backend and supports multiple
    embedding providers (OpenAI, Google, HuggingFace).

    Example:
        >>> config = IndexingConfig(chunk_size=500, embedding_provider="openai")
        >>> service = nativeRagService(config)
        >>> index, count = service.index_from_parquet(
        ...     parquet_path=Path("data.parquet"),
        ...     text_field="content",
        ...     output_dir=Path("./index")
        ... )
    """

    def __init__(self, config: IndexingConfig | None = None):
        """Initialize RAG service."""
        self.config = config or IndexingConfig()
        self.distance_function = self.config.distance_function
        self._embeddings = _build_embeddings(self.config)

    def _prepare_documents(self, documents: list[Document]) -> list[Document]:
        """Apply chunking if configured."""
        if not self.config.chunk_size:
            return documents
        return _split_documents(documents, self.config.chunk_size, self.config.chunk_overlap)

    def _build_index(
        self, documents: list[Document], collection_name: str,
        output_dir: Path | None = None, progress_bar: bool = False
    ) -> tuple[Chroma, int]:
        """Build Chroma index from documents."""
        persist_dir = _prepare_persist_dir(output_dir)
        logger.info(f"Creating {'persistent' if persist_dir else 'in-memory'} index with {len(documents)} docs")

        vectorstore = _create_chroma_index(
            documents=documents, embeddings=self._embeddings, collection_name=collection_name,
            persist_dir=str(persist_dir) if persist_dir else None,
            show_progress=progress_bar, distance_function=self.distance_function
        )
        logger.info(f"Created index with {len(documents)} chunks using {self.distance_function} distance")
        return vectorstore, len(documents)

    def _index_from_parquet(
        self, parquet_path: Path, text_field: str | None, html_field: str | None,
        metadata_fields: Sequence[str] | None, collection_name: str, output_dir: Path, progress_bar: bool
    ) -> tuple[Chroma, int]:
        """Index text or HTML from parquet."""
        field = html_field or text_field
        meta_fields = tuple(metadata_fields or ())

        logger.info(f"Reading {parquet_path}")
        df = pq.read_table(parquet_path, columns=[field, *meta_fields]).to_pandas()
        logger.info(f"Loaded {len(df)} rows")

        doc_creator = documents_from_html_dataframe if html_field else documents_from_text_dataframe
        documents = doc_creator(df, field, meta_fields, source=str(parquet_path), row_offset=0)

        return self._build_index(self._prepare_documents(documents), collection_name, output_dir, progress_bar)

    def index_from_parquet(
        self,
        parquet_path: Path,
        output_dir: Path,
        *,
        text_field: str | None = None,
        html_field: str | None = None,
        metadata_fields: Sequence[str] | None = None,
        collection_name: str = "rag",
        progress_bar: bool = False,
    ) -> tuple[Chroma, int]:
        """Load text or HTML column from Parquet and persist it to Chroma.
        
        Args:
            parquet_path: Path to the Parquet file.
            output_dir: Directory to persist the index.
            text_field: Column name containing the text content. Mutually exclusive with html_field.
            html_field: Column name containing the HTML content. Mutually exclusive with text_field.
            metadata_fields: Optional column names to include as document metadata.
            collection_name: Name for the Chroma collection.
            progress_bar: Show progress bar during indexing.
            
        Returns:
            Tuple of (Chroma vectorstore, number of document chunks indexed).
            
        Raises:
            ValueError: If neither or both text_field and html_field are provided.
        """
        if text_field is None and html_field is None:
            raise ValueError("Either text_field or html_field must be provided")
        if text_field is not None and html_field is not None:
            raise ValueError("Only one of text_field or html_field should be provided")
        
        return self._index_from_parquet(
            parquet_path, text_field, html_field, metadata_fields,
            collection_name, output_dir, progress_bar
        )

    def index_from_dataframe(
        self,
        df: pd.DataFrame,
        text_field: str | None = None,
        html_field: str | None = None,
        *,
        metadata_fields: Sequence[str] | None = None,
        output_dir: Path | None = None,
        collection_name: str = "rag",
        progress_bar: bool = False,
    ) -> tuple[Chroma, int]:
        """Create a Chroma index from a pandas DataFrame.
        
        Args:
            df: DataFrame containing the text data to index.
            text_field: Column name containing the text content.
            metadata_fields: Optional column names to include as document metadata.
            output_dir: Optional directory to persist the index. If None, index is in-memory.
            collection_name: Name for the Chroma collection.
            progress_bar: Show progress bar during indexing.
            
        Returns:
            Tuple of (Chroma vectorstore, number of document chunks indexed).
        """

        if html_field is not None:
            raise NotImplementedError("HTML field indexing from DataFrame is not implemented")
        
        metadata_fields_tuple = tuple(metadata_fields or ())
        logger.info("Indexing DataFrame with %d rows", len(df))

        documents = documents_from_text_dataframe(
            df,
            text_field,
            metadata_fields_tuple,
            source="dataframe",
            row_offset=0,
        )
        logger.info("Converted %d rows to %d documents", len(df), len(documents))

        documents = self._prepare_documents(documents)
        return self._build_index(documents, collection_name, output_dir, progress_bar)

    def index_from_dataset(
        self,
        ds: Any,
        text_field: str | None = None,
        html_field: str | None = None,
        *,
        metadata_fields: Sequence[str] | None = None,
        output_dir: Path | None = None,
        collection_name: str = "rag",
        progress_bar: bool = False,
        batch_size: int = 1000,
        resume_from_row: int = 0,
    ) -> tuple[Chroma, int]:
        """Create a Chroma index from a HuggingFace Dataset or any dataset with .iter() or .to_pandas() method.

        Processes the dataset in batches to avoid loading everything into memory at once.

        Args:
            ds: Dataset object (e.g., HuggingFace Dataset) that supports .iter(batch_size=...) or .to_pandas().
            text_field: Column name containing the text content. Required if html_field is not provided.
            html_field: Column name containing the HTML content. Required if text_field is not provided.
            metadata_fields: Optional column names to include as document metadata.
            output_dir: Optional directory to persist the index. If None, index is in-memory.
            collection_name: Name for the Chroma collection.
            progress_bar: Show progress bar during indexing.
            batch_size: Number of rows to process in each batch. Default is 1000.
            resume_from_row: Row number to resume indexing from. Used when recovering from a crash.

        Returns:
            Tuple of (Chroma vectorstore, number of document chunks indexed).

        Raises:
            ValueError: If neither text_field nor html_field is provided, or both are provided.
            TypeError: If dataset doesn't support .iter() or .to_pandas() methods.
        """
        if text_field is None and html_field is None:
            raise ValueError("Either text_field or html_field must be provided")
        if text_field is not None and html_field is not None:
            raise ValueError("Only one of text_field or html_field should be provided")

        metadata_fields_tuple = tuple(metadata_fields or ())
        use_batch_iteration = hasattr(ds, "iter")

        if use_batch_iteration:
            if resume_from_row > 0:
                logger.info("Resuming indexing from row %d...", resume_from_row)
            logger.info("Processing dataset in batches of %d rows...", batch_size)
            return self._index_from_dataset_batched(
                ds, text_field, html_field, metadata_fields_tuple,
                output_dir, collection_name, progress_bar, batch_size, resume_from_row
            )
        elif hasattr(ds, "to_pandas"):
            logger.warning(
                "Dataset does not support .iter() method. Converting entire dataset to pandas. "
                "This may use significant memory. Consider using a dataset that supports batch iteration."
            )
            logger.info("Converting dataset to pandas DataFrame...")
            df = ds.to_pandas()
            logger.info(f"Converted dataset to DataFrame with {len(df)} rows")

            if html_field is not None:
                if html_field not in df.columns:
                    raise KeyError(f"HTML field '{html_field}' missing from dataset.")
                
                documents = documents_from_html_dataframe(
                    df,
                    html_field,
                    metadata_fields_tuple,
                    source="dataset",
                    row_offset=0,
                )
                logger.info("Converted %d rows to %d documents from HTML field", len(df), len(documents))
            else:
                if text_field not in df.columns:
                    raise KeyError(f"Text field '{text_field}' missing from dataset.")
                
                documents = documents_from_text_dataframe(
                    df,
                    text_field,
                    metadata_fields_tuple,
                    source="dataset",
                    row_offset=0,
                )
                logger.info("Converted %d rows to %d documents from text field", len(df), len(documents))

            documents = self._prepare_documents(documents)
            return self._build_index(documents, collection_name, output_dir, progress_bar)
        else:
            raise TypeError(
                f"Dataset object must have either a .iter(batch_size=...) method or .to_pandas() method. "
                f"Got type: {type(ds)}"
            )

    def _index_from_dataset_batched(
        self,
        ds,
        text_field,
        html_field,
        metadata_fields_tuple,
        output_dir,
        collection_name,
        progress_bar,
        batch_size,
        resume_from_row=0,
    ):
        """Memory-efficient batch indexing for large datasets with resume capability.

        This method processes datasets in batches to avoid loading everything into memory.
        Key features:
        - Streams data in configurable batch sizes
        - Resumes from interrupted jobs
        - Periodic garbage collection to manage memory
        - Progress tracking for long-running jobs
        - Respects Chroma's write limits

        Memory optimization strategies:
        1. Process data in small batches (default 1000 rows)
        2. Use PyArrow for efficient data handling
        3. Write to Chroma in chunks (max 5000 docs per write)
        4. Force garbage collection every 10 batches
        5. Reduce logging during batch processing

        Args:
            ds: Dataset with .iter() method for batch iteration.
            text_field: Column containing text to index.
            html_field: Column containing HTML (not yet implemented).
            metadata_fields_tuple: Columns to include as metadata.
            output_dir: Directory to persist the index.
            collection_name: Name for the Chroma collection.
            progress_bar: Whether to show progress.
            batch_size: Rows to process per batch.
            resume_from_row: Row number to resume from (for crash recovery).

        Returns:
            Tuple of (Chroma vectorstore, total document chunks indexed).

        Raises:
            ValueError: If resuming but no existing index found, or no documents indexed.
        """
        CHROMA_WRITE_LIMIT = 1_000
        total_rows = getattr(ds, "num_rows", None)

        if resume_from_row > 0:
            persist_dir = output_dir / "chroma" if output_dir else None
            if persist_dir and persist_dir.exists():
                logger.info("Loading existing index from %s", persist_dir)
                vectorstore = Chroma(
                    persist_directory=str(persist_dir),
                    embedding_function=self._embeddings,
                    collection_name=collection_name,
                )
            else:
                raise ValueError(f"Cannot resume: no existing index found at {persist_dir}")
        else:
            persist_dir = _prepare_persist_dir(output_dir)
            vectorstore = None

        dataset_iter = ds.iter(batch_size=batch_size)

        if resume_from_row > 0:
            batches_to_skip = resume_from_row // batch_size
            logger.info("Skipping %d batches to reach row %d...", batches_to_skip, resume_from_row)
            for _ in range(batches_to_skip):
                try:
                    next(dataset_iter)
                except StopIteration:
                    raise ValueError(f"Cannot resume from row {resume_from_row}: dataset has fewer rows")
            gc.collect()

        total_docs = 0
        row_offset = resume_from_row
        pbar = tqdm(total=total_rows, initial=resume_from_row, unit="rows") if progress_bar and total_rows else None

        original_log_level = logger.level
        logger.setLevel(logging.WARNING)

        try:
            for batch_idx, batch in enumerate(dataset_iter):
                table = pa.table(batch) if isinstance(batch, dict) else batch
                batch_num_rows = table.num_rows

                if batch_num_rows == 0:
                    continue

                if html_field:
                    raise ValueError("documents_from_html_arrow is not implemented")

                docs = documents_from_text_arrow(table, text_field, metadata_fields_tuple, "dataset", row_offset)
                del table, batch

                docs = self._prepare_documents(docs)

                if docs:
                    for i in range(0, len(docs), CHROMA_WRITE_LIMIT):
                        chunk = docs[i : i + CHROMA_WRITE_LIMIT]
                        if vectorstore is None:
                            collection_metadata = {"hnsw:space": self.distance_function}
                            vectorstore = Chroma.from_documents(
                                documents=chunk, embedding=self._embeddings,
                                collection_name=collection_name,
                                persist_directory=str(persist_dir) if persist_dir else None,
                                collection_metadata=collection_metadata
                            )
                        else:
                            vectorstore.add_documents(chunk)
                        del chunk

                    total_docs += len(docs)

                del docs
                row_offset += batch_num_rows
                if pbar:
                    pbar.update(batch_num_rows)
                gc.collect()

            if not vectorstore:
                raise ValueError("No documents indexed")


            return vectorstore, total_docs

        finally:
            logger.setLevel(original_log_level)
            if pbar:
                pbar.close()

    def load_index(self, output_dir: Path, collection_name: str = "wiki_demo") -> Chroma | None:
        """Load existing Chroma index from disk."""
        persist_dir = output_dir / "chroma"
        if not persist_dir.exists():
            return None

        index = Chroma(
            persist_directory=str(persist_dir),
            embedding_function=self._embeddings,
            collection_name=collection_name,
        )

        try:
            if metric := index._collection.metadata.get("hnsw:space"):
                logger.info(f"Loaded index with {metric} distance")
        except Exception:
            pass

        return index

    def add_documents(self, index: VectorStoreLike, documents: Sequence[Document]):
        """Add documents to index."""
        return index.add_documents(documents) or index

    def delete_documents(self, index: VectorStoreLike, document_ids: Sequence[str]):
        """Delete documents from index."""
        return index.delete(document_ids) or index

    def get_document_by_id(self, index: VectorStoreLike, document_id: int | str) -> list[Document]:
        """Get documents by metadata 'id' field."""
        results = index._collection.get(
            where={"id": {"$eq": int(document_id)}},
            include=["documents", "metadatas"]
        )
        return [
            Document(page_content=text, metadata=results["metadatas"][i] if results.get("metadatas") else {})
            for i, text in enumerate(results.get("documents", []))
        ]

    def retrieve_documents(
        self, index: VectorStoreLike | None, text: str, *,
        top_k: int = 5, distance_function: str | None = None, fetch_k: int | None = None
    ) -> list[Document]:
        """Return documents matching query, optionally with custom distance metric."""
        if distance_function:
            return [doc for doc, _ in self.retrieve_documents_with_scores(
                index, text, top_k=top_k, distance_function=distance_function, fetch_k=fetch_k
            )]
        if not index or top_k <= 0:
            raise ValueError("Index required and top_k must be > 0")
        return index.similarity_search(text, k=top_k)

    def retrieve_documents_with_scores(
        self,
        index: VectorStoreLike | None,
        text: str,
        *,
        top_k: int = 5,
        distance_function: str | None = None,
        fetch_k: int | None = None,
    ) -> list[tuple[Document, float]]:
        """Return documents with similarity scores.

        Args:
            index: Vector store to search.
            text: Query text.
            top_k: Number of results.
            distance_function: Metric to use - None (native/fastest), "cosine", "euclidean"/"l2", "inner_product"/"ip"
            fetch_k: Candidates to fetch for re-ranking (default: top_k * 3).

        Returns:
            List of (Document, score) tuples. Lower=better except inner_product (higher=better).
        """
        if not index or top_k <= 0:
            raise ValueError("Index required and top_k must be > 0")

        # Ensure text is a valid string to prevent 'NoneType has no replace' errors
        text = str(text) if text is not None else ""

        if not distance_function:
            return index.similarity_search_with_score(text, k=top_k)

        metric_map = {"euclidean": "l2", "inner_product": "ip"}
        normalized = metric_map.get(distance_function, distance_function)

        try:
            index_metric = index._collection.metadata.get("hnsw:space", "cosine")
            if normalized == index_metric:
                return index.similarity_search_with_score(text, k=top_k)
        except (AttributeError, TypeError):
            pass

        return self._rerank_with_metric(index, text, top_k, normalized, fetch_k or top_k * 3)

    def _rerank_with_metric(
        self, index: VectorStoreLike, text: str, top_k: int, metric: str, fetch_k: int
    ) -> list[tuple[Document, float]]:
        """Fetch candidates with index's metric, then re-rank with requested metric."""
        query_vec = np.array(self._embeddings.embed_query(text))
        candidates = index.similarity_search_by_vector(query_vec.tolist(), k=fetch_k)
        embeddings = self._get_embeddings(index, candidates)

        scored = [
            (doc, self._compute_distance(query_vec, np.array(emb), metric))
            for doc, emb in zip(candidates, embeddings)
        ]

        scored.sort(key=lambda x: x[1])
        if metric == "ip":
            return [(doc, -score) for doc, score in scored[:top_k]]
        return scored[:top_k]

    def batch_retrieve(
        self,
        index: VectorStoreLike | None,
        questions: list[str],
        *,
        top_k: int = 5,
        batch_size: int = 32,
    ) -> list[list[tuple[Document, float]]]:
        """Batch retrieve documents for multiple queries using efficient embedding."""
        if not index:
            raise ValueError("Index required")
        
        all_embeddings = []
        clean_questions = [str(q) if q is not None else "" for q in questions]
        
        iterator = range(0, len(clean_questions), batch_size)
        if len(clean_questions) > 100:
             iterator = tqdm(iterator, desc="Embedding queries", unit="batch")

        for i in iterator:
            batch = clean_questions[i : i + batch_size]
            if not batch: continue
            try:
                batch_embeddings = self._embeddings.embed_documents(batch)
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Error embedding batch: {e}")
                raise e

        all_results = []
        has_vector_with_score = hasattr(index, "similarity_search_by_vector_with_score")
        
        for vec in all_embeddings:
            if has_vector_with_score:
                results = index.similarity_search_by_vector_with_score(vec, k=top_k)
            else:
                docs = index.similarity_search_by_vector(vec, k=top_k)
                results = [(d, 0.0) for d in docs]
            all_results.append(results)
            
        return all_results

    def retrieve_topk_by_metric(
        self,
        index: VectorStoreLike | None,
        questions: list[str],
        expected_ids: list[str],
        *,
        top_k: int = 5,
        metrics: list[str] | None = None,
        batch_size: int = 64
    ) -> pd.DataFrame:
        """
        Retrieve top-k results for multiple distance metrics in one pass.

        Unifies embedding generation to process multiple distance metrics (e.g. cosine, l2)
        on the same set of queries without re-embedding.
        
        Args:
            metrics: List of metrics to evaluate (e.g., ["cosine", "l2", "ip"]).
                     If None, uses the index's native metric.
        """
        if not index:
            raise ValueError("Index required")
        
        if len(questions) != len(expected_ids):
            raise ValueError(f"Length mismatch: {len(questions)} questions vs {len(expected_ids)} expected IDs")

        results_data = []
        # Normalize metrics list
        metric_map = {"euclidean": "l2", "inner_product": "ip"}
        target_metrics = metrics if metrics else ["native"]
        target_metrics = [metric_map.get(m, m) for m in target_metrics]
        # Preserve order but remove duplicates
        seen = set()
        target_metrics = [m for m in target_metrics if not (m in seen or seen.add(m))]

        # Determine internal index metric if possible
        index_metric = "cosine"
        if hasattr(index, "_collection"):
            index_metric = index._collection.metadata.get("hnsw:space", "cosine")

        total = len(questions)
        iterator = range(0, total, batch_size)
        if total > 100:
            desc = f"Evaluating ({', '.join(target_metrics)})"
            iterator = tqdm(iterator, desc=desc, unit="batch")

        for i in iterator:
            batch_qs = questions[i : i + batch_size]
            batch_ids = expected_ids[i : i + batch_size]
            
            clean_batch = [str(q) if q is not None else "" for q in batch_qs]
            try:
                batch_embeddings = self._embeddings.embed_documents(clean_batch)
            except Exception as e:
                logger.error(f"Error embedding batch {i}-{i+batch_size}: {e}")
                raise e
            
            has_vector_with_score = hasattr(index, "similarity_search_by_vector_with_score")

            for j, query_vec in enumerate(batch_embeddings):
                expected_id = str(batch_ids[j]).strip()
                query_vec_np = np.asarray(query_vec)

                # Determine whether we need re-ranking for non-native metrics
                needs_rerank = any(m not in ["native", index_metric] for m in target_metrics)
                candidates = None
                cand_embeddings = None
                cand_embs_np = None
                fetch_k = top_k * 3
                if needs_rerank:
                    candidates = index.similarity_search_by_vector(query_vec, k=fetch_k)
                    cand_embeddings = self._get_embeddings(index, candidates)
                    cand_embs_np = np.asarray(cand_embeddings)
                    # Guard against empty candidates
                    if cand_embs_np.size == 0:
                        cand_embs_np = None

                # Evaluate for EACH requested metric
                for current_metric in target_metrics:

                    docs_and_scores = []

                    if current_metric == "native" or current_metric == index_metric:
                        # Use fast native search
                        if has_vector_with_score:
                            docs_and_scores = index.similarity_search_by_vector_with_score(query_vec, k=top_k)
                        else:
                            docs = index.similarity_search_by_vector(query_vec, k=top_k)
                            docs_and_scores = [(d, 0.0) for d in docs]
                    else:
                        # Re-rank using pre-fetched candidates/embeddings (vectorized)
                        if candidates is None or cand_embs_np is None:
                            docs_and_scores = []
                        else:
                            if current_metric == "cosine":
                                denom = (np.linalg.norm(cand_embs_np, axis=1) * np.linalg.norm(query_vec_np))
                                denom[denom == 0] = 1e-12
                                scores = 1.0 - (cand_embs_np @ query_vec_np) / denom
                            elif current_metric == "l2":
                                scores = np.linalg.norm(cand_embs_np - query_vec_np, axis=1)
                            elif current_metric == "ip":
                                scores = -(cand_embs_np @ query_vec_np)
                            else:
                                scores = np.array([
                                    self._compute_distance(query_vec_np, emb, current_metric)
                                    for emb in cand_embs_np
                                ])

                            top_idx = np.argsort(scores)[:top_k]
                            if current_metric == "ip":
                                docs_and_scores = [(candidates[i], -scores[i]) for i in top_idx]
                            else:
                                docs_and_scores = [(candidates[i], scores[i]) for i in top_idx]

                    topk_ids = []
                    topk_scores = []
                    topk_popularities = []

                    for doc, score in docs_and_scores:
                        doc_id = doc.metadata.get("wikipedia_id", None)
                        if doc_id is None:
                            doc_id = doc.metadata.get("id", "")
                        topk_ids.append(str(doc_id).strip())
                        topk_scores.append(score)
                        topk_popularities.append(doc.metadata.get("popularity_avg", None))

                    results_data.append({
                        "question": batch_qs[j],
                        "wikipedia_id": expected_id,
                        "metric": current_metric,  # Capture which metric this result is for
                        "topk_ids": topk_ids,
                        "topk_scores": topk_scores,
                        "topk_popularities": topk_popularities
                    })

        return pd.DataFrame(results_data)

    def evaluate_retrieval(
        self,
        index: VectorStoreLike | None,
        questions: list[str],
        expected_ids: list[str],
        *,
        top_k: int = 5,
        metrics: list[str] | None = None,
        batch_size: int = 64
    ) -> pd.DataFrame:
        """Backward-compatible alias for retrieve_topk_by_metric (no evaluation)."""
        return self.retrieve_topk_by_metric(
            index=index,
            questions=questions,
            expected_ids=expected_ids,
            top_k=top_k,
            metrics=metrics,
            batch_size=batch_size
        )

    def _get_embeddings(self, index: VectorStoreLike, docs: list[Document]) -> list[list[float]]:
        """Get embeddings from index or re-embed documents."""
        try:
            ids = [d.metadata.get("_id") for d in docs if d.metadata.get("_id")]
            result = index._collection.get(ids=ids, include=["embeddings"])
            if result and result.get("embeddings"):
                return result["embeddings"]
        except Exception:
            pass
        return [self._embeddings.embed_query(doc.page_content) for doc in docs]

    def _compute_distance(self, query_vec: np.ndarray, doc_vec: np.ndarray, metric: str) -> float:
        """Compute distance between vectors (expects normalized metric names)."""
        if metric == "cosine":
            similarity = np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
            return 1.0 - similarity
        if metric == "l2":
            return float(np.linalg.norm(query_vec - doc_vec))
        if metric == "ip":
            return -float(np.dot(query_vec, doc_vec))
        raise ValueError(f"Invalid metric: {metric}. Expected: cosine, l2, ip")
    