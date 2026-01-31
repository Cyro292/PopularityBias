"""RAG (Retrieval-Augmented Generation) Service for Document Indexing.

This module provides a comprehensive service for indexing and querying documents using:
- Chroma vector store for efficient similarity search
- Multiple data sources: Parquet files, pandas DataFrames, HuggingFace Datasets
- Text chunking for optimal retrieval
- Multiple embedding providers (OpenAI, Google, HuggingFace)
- Rate limiting to prevent API throttling
- Memory-efficient batch processing for large datasets
- Resume capability for interrupted indexing jobs

REFACTORED: This service now uses modular components from:
- embeddings.py: Embedding providers and rate limiting
- indexing.py: Chunking, index creation, and persistence
- retrieval.py: Distance metrics and batch retrieval
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from langchain.schema import Document
from langchain_chroma import Chroma
from tqdm import tqdm

from .base import RagService, VectorStoreLike
from .document_utils import documents_from_html_dataframe, documents_from_text_arrow, documents_from_text_dataframe
from .utils import (
    IndexingConfig,
    batch_retrieve as _batch_retrieve,
    build_embeddings,
    create_chroma_index,
    prepare_persist_dir,
    rerank_with_metric,
    retrieve_topk_by_metric as _retrieve_topk_by_metric,
    split_documents,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


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
        """Initialize RAG service with configuration.

        Args:
            config: IndexingConfig instance. If None, uses default configuration.
        """
        self.config = config or IndexingConfig()
        self.distance_function = self.config.distance_function
        self._embeddings = build_embeddings(
            provider=self.config.embedding_provider,
            model=self.config.embedding_model,
            trust_remote_code=self.config.trust_remote_code,
            rate_limiter=self.config.rate_limiter,
            requests_per_second=self.config.requests_per_second,
            check_interval=self.config.rate_limit_check_interval,
            bucket_size=self.config.rate_limit_bucket_size,
        )

        # Load embedding prompt template
        self._load_embedding_prompt()

    def _load_embedding_prompt(self):
        """Load embedding prompt templates from files.

        The prompts are applied to document/query content before embedding to improve
        retrieval quality. Some embedding models (like E5, BGE) benefit from
        instructional prefixes.
        """
        from config import DATA_DIR

        # Load passage prompt (for documents)
        passage_prompt_file = Path(DATA_DIR) / "prompts" / "embeding_promt.txt"
        if passage_prompt_file.exists():
            self.embedding_prompt = passage_prompt_file.read_text().strip()
            logger.info(f"Loaded embedding prompt: {self.embedding_prompt[:50]}...")
        else:
            # Default prompt if file doesn't exist
            self.embedding_prompt = "passage: {passage}"
            logger.warning(f"Embedding prompt file not found at {passage_prompt_file}, using default")

        # Load query prompt (for retrieval)
        query_prompt_file = Path(DATA_DIR) / "prompts" / "query_promt.txt"
        if query_prompt_file.exists():
            self.query_prompt = query_prompt_file.read_text().strip()
            logger.info(f"Loaded query prompt: {self.query_prompt[:50]}...")
        else:
            # Default prompt if file doesn't exist
            self.query_prompt = "query: {query}"
            logger.warning(f"Query prompt file not found at {query_prompt_file}, using default")

    def _apply_embedding_prompt(self, documents: list[Document]) -> list[Document]:
        """Apply embedding prompt template to documents.

        Wraps each document's content with the embedding prompt template.
        This improves retrieval quality for models trained with instruction prefixes.

        Args:
            documents: Documents to apply prompt to.

        Returns:
            Documents with prompted content.
        """
        prompted_docs = []
        for doc in documents:
            # Apply prompt template
            prompted_content = self.embedding_prompt.format(passage=doc.page_content)

            # Create new document with prompted content
            prompted_docs.append(Document(
                page_content=prompted_content,
                metadata=doc.metadata.copy()
            ))

        return prompted_docs

    def _apply_query_prompt(self, query: str) -> str:
        """Apply query prompt template to search query.

        Wraps the query with the query prompt template.
        This improves retrieval quality for models trained with instruction prefixes.

        Args:
            query: Query string to apply prompt to.

        Returns:
            Prompted query string.
        """
        return self.query_prompt.format(query=query)

    def _prepare_documents(self, documents: list[Document]) -> list[Document]:
        """Apply chunking and embedding prompt if configured.

        Args:
            documents: Documents to prepare.

        Returns:
            Processed documents (chunked + prompted).
        """
        # Step 1: Chunk documents if configured
        if self.config.chunk_size:
            documents = split_documents(documents, self.config.chunk_size, self.config.chunk_overlap)

        # Step 2: Apply embedding prompt template
        documents = self._apply_embedding_prompt(documents)

        return documents

    def _build_index(
        self,
        documents: list[Document],
        collection_name: str,
        output_dir: Path | None = None,
        progress_bar: bool = False,
    ) -> tuple[Chroma, int]:
        """Build Chroma index from documents.

        Args:
            documents: Documents to index (should be pre-chunked).
            collection_name: Name for the Chroma collection.
            output_dir: Optional directory to persist the index.
            progress_bar: Whether to show progress during indexing.

        Returns:
            Tuple of (Chroma vectorstore, number of document chunks indexed).
        """
        persist_dir = prepare_persist_dir(output_dir)
        logger.info(f"Creating {'persistent' if persist_dir else 'in-memory'} index with {len(documents)} docs")

        vectorstore = create_chroma_index(
            documents=documents,
            embeddings=self._embeddings,
            collection_name=collection_name,
            persist_dir=str(persist_dir) if persist_dir else None,
            show_progress=progress_bar,
            distance_function=self.distance_function,
        )
        logger.info(f"Created index with {len(documents)} chunks using {self.distance_function} distance")
        return vectorstore, len(documents)

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

        field = html_field or text_field
        meta_fields = tuple(metadata_fields or ())

        logger.info(f"Reading {parquet_path}")
        df = pq.read_table(parquet_path, columns=[field, *meta_fields]).to_pandas()
        logger.info(f"Loaded {len(df)} rows")

        doc_creator = documents_from_html_dataframe if html_field else documents_from_text_dataframe
        documents = doc_creator(df, field, meta_fields, source=str(parquet_path), row_offset=0)

        documents = self._prepare_documents(documents)
        return self._build_index(documents, collection_name, output_dir, progress_bar)

    def index_from_dataframe(
        self,
        df: pd.DataFrame,
        text_field: str,
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
            html_field: Column name containing HTML content (not yet supported).
            metadata_fields: Optional column names to include as document metadata.
            output_dir: Optional directory to persist the index. If None, index is in-memory.
            collection_name: Name for the Chroma collection.
            progress_bar: Show progress bar during indexing.

        Returns:
            Tuple of (Chroma vectorstore, number of document chunks indexed).

        Raises:
            NotImplementedError: If html_field is provided.
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
            metadata_fields: Optional column names to include as metadata.
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
                ds,
                text_field,
                html_field,
                metadata_fields_tuple,
                output_dir,
                collection_name,
                progress_bar,
                batch_size,
                resume_from_row,
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
            persist_dir = prepare_persist_dir(output_dir)
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
                                documents=chunk,
                                embedding=self._embeddings,
                                collection_name=collection_name,
                                persist_directory=str(persist_dir) if persist_dir else None,
                                collection_metadata=collection_metadata,
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
        """Load existing Chroma index from disk.

        Args:
            output_dir: Directory containing the saved index.
            collection_name: Name of the Chroma collection.

        Returns:
            Loaded Chroma vectorstore, or None if not found.
        """
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

    def get_corpus_dataframe(
        self,
        index: VectorStoreLike | None,
        *,
        batch_size: int = 10_000,
    ) -> pd.DataFrame:
        """Retrieve the full corpus from a Chroma index as a DataFrame.

        Returns a DataFrame with a `text` column and all metadata fields found
        on documents.

        Args:
            index: Vector store to read from.
            batch_size: Number of documents to fetch per batch.

        Returns:
            DataFrame with columns: text + metadata keys.

        Raises:
            ValueError: If index is None.
        """
        if not index:
            raise ValueError("Index is required")

        collection = index._collection
        total = collection.count()
        rows: list[dict[str, Any]] = []

        for offset in range(0, total, batch_size):
            batch = collection.get(
                include=["documents", "metadatas"],
                limit=batch_size,
                offset=offset,
            )
            documents = batch.get("documents") or []
            metadatas = batch.get("metadatas") or []

            if not metadatas:
                metadatas = [{} for _ in range(len(documents))]

            for text, meta in zip(documents, metadatas):
                row = {"text": text}
                if meta:
                    row.update(meta)
                rows.append(row)

        return pd.DataFrame(rows)

    def add_documents(self, index: VectorStoreLike, documents: Sequence[Document]):
        """Add documents to an existing index.

        Args:
            index: Vector store to add documents to.
            documents: Documents to add.

        Returns:
            Updated index.
        """
        return index.add_documents(documents) or index

    def delete_documents(self, index: VectorStoreLike, document_ids: Sequence[str]):
        """Delete documents from index by ID.

        Args:
            index: Vector store to delete from.
            document_ids: List of document IDs to delete.

        Returns:
            Updated index.
        """
        return index.delete(document_ids) or index

    def get_document_by_id(self, index: VectorStoreLike, document_id: int | str) -> list[Document]:
        """Get documents by metadata 'id' field.

        Args:
            index: Vector store to search.
            document_id: Document ID to retrieve.

        Returns:
            List of matching documents.
        """
        results = index._collection.get(where={"id": {"$eq": int(document_id)}}, include=["documents", "metadatas"])
        return [
            Document(page_content=text, metadata=results["metadatas"][i] if results.get("metadatas") else {})
            for i, text in enumerate(results.get("documents", []))
        ]

    def retrieve_documents(
        self,
        index: VectorStoreLike | None,
        text: str,
        *,
        top_k: int = 5,
        distance_function: str | None = None,
        fetch_k: int | None = None,
    ) -> list[Document]:
        """Return documents matching query, optionally with custom distance metric.

        Args:
            index: Vector store to search.
            text: Query text.
            top_k: Number of results to return.
            distance_function: Distance metric to use (None for native, "cosine", "l2", "ip").
            fetch_k: Number of candidates to fetch for re-ranking (default: top_k * 3).

        Returns:
            List of matching documents.

        Raises:
            ValueError: If index is None or top_k <= 0.
        """
        # Apply query prompt
        prompted_text = self._apply_query_prompt(text)

        if distance_function:
            return [
                doc
                for doc, _ in self.retrieve_documents_with_scores(
                    index, prompted_text, top_k=top_k, distance_function=distance_function, fetch_k=fetch_k
                )
            ]
        if not index or top_k <= 0:
            raise ValueError("Index required and top_k must be > 0")
        return index.similarity_search(prompted_text, k=top_k)

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

        Raises:
            ValueError: If index is None or top_k <= 0.
        """
        if not index or top_k <= 0:
            raise ValueError("Index required and top_k must be > 0")

        # Ensure text is a valid string to prevent 'NoneType has no replace' errors
        text = str(text) if text is not None else ""

        # Apply query prompt
        prompted_text = self._apply_query_prompt(text)

        if not distance_function:
            return index.similarity_search_with_score(prompted_text, k=top_k)

        metric_map = {"euclidean": "l2", "inner_product": "ip"}
        normalized = metric_map.get(distance_function, distance_function)

        try:
            index_metric = index._collection.metadata.get("hnsw:space", "cosine")
            if normalized == index_metric:
                return index.similarity_search_with_score(prompted_text, k=top_k)
        except (AttributeError, TypeError):
            pass

        return rerank_with_metric(
            index=index,
            embeddings=self._embeddings,
            text=prompted_text,
            top_k=top_k,
            metric=normalized,
            fetch_k=fetch_k or top_k * 3,
        )

    def batch_retrieve(
        self,
        index: VectorStoreLike | None,
        questions: list[str],
        *,
        top_k: int = 5,
        batch_size: int = 32,
    ) -> list[list[tuple[Document, float]]]:
        """Batch retrieve documents for multiple queries using efficient embedding.

        Args:
            index: Vector store to search.
            questions: List of query texts.
            top_k: Number of results per query.
            batch_size: Number of queries to embed at once.

        Returns:
            List of results, one per query. Each result is a list of (Document, score) tuples.

        Raises:
            ValueError: If index is None.
        """
        return _batch_retrieve(
            index=index,
            embeddings=self._embeddings,
            questions=questions,
            top_k=top_k,
            batch_size=batch_size,
        )

    def retrieve_topk_by_metric(
        self,
        index: VectorStoreLike | None,
        questions: list[str],
        expected_ids: list[str],
        *,
        top_k: int = 5,
        metrics: list[str] | None = None,
        batch_size: int = 64,
    ) -> pd.DataFrame:
        """Retrieve top-k results for multiple distance metrics in one pass.

        Unifies embedding generation to process multiple distance metrics (e.g. cosine, l2)
        on the same set of queries without re-embedding.

        Args:
            index: Vector store to search.
            questions: Query texts.
            expected_ids: Expected document IDs for each query.
            top_k: Number of results per query.
            metrics: List of metrics to evaluate (e.g., ["cosine", "l2", "ip"]).
                     If None, uses the index's native metric.
            batch_size: Number of queries to process at once.

        Returns:
            DataFrame with columns: question, wikipedia_id, metric, topk_ids, topk_scores, topk_popularities.

        Raises:
            ValueError: If index is None or questions/expected_ids length mismatch.
        """
        results = _retrieve_topk_by_metric(
            index=index,
            embeddings=self._embeddings,
            questions=questions,
            expected_ids=expected_ids,
            top_k=top_k,
            metrics=metrics,
            batch_size=batch_size,
        )
        return pd.DataFrame(results)
    