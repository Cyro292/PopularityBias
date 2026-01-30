"""BM25-based RAG service for keyword-based retrieval.

BM25 (Best Match 25) is a ranking function used by search engines to estimate
the relevance of documents to a given search query. Unlike dense vector embeddings,
BM25 uses sparse keyword matching and is particularly effective for:
- Exact keyword matches
- Domain-specific terminology
- Queries where semantic similarity is less important
- Lower latency and no API costs (no embedding model needed)

This implementation uses the rank-bm25 library and follows the LangChain BM25Retriever pattern.

Reference: https://python.langchain.com/docs/integrations/retrievers/bm25
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
import pyarrow.parquet as pq
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever

from .base import RagService, VectorStoreLike
from .document_utils import documents_from_text_dataframe

logger = logging.getLogger(__name__)


class BM25Index:
    """Wrapper for BM25Retriever to match VectorStoreLike protocol.

    This class provides a consistent interface for BM25 retrieval that matches
    the vector store interface used by other RAG services.
    """

    def __init__(self, retriever: BM25Retriever):
        """Initialize BM25 index wrapper.

        Args:
            retriever: LangChain BM25Retriever instance.
        """
        self.retriever = retriever

    def similarity_search(self, query: str, k: int = 5, **kwargs) -> list[Document]:
        """Search for documents similar to the query using BM25 scoring.

        Args:
            query: Query text.
            k: Number of documents to return.
            **kwargs: Additional arguments (ignored for BM25).

        Returns:
            List of matching documents, ranked by BM25 score.
        """
        # BM25Retriever uses invoke() in newer versions, get_relevant_documents() in older ones
        if hasattr(self.retriever, 'invoke'):
            self.retriever.k = k
            return self.retriever.invoke(query)
        else:
            self.retriever.k = k
            return self.retriever.get_relevant_documents(query)

    def similarity_search_by_vector(self, embedding: Sequence[float], k: int = 4, **kwargs) -> list[Document]:
        """Not supported for BM25 (keyword-based, not vector-based).

        Args:
            embedding: Query embedding (not used).
            k: Number of results.
            **kwargs: Additional arguments.

        Raises:
            NotImplementedError: BM25 doesn't use vector embeddings.
        """
        raise NotImplementedError("BM25 is keyword-based and does not support vector search")

    def save(self, path: Path):
        """Persist the BM25 index to disk.

        Args:
            path: Path to save the index.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.retriever, f)
        logger.info(f"Saved BM25 index to {path}")

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        """Load a BM25 index from disk.

        Args:
            path: Path to the saved index.

        Returns:
            Loaded BM25Index instance.
        """
        with open(path, 'rb') as f:
            retriever = pickle.load(f)
        logger.info(f"Loaded BM25 index from {path}")
        return cls(retriever)


class BM25RagService(RagService):
    """RAG Service using BM25 keyword-based retrieval.

    BM25 is a probabilistic retrieval function that ranks documents based on
    term frequency and inverse document frequency. It's particularly useful:
    - When you don't want to pay for embedding API calls
    - For exact keyword matching scenarios
    - As a baseline for comparing against dense retrieval
    - For hybrid retrieval (combining BM25 + vector search)

    Example:
        >>> service = BM25RagService()
        >>> index, count = service.index_from_parquet(
        ...     parquet_path=Path("data.parquet"),
        ...     text_field="content",
        ...     output_dir=Path("./bm25_index")
        ... )
        >>> results = service.retrieve_documents(index, "your query", top_k=5)
    """

    def __init__(self, k: int = 5):
        """Initialize BM25 RAG service.

        Args:
            k: Default number of documents to retrieve.
        """
        self.k = k

    def index_from_parquet(
        self,
        parquet_path: Path,
        output_dir: Path,
        *,
        text_field: str | None = None,
        html_field: str | None = None,
        metadata_fields: Sequence[str] | None = None,
        collection_name: str = "bm25",
    ) -> tuple[BM25Index, int]:
        """Load text from Parquet and create BM25 index.

        Args:
            parquet_path: Path to the Parquet file.
            output_dir: Directory to save the index.
            text_field: Column name containing text content.
            html_field: Column name containing HTML content (not yet supported).
            metadata_fields: Optional column names to include as metadata.
            collection_name: Name for the index file.

        Returns:
            Tuple of (BM25Index, number of documents indexed).

        Raises:
            ValueError: If neither or both text_field and html_field are provided.
            NotImplementedError: If html_field is used (not yet supported).
        """
        if text_field is None and html_field is None:
            raise ValueError("Either text_field or html_field must be provided")
        if text_field is not None and html_field is not None:
            raise ValueError("Only one of text_field or html_field should be provided")
        if html_field is not None:
            raise NotImplementedError("HTML field indexing is not yet supported for BM25")

        metadata_fields_tuple = tuple(metadata_fields or ())

        logger.info(f"Reading {parquet_path}")
        df = pq.read_table(parquet_path, columns=[text_field, *metadata_fields_tuple]).to_pandas()
        logger.info(f"Loaded {len(df)} rows")

        documents = documents_from_text_dataframe(
            df,
            text_field,
            metadata_fields_tuple,
            source=str(parquet_path),
            row_offset=0,
        )
        logger.info(f"Converted {len(df)} rows to {len(documents)} documents")

        # Create BM25 retriever
        retriever = BM25Retriever.from_documents(documents)
        retriever.k = self.k

        index = BM25Index(retriever)

        # Save to disk
        output_dir.mkdir(parents=True, exist_ok=True)
        index_path = output_dir / f"{collection_name}.pkl"
        index.save(index_path)

        logger.info(f"Created BM25 index with {len(documents)} documents")
        return index, len(documents)

    def index_from_dataframe(
        self,
        df: pd.DataFrame,
        text_field: str,
        html_field: str | None = None,
        *,
        metadata_fields: Sequence[str] | None = None,
        output_dir: Path | None = None,
        collection_name: str = "bm25",
    ) -> tuple[BM25Index, int]:
        """Create a BM25 index from a pandas DataFrame.

        Args:
            df: DataFrame containing the text data.
            text_field: Column name containing text content.
            html_field: Column name containing HTML (not supported).
            metadata_fields: Optional column names to include as metadata.
            output_dir: Optional directory to save the index. If None, index stays in memory.
            collection_name: Name for the index file.

        Returns:
            Tuple of (BM25Index, number of documents indexed).

        Raises:
            NotImplementedError: If html_field is provided.
        """
        if html_field is not None:
            raise NotImplementedError("HTML field indexing is not supported for BM25")

        metadata_fields_tuple = tuple(metadata_fields or ())
        logger.info(f"Indexing DataFrame with {len(df)} rows")

        documents = documents_from_text_dataframe(
            df,
            text_field,
            metadata_fields_tuple,
            source="dataframe",
            row_offset=0,
        )
        logger.info(f"Converted {len(df)} rows to {len(documents)} documents")

        # Create BM25 retriever
        retriever = BM25Retriever.from_documents(documents)
        retriever.k = self.k

        index = BM25Index(retriever)

        # Save to disk if output_dir provided
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            index_path = output_dir / f"{collection_name}.pkl"
            index.save(index_path)

        logger.info(f"Created BM25 index with {len(documents)} documents")
        return index, len(documents)

    def load_index(self, output_dir: Path, collection_name: str = "bm25") -> Optional[BM25Index]:
        """Load existing BM25 index from disk.

        Args:
            output_dir: Directory containing the saved index.
            collection_name: Name of the index file (without .pkl extension).

        Returns:
            Loaded BM25Index, or None if not found.
        """
        index_path = output_dir / f"{collection_name}.pkl"
        if not index_path.exists():
            logger.warning(f"No BM25 index found at {index_path}")
            return None

        return BM25Index.load(index_path)

    def retrieve_documents(
        self,
        index: VectorStoreLike | None,
        text: str,
        *,
        top_k: int = 5,
    ) -> list[Document]:
        """Retrieve documents using BM25 keyword matching.

        Args:
            index: BM25Index instance.
            text: Query text.
            top_k: Number of documents to return.

        Returns:
            List of matching documents, ranked by BM25 score.

        Raises:
            ValueError: If index is None or not a BM25Index.
        """
        if not index:
            raise ValueError("Index is required")

        if not isinstance(index, BM25Index):
            raise ValueError("Index must be a BM25Index instance")

        return index.similarity_search(text, k=top_k)
