"""Abstract RAG service contract for future implementations."""

from __future__ import annotations

import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Protocol, Sequence

from langchain.schema import Document


class VectorStoreLike(Protocol):
    """Minimal interface representing an object that can forward similarity queries."""

    def similarity_search_by_vector(self, embedding: Sequence[float], k: int = 4, **kwargs: Sequence[float]) -> list[Document]:
        ...


IndexResult = tuple[Optional[VectorStoreLike], int]


class RagService(ABC):
    """Abstract definition of a retrieval-augmented generation service."""

    @abstractmethod
    def index_from_dataframe(
        self,
        df: pd.DataFrame,
        text_field: str,
        html_field: str | None = None,
        *,
        metadata_fields: Sequence[str] | None = None,
        output_dir: Path | None = None,
        collection_name: str = "rag",
    ) -> IndexResult:
        """Create a Chroma index from a pandas DataFrame.
        
        Args:
            df: DataFrame containing the text data to index.
            text_field: Column name containing the text content.
            metadata_fields: Optional column names to include as document metadata.
            output_dir: Optional directory to persist the index. If None, index is in-memory.
            collection_name: Name for the Chroma collection.
        """

        raise NotImplementedError("index_from_dataframe is not implemented")

    @abstractmethod 
    def index_from_parquet(
        self,
        parquet_path: Path,
        output_dir: Path,
        *,
        text_field: str | None = None,
        html_field: str | None = None,
        metadata_fields: Sequence[str] | None = None,
        collection_name: str = "rag",
    ) -> IndexResult:
        """Load text or HTML from Parquet into whatever storage the service owns.
        
        Args:
            parquet_path: Path to the Parquet file.
            output_dir: Directory to persist the index.
            text_field: Column name containing text content. Mutually exclusive with html_field.
            html_field: Column name containing HTML content. Mutually exclusive with text_field.
            metadata_fields: Optional column names to include as document metadata.
            collection_name: Name for the collection.
            
        Raises:
            ValueError: If neither or both text_field and html_field are provided.
        """
        raise NotImplementedError("index_from_parquet is not implemented")

    @abstractmethod
    def retrieve_documents(
        self,
        index: Optional[VectorStoreLike],
        text: str,
        *,
        top_k: int = 5,
    ) -> list[Document]:
        """Return documents matching an embedding using the configured similarity strategy."""

        raise NotImplementedError("retrieve_documents is not implemented")