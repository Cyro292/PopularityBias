"""Abstract RAG service contract for future implementations."""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Optional, Protocol, Sequence

from langchain.schema import Document
from langchain.vectorstores import VectorStore


IndexResult = tuple[Optional[VectorStore], int]


class RagService():
    """Abstract definition of a retrieval-augmented generation service."""

    def index_from_dataframe(
        self,
        df: pd.DataFrame,
        text_field: str,
        html_field: str | None = None,
        metadata_fields: Sequence[str] | None = None,
        output_dir: Path | None = None,
        collection_name: str = "rag",
        *args, 
        **kwargs,
    ) -> IndexResult:
        """Create a inde index from a pandas DataFrame."""

        raise NotImplementedError("index_from_dataframe is not implemented")
    
    def batch_index_from_dataframe(
        self,
        df: pd.DataFrame,
        text_field: str,
        html_field: str | None = None,
        metadata_fields: Sequence[str] | None = None,
        output_dir: Path | None = None,
        collection_name: str | None = None,
        batch_size: int = 1000,
        *args, 
        **kwargs,
    ) -> IndexResult:
        """Batch version of index_from_dataframe."""
        raise NotImplementedError("batch_index_from_dataframe is not implemented")

    def index_from_parquet(
        self,
        parquet_path: Path,
        output_dir: Path,
        text_field: str | None = None,
        html_field: str | None = None,
        metadata_fields: Sequence[str] | None = None,
        collection_name: str = "rag",
        *args, 
        **kwargs,
    ) -> IndexResult:
        """Create a index from a Parquet file."""
        raise NotImplementedError("index_from_parquet is not implemented")


    def index_from_parquet_batches(
        self,
        parquet_path: Path,
        output_dir: Path,
        text_field: str | None = None,
        metadata_fields: Sequence[str] | None = None,
        collection_name: str | None = None,
        batch_size: int = 1000,
        *args, **kwargs,
    ) -> IndexResult:
        """Batch version of index_from_parquet."""
        raise NotImplementedError("index_from_parquet_batches is not implemented")

    def retrieve_document(
        self,
        text: str,
        top_k: int = 5,
        *args, 
        **kwargs,
    ) -> list[Document]:
        """Return documents matching an embedding using the configured similarity strategy."""

        raise NotImplementedError("retrieve_document is not implemented")
    
    def batch_retrieve(
        self,
        texts: list[str],
        *args, 
        **kwargs,
    ) -> list[list[list[Document]]]:
        """Batch version of retrieve_document."""
        raise NotImplementedError("batch_retrieve is not implemented")
    
    def batch_retrieve_with_scores(
        self,
        texts: list[str],
        top_k: int = 5,
        *args, 
        **kwargs,
    ) -> list[list[tuple[Document, float]]]:
        """Return documents and similarity scores matching an embedding using the configured similarity strategy."""
        raise NotImplementedError("batch_retrieve_with_scores is not implemented")
    
    