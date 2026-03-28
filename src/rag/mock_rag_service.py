"""Lightweight RAG service used for testing and prototyping."""

from __future__ import annotations

import logging
from pathlib import Path
from random import sample
from typing import Sequence

import pyarrow.parquet as pq
from langchain.schema import Document

from .base import RagService, VectorStoreLike
from .document_utils import documents_from_text_dataframe, documents_from_html_dataframe

logger = logging.getLogger(__name__)


class mockRagService(RagService):
    """Minimal implementation that only stores documents without embeddings."""

    def __init__(self):
        self.documents: list[Document] = []

    def index_from_parquet(
        self,
        parquet_path: Path,
        output_dir: Path,
        *,
        text_field: str | None = None,
        html_field: str | None = None,
        metadata_fields: Sequence[str] | None = None,
        collection_name: str = "rag",
    ) -> tuple[VectorStoreLike | None, int]:
        """Load text or HTML column from Parquet and persist it to mock storage."""
        if text_field is None and html_field is None:
            raise ValueError("Either text_field or html_field must be provided")
        if text_field is not None and html_field is not None:
            raise ValueError("Only one of text_field or html_field should be provided")
        
        field = html_field if html_field else text_field
        columns = [field] + list(metadata_fields or ())
        table = pq.read_table(parquet_path, columns=columns)
        batch_df = table.to_pandas()

        doc_creator = documents_from_html_dataframe if html_field else documents_from_text_dataframe
        self.documents = doc_creator(
            batch_df,
            field,
            metadata_fields,
            source=str(parquet_path),
            row_offset=0,
        )
        logger.info("Mock RAG saved %d documents", len(self.documents))
        return None, len(self.documents)

    def retrieve_documents(
        self,
        index: VectorStoreLike | None,
        text: str,
        *,
        top_k: int = 5,
    ) -> list[Document]:
        """Return a random subset of stored documents."""
        if top_k <= 0 or not self.documents:
            return []

        chosen = sample(self.documents, k=min(top_k, len(self.documents)))
        return [(doc, 0.0) for doc in chosen]
