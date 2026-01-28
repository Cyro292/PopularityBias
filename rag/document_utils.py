"""Helpers for turning Parquet batches into LangChain documents."""

from __future__ import annotations

from typing import Any, Sequence

import pandas as pd
from bs4 import BeautifulSoup
from langchain.schema import Document
import pyarrow as pa


def _normalize_metadata_value(value: Any) -> str | int | float | bool | None:
    """Normalize metadata so Chroma receives only supported scalar types."""
    if value is None:
        return None
    if isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, float)):
        return value

    if hasattr(value, "item"):
        try:
            scalar = value.item()
            if isinstance(scalar, (str, int, float, bool)):
                return scalar
        except Exception:
            pass

    if isinstance(value, dict):
        text_value = value.get("text")
        if isinstance(text_value, (str, int, float, bool)):
            return text_value

    return str(value)

def documents_from_text_arrow(
        table: pa.Table,
        text_field: str,
        metadata_fields: tuple[str, ...],
        source: str,
        row_offset: int = 0,
    ) -> list[Document]:
        """
        Convert an Arrow table into LangChain Documents without pandas.

        Assumes:
        - text_field exists in table
        - metadata_fields exist in table
        """
        text_col = table[text_field]

        meta_cols = {name: table[name] for name in metadata_fields}

        documents: list[Document] = []

        for i in range(table.num_rows):
            text = text_col[i].as_py()
            if not text:
                continue

            metadata = {
                "source": source,
                "row": row_offset + i,
            }

            for name, col in meta_cols.items():
                val = col[i].as_py()
                if val is not None:
                    metadata[name] = val

            documents.append(
                Document(
                    page_content=text,
                    metadata=metadata,
                )
            )

        return documents

def documents_from_text_dataframe(
    batch_df: pd.DataFrame,
    text_field: str,
    metadata_fields: Sequence[str] | None,
    source: str,
    row_offset: int,
) -> list[Document]:
    """Extract Document objects from a DataFrame slice of Parquet rows."""
    if text_field not in batch_df.columns:
        raise KeyError(f"Text field '{text_field}' missing from batch.")

    available_metadata_fields = [field for field in (metadata_fields or []) if field in batch_df.columns]

    documents: list[Document] = []
    for local_idx in range(len(batch_df)):
        row = batch_df.iloc[local_idx]
        text = row.get(text_field)
        if pd.isna(text):
            continue

        metadata = {"source": source, "row_index": row_offset + local_idx}
        for field in available_metadata_fields:
            value = row.get(field)
            normalized = _normalize_metadata_value(value)
            if normalized is not None:
                metadata[field] = normalized

        documents.append(Document(page_content=text, metadata=metadata))
    return documents

def documents_from_html_dataframe(
    batch_df: pd.DataFrame,
    html_field: str,
    metadata_fields: Sequence[str] | None,
    source: str,
    row_offset: int,
) -> list[Document]:
    """Extract Document objects from a DataFrame slice of Parquet rows."""
    if html_field not in batch_df.columns:
        raise KeyError(f"HTML field '{html_field}' missing from batch.")

    available_metadata_fields = [field for field in (metadata_fields or []) if field in batch_df.columns]

    documents: list[Document] = []
    for local_idx in range(len(batch_df)):
        row = batch_df.iloc[local_idx]
        html = row.get(html_field)
        if pd.isna(html):
            continue

        metadata = {"source": source, "row_index": row_offset + local_idx}
        for field in available_metadata_fields:
            value = row.get(field)
            normalized = _normalize_metadata_value(value)
            if normalized is not None:
                metadata[field] = normalized

        text = BeautifulSoup(str(html), "html.parser").get_text(separator="\n", strip=True)
        documents.append(Document(page_content=text, metadata=metadata))
    return documents
