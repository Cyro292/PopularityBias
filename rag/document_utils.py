"""Helpers for turning Parquet batches into LangChain documents."""

from __future__ import annotations

from typing import Any, Sequence

import pandas as pd
from bs4 import BeautifulSoup
from langchain.schema import Document
from tqdm import tqdm

def documents_from_dataframe(
    df: pd.DataFrame,
    text_field: str,
    metadata_fields: Sequence[str] | None = None,
    *,
    progress_bar: bool = False,
) -> list[Document]:
    """Transform a DataFrame into LangChain documents.
    
    Args:
        df: DataFrame to convert
        text_field: Column name to use as document content
        metadata_fields: Column names to include as metadata
        
    Returns:
        List of Document objects
        
    Raises:
        ValueError: If text_field or metadata_fields don't exist in DataFrame
    """
    if text_field not in df.columns:
        raise ValueError(f"text_field '{text_field}' not found in DataFrame columns")
    
    if metadata_fields is None:
        metadata_fields = []
    
    missing_fields = set(metadata_fields) - set(df.columns)
    if missing_fields:
        raise ValueError(f"metadata_fields {missing_fields} not found in DataFrame columns")
    
    documents = []
    rows_iter = tqdm(df.iterrows(), total=len(df), desc="Building docs",
                     unit="row", disable=not progress_bar)
    for _, row in rows_iter:
        metadata = {field: row[field] for field in metadata_fields}
        doc = Document(page_content=str(row[text_field]), metadata=metadata)
        documents.append(doc)
    
    return documents
