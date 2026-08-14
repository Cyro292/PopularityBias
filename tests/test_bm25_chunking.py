"""Tests for BM25 and FAISS chunk-boundary parity."""

from __future__ import annotations

import pytest
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.rag.bm25_rag_service import BM25RagService


def test_bm25_uses_faiss_recursive_chunking() -> None:
    """BM25 produces the same chunks as the splitter configured by FAISS."""
    text = (
        "First paragraph has several words and a natural boundary.\n\n"
        "Second paragraph is deliberately longer and contains enough text "
        "to require recursive splitting across multiple chunks.\n"
        "Final line has more words for overlap behavior."
    )
    expected = RecursiveCharacterTextSplitter(
        chunk_size=60,
        chunk_overlap=10,
    ).split_text(text)

    service = BM25RagService(chunk_size=60, chunk_overlap=10)

    assert service._text_splitter is not None
    assert service._text_splitter.split_text(text) == expected


def test_bm25_can_disable_chunking() -> None:
    """Disabling BM25 chunking retains whole source articles."""
    service = BM25RagService(chunk=False)

    assert service._text_splitter is None


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [(0, 0), (100, -1), (100, 100), (100, 101)],
)
def test_bm25_rejects_invalid_chunk_configuration(
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """Invalid splitter settings fail before a long-running index build."""
    with pytest.raises(ValueError):
        BM25RagService(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
