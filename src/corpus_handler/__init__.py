"""Corpus handler implementations."""

from __future__ import annotations

from src.corpus_handler.base import CorpusHandler
from src.corpus_handler.parquet_corpus_handler import ParquetCorpusHandler

__all__ = ["CorpusHandler", "ParquetCorpusHandler"]
