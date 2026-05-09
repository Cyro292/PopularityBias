"""BM25-based RAG service backed by bm25s.

Uses bm25s native corpus storage — no SQLite, no custom file formats.
At query time the score arrays are memory-mapped (low RAM); the corpus
dictionaries are loaded into RAM only for the top-k results.

Index lifecycle
---------------
Build once with :meth:`index_from_parquet`, then reload cheaply with
:meth:`load_index`.  The on-disk layout is whatever bm25s writes:
``scores/`` (vocab + score arrays) and ``corpus.jsonl``.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from langchain.schema import Document
from tqdm import tqdm

from .base import IndexResult, RagService

logger = logging.getLogger(__name__)

# ── Optional stemmer ──────────────────────────────────────────────────────────
try:
    import Stemmer as _PyStemmer
    _stemmer = _PyStemmer.Stemmer("english")
except ImportError:
    _stemmer = None
    logger.warning("PyStemmer not installed — stemming disabled")


# ── Fast chunker (no LangChain overhead) ─────────────────────────────────────

def _chunk(text: str, chunk_size: int, overlap: int) -> list[str]:
    if not text:
        return [""]
    step = max(1, chunk_size - overlap)
    return [text[i : i + chunk_size] for i in range(0, len(text), step)]


# ── BM25RagService ────────────────────────────────────────────────────────────

class BM25RagService(RagService):
    """RAG service using local BM25 retrieval via bm25s.

    Args:
        chunk: Split articles into chunks before indexing.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Character overlap between consecutive chunks.
        k1: BM25 term-saturation parameter.
        b: BM25 length-normalisation parameter.
    """

    def __init__(
        self,
        *,
        chunk: bool = True,
        chunk_size: int = 1_000,
        chunk_overlap: int = 100,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.chunk = chunk
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k1 = k1
        self.b = b
        self._retriever: Any = None
        self._index_dir: Path | None = None

    # ── Helpers ───────────────────────────────────────────────────────────

    def _require_retriever(self) -> Any:
        if self._retriever is None:
            raise ValueError("No index loaded. Call index_from_parquet() or load_index() first.")
        return self._retriever

    def _results_to_docs(
        self, results: np.ndarray, scores: np.ndarray, k: int
    ) -> list[tuple[Document, float]]:
        """Convert bm25s result arrays to (Document, score) pairs."""
        out = []
        for i in range(min(results.shape[1], k)):
            entry = results[0, i]
            score = float(scores[0, i])
            if isinstance(entry, dict):
                text = entry.get("text", "")
                meta = {key: val for key, val in entry.items() if key != "text"}
            else:
                text = str(entry)
                meta = {}
            out.append((Document(page_content=text, metadata=meta), score))
        return out

    # ── Indexing ──────────────────────────────────────────────────────────

    def index_from_dataframe(
        self,
        df: pd.DataFrame,
        text_field: str = "text",
        *,
        metadata_fields: Sequence[str] | None = None,
        collection_name: str = "bm25",
        output_dir: Path | None = None,
        **kwargs: Any,
    ) -> IndexResult:
        """Build a BM25 index from a DataFrame."""
        import bm25s

        meta_fields = list(metadata_fields or [])
        texts_raw: list[str] = df[text_field].fillna("").tolist()

        corpus_texts: list[str] = []
        corpus_dicts: list[dict] = []

        for i, text in enumerate(texts_raw):
            meta = {f: df[f].iloc[i] for f in meta_fields if f in df.columns}
            chunks = _chunk(text, self.chunk_size, self.chunk_overlap) if self.chunk else [text]
            for c in chunks:
                corpus_texts.append(c)
                corpus_dicts.append({"text": c, **meta})

        logger.info("Tokenising %d chunks …", len(corpus_texts))
        tokens = bm25s.tokenize(corpus_texts, stopwords="en", stemmer=_stemmer)
        retriever = bm25s.BM25(k1=self.k1, b=self.b)
        retriever.index(tokens)

        if output_dir is not None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            retriever.save(str(output_dir), corpus=corpus_dicts)
            self._index_dir = Path(output_dir)

        self._retriever = retriever
        self._retriever._corpus = corpus_dicts  # keep corpus attached for retrieval
        return IndexResult(retriever, len(corpus_texts))

    def index_from_parquet(
        self,
        parquet_path: Path,
        *,
        text_field: str = "text",
        metadata_fields: Sequence[str] | None = None,
        collection_name: str = "bm25",
        output_dir: Path | None = None,
        **kwargs: Any,
    ) -> IndexResult:
        """Build a BM25 index by streaming a Parquet file.

        Streams row-group by row-group so pandas RAM stays bounded.
        All chunk texts are accumulated for tokenisation (bm25s requires
        the full corpus for IDF computation).

        Args:
            parquet_path: Path to the corpus Parquet file.
            text_field: Column containing document text.
            metadata_fields: Columns stored as document metadata.
            output_dir: Directory to persist the index.
            **kwargs: Ignored.

        Returns:
            ``IndexResult`` with the bm25s retriever and chunk count.
        """
        import bm25s

        parquet_path = Path(parquet_path)
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        meta_fields = list(metadata_fields or [])
        columns = [text_field] + meta_fields
        pf = pq.ParquetFile(parquet_path)
        total_rg = pf.metadata.num_row_groups

        logger.info("Streaming %s (%d row-groups) …", parquet_path.name, total_rg)

        corpus_texts: list[str] = []
        corpus_dicts: list[dict] = []
        rows_seen = 0

        with tqdm(
            total=total_rg,
            desc="Pass 1/2  reading",
            unit="rg",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} rg  [{elapsed}<{remaining}, {rate_fmt}]",
        ) as pbar:
            for batch in pf.iter_batches(columns=columns):
                df = batch.to_pandas()
                rows_seen += len(df)

                texts_raw: list[str] = df[text_field].fillna("").tolist()
                meta_cols = {f: df[f].tolist() for f in meta_fields if f in df.columns}

                for i, text in enumerate(texts_raw):
                    meta = {f: meta_cols[f][i] for f in meta_cols}
                    chunks = _chunk(text, self.chunk_size, self.chunk_overlap) if self.chunk else [text]
                    for c in chunks:
                        corpus_texts.append(c)
                        corpus_dicts.append({"text": c, **meta})

                del df
                pbar.update(1)
                pbar.set_postfix(
                    articles=f"{rows_seen:,}",
                    chunks=f"{len(corpus_texts):,}",
                    refresh=False,
                )

        logger.info("Pass 1 done — %d chunks from %d articles", len(corpus_texts), rows_seen)

        logger.info("Pass 2/2  tokenising %d chunks …", len(corpus_texts))
        tokens = bm25s.tokenize(corpus_texts, stopwords="en", stemmer=_stemmer)
        del corpus_texts
        gc.collect()

        logger.info("Building bm25s index …")
        retriever = bm25s.BM25(k1=self.k1, b=self.b)
        retriever.index(tokens)
        del tokens
        gc.collect()

        if output_dir is not None:
            save_dir = Path(output_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Saving index to %s …", save_dir)
            retriever.save(str(save_dir), corpus=corpus_dicts)
            self._index_dir = save_dir

        self._retriever = retriever
        self._retriever._corpus = corpus_dicts
        return IndexResult(retriever, len(corpus_dicts))

    # alias
    index_from_parquet_batches = index_from_parquet  # type: ignore[assignment]

    # ── Index lifecycle ───────────────────────────────────────────────────

    def load_index(self, path_or_name: str | Path, **kwargs: Any) -> Any:
        """Load a persisted index from disk.

        Args:
            path_or_name: Directory written by :meth:`index_from_parquet`.

        Returns:
            The loaded bm25s retriever.
        """
        import bm25s

        index_dir = Path(path_or_name)
        if not index_dir.exists():
            raise FileNotFoundError(f"Index directory not found: {index_dir}")

        logger.info("Loading BM25 index from %s …", index_dir)
        self._retriever = bm25s.BM25.load(str(index_dir), load_corpus=True, mmap=True)
        self._index_dir = index_dir
        logger.info("BM25 index loaded")
        return self._retriever

    def save_index(self, path: str | Path, **kwargs: Any) -> None:
        """No-op — the index is always saved during build."""
        logger.warning("save_index() is a no-op; index is saved during index_from_parquet().")

    def delete_index(self, *, delete_files: bool = False, **kwargs: Any) -> None:
        if delete_files and self._index_dir and self._index_dir.exists():
            import shutil
            shutil.rmtree(self._index_dir)
            logger.info("Deleted index directory: %s", self._index_dir)
        self._retriever = None
        self._index_dir = None

    # ── Retrieval ─────────────────────────────────────────────────────────

    def retrieve_documents(self, query: str, *, top_k: int = 5, **kwargs: Any) -> list[Document]:
        return [doc for doc, _ in self.retrieve_documents_with_scores(query, top_k=top_k)]

    def retrieve_documents_with_scores(
        self, query: str, *, top_k: int = 5, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        import bm25s

        retriever = self._require_retriever()
        query_tokens = bm25s.tokenize([query], stopwords="en", stemmer=_stemmer)
        results, scores = retriever.retrieve(query_tokens, k=min(top_k, self.get_doc_count()))
        return self._results_to_docs(results, scores, top_k)

    def batch_retrieve(
        self,
        queries: list[str],
        *,
        top_k: int = 5,
        progress_bar: bool = True,
        batch_size: int = 64,
        **kwargs: Any,
    ) -> list[list[Document]]:
        return [
            [doc for doc, _ in per_q]
            for per_q in self.batch_retrieve_with_scores(
                queries, top_k=top_k, progress_bar=progress_bar, batch_size=batch_size
            )
        ]

    def batch_retrieve_with_scores(
        self,
        queries: list[str],
        *,
        top_k: int = 5,
        progress_bar: bool = True,
        batch_size: int = 64,
        **kwargs: Any,
    ) -> list[list[tuple[Document, float]]]:
        import bm25s

        retriever = self._require_retriever()
        k = min(top_k, self.get_doc_count())
        output: list[list[tuple[Document, float]]] = []

        for i in tqdm(
            range(0, len(queries), batch_size),
            desc="Retrieving (bm25s)",
            disable=not progress_bar,
        ):
            batch = queries[i : i + batch_size]
            query_tokens = bm25s.tokenize(batch, stopwords="en", stemmer=_stemmer)
            results, scores = retriever.retrieve(query_tokens, k=k)
            for q_idx in range(len(batch)):
                per_q_results = results[q_idx : q_idx + 1]
                per_q_scores  = scores[q_idx : q_idx + 1]
                output.append(self._results_to_docs(per_q_results, per_q_scores, top_k))

        return output

    # ── Inspection ────────────────────────────────────────────────────────

    def get_doc_count(self) -> int:
        if self._retriever is None:
            return 0
        # bm25s stores num_docs on the retriever
        return int(getattr(self._retriever, "num_docs", 0))

    def get_all_documents(self, **kwargs: Any) -> list[Document]:
        raise NotImplementedError("Corpus is too large to load fully into memory.")

    def get_index_stats(self) -> dict[str, Any]:
        if self._retriever is None:
            return {"loaded": False}
        return {
            "loaded": True,
            "n_chunks": self.get_doc_count(),
            "k1": self.k1,
            "b": self.b,
            "chunk": self.chunk,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "index_dir": str(self._index_dir) if self._index_dir else None,
        }

    def embed_prompt(self, text: str) -> str:
        return text

    def embed_passage(self, text: str) -> str:
        return text
