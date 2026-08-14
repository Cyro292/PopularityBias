"""BM25-based RAG service backed by bm25s.

Uses bm25s native corpus storage — no SQLite, no custom file formats.
At query time the score arrays are memory-mapped (low RAM); the corpus
dictionaries are loaded into RAM only for the top-k results.

Index lifecycle
---------------
Build once with :meth:`index_from_parquet`, then reload cheaply with
:meth:`load_index`.  The on-disk layout is whatever bm25s writes:
``scores/`` (vocab + score arrays) and ``corpus.jsonl``.

RAM-efficiency notes
--------------------
``index_from_parquet`` is designed to minimise peak heap usage:

* **Pass 1** streams the parquet row-by-row and writes chunk metadata to a
  temporary JSONL sidecar on disk — ``corpus_dicts`` is *never* held in RAM.
* **Pass 2** re-reads that sidecar as a *generator* fed directly to the
  streaming indexer, so only one chunk's text exists in Python at a time.
* After ``retriever.save()`` the index is immediately reloaded with
  ``mmap=True`` so the score arrays live on disk, not in RAM.
* Intermediate lists (token arrays, etc.) are deleted and ``gc.collect()``
  is called at every phase boundary.
"""

from __future__ import annotations

import array
import gc
import json
import logging
import math
import re
import struct
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Generator, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from langchain.schema import Document
from tqdm import tqdm

from .base import IndexResult, RagService

import Stemmer

logger = logging.getLogger(__name__)

_stemmer = Stemmer.Stemmer("english")


# ═══════════════════════════════════════════════════════════════════════════════
# Streaming indexer — private helpers (O(vocab) peak RAM)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Produces the exact same on-disk format as ``bm25s.BM25.save()``, so the
# result can be loaded with ``bm25s.BM25.load(..., mmap=True)`` as normal.

_TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")
_DTYPE = "float32"
_INT_DTYPE = "int32"
_EMPTY_TOKEN = ""

# ── Stopwords ─────────────────────────────────────────────────────────────────

def _load_stopwords() -> frozenset[str]:
    try:
        from bm25s.stopwords import STOPWORDS_EN
        return frozenset(STOPWORDS_EN)
    except Exception:
        return frozenset()


# ── Tokeniser (mirrors bm25s.tokenize logic) ──────────────────────────────────

def _tokenise_doc(
    text: str,
    stopwords: frozenset[str],
    stemmer: Callable | None,
) -> list[str]:
    tokens = _TOKEN_PATTERN.findall(text.lower())
    tokens = [t for t in tokens if t not in stopwords]
    if stemmer is not None:
        tokens = stemmer.stemWords(tokens)
    return tokens


# ── IDF scorers matching bm25s methods ────────────────────────────────────────

def _idf_lucene(df: int, N: int) -> float:
    return math.log(1 + (N - df + 0.5) / (df + 0.5))


def _idf_robertson(df: int, N: int) -> float:
    return math.log((N - df + 0.5) / (df + 0.5))


def _idf_atire(df: int, N: int) -> float:
    return math.log(N / df)


def _idf_bm25l(df: int, N: int) -> float:
    return math.log((N + 1) / (df + 0.5))


def _idf_bm25plus(df: int, N: int) -> float:
    return math.log((N + 1) / df)


_IDF_FN: dict[str, Callable] = {
    "lucene":     _idf_lucene,
    "robertson":  _idf_robertson,
    "atire":      _idf_atire,
    "bm25l":      _idf_bm25l,
    "bm25+":      _idf_bm25plus,
}

# ── TF-component scorers ───────────────────────────────────────────────────────

def _tfc_robertson(tf: float, l_d: float, l_avg: float, k1: float, b: float, delta: float = 0.0) -> float:
    return tf / (k1 * ((1 - b) + b * l_d / l_avg) + tf)


def _tfc_atire(tf: float, l_d: float, l_avg: float, k1: float, b: float, delta: float = 0.0) -> float:
    return (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * l_d / l_avg))


def _tfc_bm25l(tf: float, l_d: float, l_avg: float, k1: float, b: float, delta: float = 0.5) -> float:
    c = tf / (1 - b + b * l_d / l_avg)
    return ((k1 + 1) * (c + delta)) / (k1 + c + delta)


def _tfc_bm25plus(tf: float, l_d: float, l_avg: float, k1: float, b: float, delta: float = 1.0) -> float:
    return _tfc_robertson(tf, l_d, l_avg, k1, b) + delta


_TFC_FN: dict[str, Callable] = {
    "lucene":     _tfc_robertson,
    "robertson":  _tfc_robertson,
    "atire":      _tfc_atire,
    "bm25l":      _tfc_bm25l,
    "bm25+":      _tfc_bm25plus,
}

_DELTA: dict[str, float] = {
    "lucene":    0.0,
    "robertson": 0.0,
    "atire":     0.0,
    "bm25l":     0.5,
    "bm25+":     1.0,
}

_METHODS_REQUIRING_NONOCCURRENCE: frozenset[str] = frozenset({"bm25l", "bm25+"})


# ── Token sidecar helpers ──────────────────────────────────────────────────────
# Each doc is stored as:  [uint32 length][int32 × length]
# This lets us stream it back without knowing lengths upfront.

def _write_doc_ids(fh: Any, ids: list[int]) -> None:
    n = len(ids)
    fh.write(struct.pack("<I", n))
    if n:
        fh.write(array.array("i", ids).tobytes())


def _iter_doc_ids(path: Path) -> Generator[list[int], None, None]:
    with open(path, "rb") as fh:
        while True:
            header = fh.read(4)
            if not header:
                break
            (n,) = struct.unpack("<I", header)
            if n == 0:
                yield []
            else:
                buf = fh.read(n * 4)
                yield array.array("i", buf).tolist()


# ── bm25s version helper ───────────────────────────────────────────────────────

def _bm25s_version() -> str:
    try:
        import bm25s
        return getattr(bm25s, "__version__", "unknown")
    except Exception:
        return "unknown"


# ── Main streaming indexer ─────────────────────────────────────────────────────

def _build_streaming_index(
    jsonl_path: Path,
    save_dir: Path,
    *,
    n_docs: int,
    stemmer: Callable | None = None,
    k1: float = 1.5,
    b: float = 0.75,
    method: str = "lucene",
) -> None:
    """Build a bm25s-compatible index from a JSONL sidecar without loading
    the full corpus into RAM.

    Algorithm
    ---------
    Pass 1 (tokenise + count)
        Stream every document, tokenise it, accumulate vocab, doc_freq,
        doc_lengths, and a binary token-id sidecar on disk.
        Peak RAM: vocab dict + doc_freq dict + doc_lengths list.

    Pass 2 (score + write CSC)
        Re-stream the token-id sidecar, compute BM25 scores, and accumulate
        the three CSC arrays directly into pre-allocated memmaps on disk.
        Peak RAM: one doc's token list + OS memmap windows.

    Args:
        jsonl_path: Path to the JSONL sidecar (one ``{"text": ..., ...}`` per line).
        save_dir: Directory to write the index files into.
        n_docs: Total number of documents (chunks) — used for progress bars.
        stemmer: Optional PyStemmer-compatible stemmer.
        k1: BM25 term-saturation parameter.
        b: BM25 length-normalisation parameter.
        method: BM25 scoring variant — ``"lucene"``, ``"bm25+"``, ``"atire"``,
            ``"robertson"``, or ``"bm25l"``.

    Raises:
        ValueError: If ``method`` is not recognised.
    """
    if method not in _IDF_FN:
        raise ValueError(f"Unknown BM25 method '{method}'. Choose from: {list(_IDF_FN)}")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    stopwords = _load_stopwords()
    idf_fn = _IDF_FN[method]
    tfc_fn = _TFC_FN[method]
    delta = _DELTA[method]

    # ── Pass 1: tokenise, build vocab + doc_freq + doc_lengths ───────────────
    logger.info("Streaming index pass 1/2: tokenising & counting …")

    vocab: dict[str, int] = {}      # token → int id
    doc_freq: dict[int, int] = {}   # token_id → doc count
    doc_lengths: list[int] = []
    total_tokens: int = 0

    with tempfile.NamedTemporaryFile(suffix=".tokids", delete=False) as tok_fh:
        tok_path = Path(tok_fh.name)

    try:
        with open(tok_path, "wb") as tok_fh, \
             open(jsonl_path, encoding="utf-8") as jsonl_fh:

            for line in tqdm(
                jsonl_fh,
                total=n_docs,
                desc="Pass 1/2  tokenising",
                unit="doc",
                unit_scale=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}  [{elapsed}<{remaining}, {rate_fmt}]",
            ):
                text = json.loads(line).get("text", "")
                tokens = _tokenise_doc(text, stopwords, stemmer)

                ids: list[int] = []
                seen_in_doc: set[int] = set()
                for tok in tokens:
                    if tok not in vocab:
                        tid = len(vocab)
                        vocab[tok] = tid
                        doc_freq[tid] = 0
                    else:
                        tid = vocab[tok]
                    ids.append(tid)
                    seen_in_doc.add(tid)

                for tid in seen_in_doc:
                    doc_freq[tid] += 1

                doc_lengths.append(len(ids))
                total_tokens += len(ids)
                _write_doc_ids(tok_fh, ids)

        # ensure empty token exists (bm25s always adds it)
        if _EMPTY_TOKEN not in vocab:
            vocab[_EMPTY_TOKEN] = len(vocab)

        n_vocab = len(vocab)
        avg_doc_len = total_tokens / max(n_docs, 1)
        logger.info("Pass 1 done — vocab=%d tokens, avg_doc_len=%.1f", n_vocab, avg_doc_len)

        # ── IDF array ─────────────────────────────────────────────────────────
        idf_array = np.zeros(n_vocab, dtype=_DTYPE)
        for tok, tid in vocab.items():
            df = doc_freq.get(tid, 0)
            if df > 0:
                idf_array[tid] = idf_fn(df, n_docs)

        # ── Non-occurrence array (bm25l / bm25+) ──────────────────────────────
        nonoccurrence_array: np.ndarray | None = None
        if method in _METHODS_REQUIRING_NONOCCURRENCE:
            nonoccurrence_array = np.zeros(n_vocab, dtype=_DTYPE)
            for tok, tid in vocab.items():
                df = doc_freq.get(tid, 0)
                if df > 0:
                    idf = idf_fn(df, n_docs)
                    tfc = tfc_fn(0, avg_doc_len, avg_doc_len, k1, b, delta)
                    nonoccurrence_array[tid] = idf * tfc

        # ── Pass 2: compute scores → CSC arrays ───────────────────────────────
        nnz = sum(doc_freq.values())
        logger.info("Streaming index pass 2/2: computing scores (nnz=%d) …", nnz)

        # Allocate CSC building arrays as memmaps so they never fully enter heap.
        _data_mm    = np.memmap(save_dir / "_tmp_data.mm",    dtype=_DTYPE,     mode="w+", shape=(nnz,))
        _doc_idx_mm = np.memmap(save_dir / "_tmp_docidx.mm",  dtype=_INT_DTYPE, mode="w+", shape=(nnz,))
        _voc_idx_mm = np.memmap(save_dir / "_tmp_vocidx.mm",  dtype=_INT_DTYPE, mode="w+", shape=(nnz,))

        ptr = 0
        doc_lengths_arr = np.array(doc_lengths, dtype=_INT_DTYPE)
        del doc_lengths

        for doc_idx, ids in enumerate(
            tqdm(
                _iter_doc_ids(tok_path),
                total=n_docs,
                desc="Pass 2/2  scoring  ",
                unit="doc",
                unit_scale=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}  [{elapsed}<{remaining}, {rate_fmt}]",
            )
        ):
            if not ids:
                continue

            l_d = int(doc_lengths_arr[doc_idx])
            counts = Counter(ids)
            voc_ind = np.array(list(counts.keys()), dtype=_INT_DTYPE)
            tf_arr  = np.array(list(counts.values()), dtype=_DTYPE)

            tfc = tfc_fn(tf_arr, l_d, avg_doc_len, k1, b, delta)
            score = idf_array[voc_ind] * tfc
            if nonoccurrence_array is not None:
                score -= nonoccurrence_array[voc_ind]

            n = len(score)
            _data_mm[ptr : ptr + n]    = score
            _doc_idx_mm[ptr : ptr + n] = doc_idx
            _voc_idx_mm[ptr : ptr + n] = voc_ind
            ptr += n

        _data_mm.flush()
        _doc_idx_mm.flush()
        _voc_idx_mm.flush()

        # ── Build CSC via counting sort (O(nnz), no argsort) ──────────────────
        # np.argsort on 1.5 B entries stalls for minutes and needs ~6 GB RAM.
        # Instead we use a two-pass counting sort keyed on voc_idx, which is
        # bounded by n_vocab (~2-5 M) — far smaller than nnz.
        #
        # Pass A: count occurrences per vocab column → build indptr.
        # Pass B: stream COO once more, placing each entry at its destination
        #         offset using a cursor array derived from indptr.
        logger.info("Building CSC indptr (counting sort, nnz=%d, vocab=%d) …", ptr, n_vocab)

        CHUNK = 1 << 22  # 4 M entries ≈ 16-32 MB per read chunk

        # Pass A — count entries per vocab column
        col_counts = np.zeros(n_vocab, dtype=np.int64)
        for start in tqdm(
            range(0, ptr, CHUNK),
            desc="CSC pass A  counting ",
            unit="chunk",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}  [{elapsed}<{remaining}, {rate_fmt}]",
        ):
            end = min(start + CHUNK, ptr)
            np.add.at(col_counts, _voc_idx_mm[start:end], 1)

        indptr = np.zeros(n_vocab + 1, dtype=np.int64)
        np.cumsum(col_counts, out=indptr[1:])
        del col_counts

        # Pass B — scatter COO entries into sorted positions
        sorted_data = np.memmap(save_dir / "_tmp_data_sorted.mm",   dtype=_DTYPE,     mode="w+", shape=(ptr,))
        sorted_doc  = np.memmap(save_dir / "_tmp_docidx_sorted.mm", dtype=_INT_DTYPE, mode="w+", shape=(ptr,))
        sorted_voc  = np.memmap(save_dir / "_tmp_vocidx_sorted.mm", dtype=_INT_DTYPE, mode="w+", shape=(ptr,))

        cursor = indptr[:-1].copy().astype(np.int64)

        for start in tqdm(
            range(0, ptr, CHUNK),
            desc="CSC pass B  scattering",
            unit="chunk",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}  [{elapsed}<{remaining}, {rate_fmt}]",
        ):
            end = min(start + CHUNK, ptr)
            v_chunk = _voc_idx_mm[start:end].astype(np.int64)
            d_chunk = _data_mm[start:end]
            r_chunk = _doc_idx_mm[start:end]

            local_order = np.argsort(v_chunk, kind="stable")
            v_sorted = v_chunk[local_order]

            _, first_occ, cnts_in_chunk = np.unique(
                v_sorted, return_index=True, return_counts=True
            )
            within_offsets = np.arange(len(v_sorted), dtype=np.int64)
            within_offsets -= np.repeat(first_occ, cnts_in_chunk)

            positions = cursor[v_sorted] + within_offsets

            unique_v = v_sorted[first_occ]
            cursor[unique_v] += cnts_in_chunk

            sorted_data[positions] = d_chunk[local_order]
            sorted_doc[positions]  = r_chunk[local_order]
            sorted_voc[positions]  = v_sorted

        del cursor, _voc_idx_mm, _data_mm, _doc_idx_mm
        for p in ("_tmp_vocidx.mm", "_tmp_data.mm", "_tmp_docidx.mm"):
            (save_dir / p).unlink(missing_ok=True)

        sorted_data.flush()
        sorted_doc.flush()
        sorted_voc.flush()

        # ── Save in bm25s format ───────────────────────────────────────────────
        logger.info("Saving index to %s …", save_dir)
        np.save(save_dir / "data.csc.index.npy",    sorted_data[:ptr], allow_pickle=False)
        np.save(save_dir / "indices.csc.index.npy", sorted_doc[:ptr],  allow_pickle=False)
        np.save(save_dir / "indptr.csc.index.npy",  indptr,            allow_pickle=False)

        del sorted_data, sorted_doc, sorted_voc

        if nonoccurrence_array is not None:
            np.save(save_dir / "nonoccurrence_array.index.npy", nonoccurrence_array, allow_pickle=False)

        with open(save_dir / "vocab.index.json", "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False)

        params = {
            "k1":        k1,
            "b":         b,
            "delta":     delta,
            "method":    method,
            "idf_method": method,
            "dtype":     _DTYPE,
            "int_dtype": _INT_DTYPE,
            "num_docs":  n_docs,
            "version":   _bm25s_version(),
            "backend":   "numpy",
        }
        with open(save_dir / "params.index.json", "w") as f:
            json.dump(params, f, indent=4)

    finally:
        for p in [
            tok_path,
            save_dir / "_tmp_data.mm",
            save_dir / "_tmp_docidx.mm",
            save_dir / "_tmp_vocidx.mm",
            save_dir / "_tmp_data_sorted.mm",
            save_dir / "_tmp_docidx_sorted.mm",
            save_dir / "_tmp_vocidx_sorted.mm",
        ]:
            try:
                Path(p).unlink(missing_ok=True)
            except OSError:
                pass


# ═══════════════════════════════════════════════════════════════════════════════
# BM25RagService
# ═══════════════════════════════════════════════════════════════════════════════

class BM25RagService(RagService):
    """RAG service using local BM25 retrieval via bm25s.

    Args:
        chunk: Split articles into chunks before indexing.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Character overlap between consecutive chunks.
        k1: BM25 term-saturation parameter.
        b: BM25 length-normalisation parameter.
        method: bm25s scoring method — ``"lucene"`` (default), ``"bm25+"``,
            ``"atire"``, ``"robertson"``, or ``"bm25l"``.
    """

    def __init__(
        self,
        *,
        chunk: bool = True,
        chunk_size: int = 1_000,
        chunk_overlap: int = 100,
        k1: float = 1.5,
        b: float = 0.75,
        method: str = "lucene",
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")

        self.chunk = chunk
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k1 = k1
        self.b = b
        self.method = method
        self._retriever: Any = None
        self._index_dir: Path | None = None

        if chunk:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            self._text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        else:
            self._text_splitter = None

    # ── Helpers ───────────────────────────────────────────────────────────────

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

    # ── Indexing ──────────────────────────────────────────────────────────────

    def index_from_dataframe(self, *args: Any, **kwargs: Any) -> IndexResult:
        """Not supported — loading a full DataFrame into RAM defeats the purpose.

        Use :meth:`index_from_parquet` instead, which streams the corpus
        row-group by row-group without ever materialising it in memory.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "index_from_dataframe is not supported for BM25RagService because loading "
            "the full corpus into a DataFrame exhausts memory on large collections. "
            "Save your data as a Parquet file and call index_from_parquet() instead."
        )

    def index_from_parquet(
        self,
        parquet_path: Path,
        *,
        text_field: str = "text",
        metadata_fields: Sequence[str] | None = None,
        collection_name: str = "bm25",
        output_dir: Path,
        **kwargs: Any,
    ) -> IndexResult:
        """Build a BM25 index by streaming a Parquet file with minimal RAM.

        Uses a two-pass strategy to avoid holding the full corpus in RAM:

        * **Pass 1** streams the parquet file row-group by row-group, writing
          each chunk as a JSON line directly to ``output_dir/corpus.jsonl``.
          No corpus list is accumulated in Python heap.
        * **Pass 2** re-reads that file as a generator and feeds text tokens
          directly to :func:`_build_streaming_index`, so only one chunk's text
          occupies RAM at a time.
        * After the index is built it is loaded with ``mmap=True`` so score
          arrays are memory-mapped rather than copied into the heap.

        Args:
            parquet_path: Path to the corpus Parquet file.
            text_field: Column containing document text.
            metadata_fields: Columns stored as document metadata.
            output_dir: Directory to persist the index.  ``corpus.jsonl`` and
                all index files are written here.
            **kwargs: Ignored.

        Returns:
            ``IndexResult`` with the bm25s retriever and chunk count.
        """
        import bm25s

        parquet_path = Path(parquet_path)
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        save_dir = Path(output_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        corpus_path = save_dir / "corpus.jsonl"

        meta_fields = list(metadata_fields or [])
        columns = [text_field] + meta_fields
        pf = pq.ParquetFile(parquet_path)
        total_rg = pf.metadata.num_row_groups

        logger.info("Streaming %s (%d row-groups) …", parquet_path.name, total_rg)

        # ── Pass 1: stream parquet → corpus.jsonl ────────────────────────────
        rows_seen = 0
        chunk_count = 0

        with open(corpus_path, "w", encoding="utf-8") as corpus_fh, tqdm(
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
                    chunks = (
                        self._text_splitter.split_text(text)
                        if self._text_splitter is not None
                        else [text]
                    )
                    for c in chunks:
                        corpus_fh.write(json.dumps({"text": c, **meta}, ensure_ascii=False))
                        corpus_fh.write("\n")
                        chunk_count += 1

                del df, texts_raw, meta_cols
                pbar.update(1)
                pbar.set_postfix(
                    articles=f"{rows_seen:,}",
                    chunks=f"{chunk_count:,}",
                    refresh=False,
                )

        logger.info("Pass 1 done — %d chunks from %d articles", chunk_count, rows_seen)
        gc.collect()

        # ── Pass 2: streaming tokenise + score + save ─────────────────────────
        _build_streaming_index(
            jsonl_path=corpus_path,
            save_dir=save_dir,
            n_docs=chunk_count,
            stemmer=_stemmer,
            k1=self.k1,
            b=self.b,
            method=self.method,
        )
        gc.collect()

        # ── Build .mmindex sidecar then load with mmap ────────────────────────
        try:
            import bm25s.utils.corpus as _bm25s_corpus
            mmidx = _bm25s_corpus.find_newline_positions(corpus_path, show_progress=False)
            _bm25s_corpus.save_mmindex(mmidx, path=corpus_path)
        except Exception as e:
            logger.warning("Could not build corpus mmindex (non-fatal): %s", e)

        logger.info("Loading index with mmap=True …")
        self._retriever = bm25s.BM25.load(str(save_dir), load_corpus=True, mmap=True)
        self._index_dir = save_dir

        return IndexResult(self._retriever, chunk_count)

    # alias
    index_from_parquet_batches = index_from_parquet  # type: ignore[assignment]

    # ── Index lifecycle ───────────────────────────────────────────────────────

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

    # ── Retrieval ─────────────────────────────────────────────────────────────

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
        batch_size: int = 124,
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
        batch_size: int = 124,
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
            results, scores = retriever.retrieve(query_tokens, k=k, show_progress=progress_bar)
            for q_idx in range(len(batch)):
                per_q_results = results[q_idx : q_idx + 1]
                per_q_scores  = scores[q_idx : q_idx + 1]
                output.append(self._results_to_docs(per_q_results, per_q_scores, top_k))

        return output

    def batch_retrieve_metadata_with_scores(
        self,
        queries: list[str],
        *,
        top_k: int = 5,
        progress_bar: bool = True,
        batch_size: int = 124,
        **kwargs: Any,
    ) -> list[list[tuple[dict[str, Any], float]]]:
        """Retrieve document metadata and BM25 scores without materializing text.

        This is suitable for analyses that need ranked document identifiers and
        scores but do not need ``Document.page_content``. Avoiding ``Document``
        creation substantially reduces memory and CPU use for large top-k runs.

        Args:
            queries: Query strings to retrieve for.
            top_k: Maximum number of ranked documents returned per query.
            progress_bar: Whether to render retrieval progress.
            batch_size: Number of queries sent to bm25s at once.
            **kwargs: Ignored compatibility arguments.

        Returns:
            One ranked list of ``(metadata, score)`` pairs per query.
        """
        import bm25s

        retriever = self._require_retriever()
        k = min(top_k, self.get_doc_count())
        output: list[list[tuple[dict[str, Any], float]]] = []

        for i in tqdm(
            range(0, len(queries), batch_size),
            desc="Retrieving metadata (bm25s)",
            disable=not progress_bar,
        ):
            batch = queries[i : i + batch_size]
            query_tokens = bm25s.tokenize(batch, stopwords="en", stemmer=_stemmer)
            results, scores = retriever.retrieve(query_tokens, k=k, show_progress=progress_bar)
            for result_row, score_row in zip(results, scores):
                output.append([
                    (
                        {key: value for key, value in entry.items() if key != "text"}
                        if isinstance(entry, dict)
                        else {},
                        float(score),
                    )
                    for entry, score in zip(result_row[:top_k], score_row[:top_k])
                ])
        return output

    # ── Inspection ────────────────────────────────────────────────────────────

    def get_doc_count(self) -> int:
        if self._retriever is None:
            return 0
        # bm25s stores num_docs inside the scores dict, not as a top-level attribute
        scores = getattr(self._retriever, "scores", None)
        if isinstance(scores, dict) and "num_docs" in scores:
            return int(scores["num_docs"])
        return int(getattr(self._retriever, "num_docs", 0))

    def get_all_documents(self, **kwargs: Any) -> list[Document]:
        raise NotImplementedError("Corpus is too large to load fully into memory.")

    def get_index_stats(self) -> dict[str, Any]:
        if self._retriever is None:
            return {"loaded": False}
        return {
            "loaded":        True,
            "n_chunks":      self.get_doc_count(),
            "k1":            self.k1,
            "b":             self.b,
            "chunk":         self.chunk,
            "chunk_size":    self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "index_dir":     str(self._index_dir) if self._index_dir else None,
        }

    def embed_prompt(self, text: str) -> str:
        return text

    def embed_passage(self, text: str) -> str:
        return text
