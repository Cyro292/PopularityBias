"""Standardised popularity-decile utilities.

Provides a single, canonical way to:
1. Compute corpus decile boundaries (unweighted *and* chunk-weighted).
2. Assign decile labels to any popularity values / DataFrames.
3. Persist & reload boundaries via ``metadata.json``.

Every notebook (retrieval, evaluation, QA-prep) must use these helpers so
that boundary computation and bin assignment are **identical** everywhere.

Two decile flavours are always computed side-by-side:
    * **unweighted**      – equal *document* distribution   (1 doc = 1 count)
    * **chunk-weighted**   – equal *chunk* distribution      (1 chunk = 1 count)

Column-name constants
---------------------
``COL_DECILE_UNWEIGHTED``       = ``"pop_decile_unweighted"``
``COL_DECILE_CHUNK_WEIGHTED``   = ``"pop_decile_chunk_weighted"``
``COL_POPULARITY``              = ``"popularity_avg"``
"""

from __future__ import annotations

import gc
import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public column-name constants – import these instead of hard-coding strings
# ---------------------------------------------------------------------------
NUM_DECILES: int = 10
COL_POPULARITY: str = "popularity_avg"
COL_DECILE_UNWEIGHTED: str = "pop_decile_unweighted"
COL_DECILE_CHUNK_WEIGHTED: str = "pop_decile_chunk_weighted"

# Keys used inside metadata.json
_META_KEY_UNWEIGHTED: str = "decile_boundaries_unweighted"
_META_KEY_CHUNK_WEIGHTED: str = "decile_boundaries_chunk_weighted"
_META_KEY_CHUNK_CFG: str = "decile_boundaries_chunk_weighted_config"
_META_KEY_CORPUS_STATS: str = "corpus_stats"


# =====================================================================
# 1.  Boundary computation  (streaming, memory-efficient)
# =====================================================================

def compute_corpus_boundaries(
    corpus_path: Path | str,
    batch_size: int = 100_000,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    n_deciles: int = NUM_DECILES,
) -> tuple[np.ndarray, np.ndarray, dict, dict[int, float]]:
    """Stream the corpus parquet and return both boundary arrays.

    Returns
    -------
    boundaries_unweighted : np.ndarray, shape (n_deciles + 1,)
        Percentile edges where each *document* counts once.
    boundaries_chunk_weighted : np.ndarray, shape (n_deciles + 1,)
        Percentile edges where each *chunk* counts once (popular long
        articles dominate more because they produce more chunks).
    stats : dict
        ``total_documents``, ``unique_documents_with_popularity``,
        ``total_chunks_after_splitting``.
    id_to_pop : dict[int, float]
        Mapping ``wikipedia_id → popularity_avg`` for every unique doc.
    """
    import pyarrow.parquet as pq
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from tqdm.auto import tqdm

    corpus_path = Path(corpus_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    pf = pq.ParquetFile(corpus_path)
    total_rows = pf.metadata.num_rows

    logger.info("Computing decile boundaries from %s docs …", f"{total_rows:,}")

    id_to_pop: dict[int, float] = {}
    pops_uw: list[float] = []       # unweighted
    pops_cw: list[float] = []       # chunk-weighted
    doc_lengths: list[int] = []     # text length per unique doc (same order as pops_uw)
    total_chunks = 0

    for batch in tqdm(
        pf.iter_batches(batch_size=batch_size,
                        columns=["wikipedia_id", COL_POPULARITY, "text"]),
        total=(total_rows + batch_size - 1) // batch_size,
        desc="Corpus boundaries",
    ):
        bdf = batch.to_pandas()
        bdf["wikipedia_id"] = bdf["wikipedia_id"].astype(int)
        valid = bdf.dropna(subset=[COL_POPULARITY])
        valid = valid[valid[COL_POPULARITY] >= 0]

        for _, row in valid.iterrows():
            wid = int(row["wikipedia_id"])
            if wid in id_to_pop:
                continue
            pop = float(row[COL_POPULARITY])

            text = str(row.get("text") or "")
            chunks = max(1, len(splitter.split_text(text))) if text else 1

            id_to_pop[wid] = pop
            pops_uw.append(pop)
            pops_cw.extend([pop] * chunks)
            doc_lengths.append(len(text))
            total_chunks += chunks

        del bdf, valid
        gc.collect()

    unique_docs = len(id_to_pop)
    logger.info("Unique docs with popularity: %s  |  total chunks: %s",
                f"{unique_docs:,}", f"{total_chunks:,}")

    pops_uw_arr = np.asarray(pops_uw, dtype=np.float32)
    pops_cw_arr = np.asarray(pops_cw, dtype=np.float32)

    pctiles = np.linspace(0, 100, n_deciles + 1)
    boundaries_uw = np.percentile(pops_uw_arr, pctiles)
    boundaries_cw = np.percentile(pops_cw_arr, pctiles)

    # -----------------------------------------------------------------
    # Pre-compute per-decile distributions (stored in metadata so the
    # evaluation notebook never has to re-scan the corpus)
    # -----------------------------------------------------------------
    _internal_uw = boundaries_uw[1:-1]
    _internal_cw = boundaries_cw[1:-1]

    # Documents per decile for each boundary set
    docs_dec_uw = np.searchsorted(_internal_uw, pops_uw_arr, side="right")
    docs_dec_cw = np.searchsorted(_internal_cw, pops_uw_arr, side="right")
    docs_per_decile_uw = np.bincount(docs_dec_uw, minlength=n_deciles)[:n_deciles]
    docs_per_decile_cw = np.bincount(docs_dec_cw, minlength=n_deciles)[:n_deciles]

    # Chunks per decile for each boundary set
    chunks_dec_uw = np.searchsorted(_internal_uw, pops_cw_arr, side="right")
    chunks_dec_cw = np.searchsorted(_internal_cw, pops_cw_arr, side="right")
    chunks_per_decile_uw = np.bincount(chunks_dec_uw, minlength=n_deciles)[:n_deciles]
    chunks_per_decile_cw = np.bincount(chunks_dec_cw, minlength=n_deciles)[:n_deciles]

    # Average document length (chars) per decile
    doc_lengths_arr = np.asarray(doc_lengths, dtype=np.float64)
    avg_doc_len_uw = np.zeros(n_deciles, dtype=np.float64)
    avg_doc_len_cw = np.zeros(n_deciles, dtype=np.float64)
    for d in range(n_deciles):
        m_uw = docs_dec_uw == d
        m_cw = docs_dec_cw == d
        if m_uw.sum() > 0:
            avg_doc_len_uw[d] = doc_lengths_arr[m_uw].mean()
        if m_cw.sum() > 0:
            avg_doc_len_cw[d] = doc_lengths_arr[m_cw].mean()

    del pops_uw, pops_cw, pops_uw_arr, pops_cw_arr, doc_lengths, doc_lengths_arr
    gc.collect()

    stats = {
        "total_documents": total_rows,
        "unique_documents_with_popularity": unique_docs,
        "total_chunks_after_splitting": total_chunks,
        # Per-decile distributions (4 arrays)
        "docs_per_decile_uw": docs_per_decile_uw.tolist(),
        "docs_per_decile_cw": docs_per_decile_cw.tolist(),
        "chunks_per_decile_uw": chunks_per_decile_uw.tolist(),
        "chunks_per_decile_cw": chunks_per_decile_cw.tolist(),
        # Average document length (chars) per decile
        "avg_doc_length_per_decile_uw": [round(v, 1) for v in avg_doc_len_uw.tolist()],
        "avg_doc_length_per_decile_cw": [round(v, 1) for v in avg_doc_len_cw.tolist()],
    }
    return boundaries_uw, boundaries_cw, stats, id_to_pop


# =====================================================================
# 2.  Canonical decile assignment
# =====================================================================

def assign_decile(
    popularity: float | np.ndarray | pd.Series,
    boundaries: np.ndarray,
) -> int | np.ndarray | pd.Series:
    """Assign decile label(s) **0 … N-1** using ``np.searchsorted``.

    Boundary array must have ``N + 1`` edges (e.g. 11 for 10 deciles).
    Internal edges are ``boundaries[1:-1]``.  ``side='right'`` ensures
    that a value exactly on the *k*-th internal edge falls into bin *k*
    (i.e. the upper bin wins on ties).  Values below the global minimum
    are placed in bin 0; above the global maximum → bin N-1.

    This function is the **single source of truth** for bin assignment.
    """
    internal = boundaries[1:-1]
    result = np.searchsorted(internal, popularity, side="right")
    # np.searchsorted returns int64 ndarray for array input, Python int for scalar
    if isinstance(popularity, pd.Series):
        return pd.array(result, dtype="Int64")
    return result


def assign_both_deciles(
    df: pd.DataFrame,
    boundaries_uw: np.ndarray,
    boundaries_cw: np.ndarray,
    popularity_col: str = COL_POPULARITY,
    drop_missing: bool = True,
) -> pd.DataFrame:
    """Add **both** decile columns to *df* in-place and return it.

    Columns added:
        ``pop_decile_unweighted``      (int, 0–9)
        ``pop_decile_chunk_weighted``   (int, 0–9)

    Parameters
    ----------
    df : DataFrame
        Must contain *popularity_col*.
    boundaries_uw, boundaries_cw : np.ndarray
        Boundary arrays with shape ``(n_deciles + 1,)``.
    drop_missing : bool
        If True, drop rows where popularity is NaN.
    """
    df = df.copy()
    pop = df[popularity_col]

    df[COL_DECILE_UNWEIGHTED] = assign_decile(pop, boundaries_uw)
    df[COL_DECILE_CHUNK_WEIGHTED] = assign_decile(pop, boundaries_cw)

    if drop_missing:
        mask = df[COL_DECILE_UNWEIGHTED].notna() & df[COL_DECILE_CHUNK_WEIGHTED].notna()
        df = df[mask].copy()

    df[COL_DECILE_UNWEIGHTED] = df[COL_DECILE_UNWEIGHTED].astype(int)
    df[COL_DECILE_CHUNK_WEIGHTED] = df[COL_DECILE_CHUNK_WEIGHTED].astype(int)
    return df


# =====================================================================
# 3.  Metadata persistence helpers
# =====================================================================

def boundaries_to_metadata(
    boundaries_uw: np.ndarray,
    boundaries_cw: np.ndarray,
    stats: dict,
    chunk_size: int,
    chunk_overlap: int,
) -> dict:
    """Return a dict fragment ready to be merged into ``metadata.json``.

    >>> meta = {**existing_meta, **boundaries_to_metadata(…)}
    """
    return {
        _META_KEY_UNWEIGHTED: boundaries_uw.tolist(),
        _META_KEY_CHUNK_WEIGHTED: boundaries_cw.tolist(),
        _META_KEY_CHUNK_CFG: {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        },
        _META_KEY_CORPUS_STATS: stats,
    }


def load_boundaries_from_metadata(
    metadata_path: Path | str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load both boundary arrays and stats from a ``metadata.json`` file.

    Returns
    -------
    boundaries_uw : np.ndarray
    boundaries_cw : np.ndarray
    stats : dict   (may be empty if not stored)

    Raises
    ------
    FileNotFoundError
        If *metadata_path* does not exist.
    KeyError
        If the expected keys are missing.
    """
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"metadata.json not found at {metadata_path}. "
            "Run rag_retrieval.ipynb first."
        )

    with open(metadata_path) as f:
        meta = json.load(f)

    if _META_KEY_UNWEIGHTED not in meta:
        raise KeyError(
            f"'{_META_KEY_UNWEIGHTED}' missing from {metadata_path}. "
            "Re-run the retrieval notebook to regenerate metadata."
        )
    if _META_KEY_CHUNK_WEIGHTED not in meta:
        raise KeyError(
            f"'{_META_KEY_CHUNK_WEIGHTED}' missing from {metadata_path}. "
            "Re-run the retrieval notebook to regenerate metadata."
        )

    return (
        np.array(meta[_META_KEY_UNWEIGHTED]),
        np.array(meta[_META_KEY_CHUNK_WEIGHTED]),
        meta.get(_META_KEY_CORPUS_STATS, {}),
    )


# =====================================================================
# 4.  Convenience / display helpers
# =====================================================================

def print_boundaries(
    boundaries_uw: np.ndarray,
    boundaries_cw: np.ndarray,
    n_deciles: int = NUM_DECILES,
) -> None:
    """Pretty-print both boundary arrays."""
    for label, b in [("Unweighted (equal doc)", boundaries_uw),
                     ("Chunk-weighted (equal chunk)", boundaries_cw)]:
        print(f"\n  {label}:")
        for i in range(n_deciles):
            lo, hi = b[i], b[i + 1]
            print(f"    Decile {i} (shown as {i+1}): [{lo:.4f}, {hi:.4f})")


def decile_col_for(mode: str) -> str:
    """Return the column name for *mode* ∈ {``'unweighted'``, ``'chunk_weighted'``}."""
    if mode == "unweighted":
        return COL_DECILE_UNWEIGHTED
    if mode in ("chunk_weighted", "weighted"):
        return COL_DECILE_CHUNK_WEIGHTED
    raise ValueError(f"Unknown decile mode {mode!r}; use 'unweighted' or 'chunk_weighted'.")


def boundaries_for(
    mode: str,
    boundaries_uw: np.ndarray,
    boundaries_cw: np.ndarray,
) -> np.ndarray:
    """Return the boundary array for *mode*."""
    if mode == "unweighted":
        return boundaries_uw
    if mode in ("chunk_weighted", "weighted"):
        return boundaries_cw
    raise ValueError(f"Unknown decile mode {mode!r}; use 'unweighted' or 'chunk_weighted'.")


def load_corpus_distributions(
    stats: dict,
    mode: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return ``(docs_per_decile, chunks_per_decile)`` for *mode* from stats.

    Returns None if the distributions are not stored in *stats*
    (i.e. metadata was generated before this feature was added).
    """
    if mode == "unweighted":
        docs_key, chunks_key = "docs_per_decile_uw", "chunks_per_decile_uw"
    elif mode in ("chunk_weighted", "weighted"):
        docs_key, chunks_key = "docs_per_decile_cw", "chunks_per_decile_cw"
    else:
        raise ValueError(f"Unknown decile mode {mode!r}")

    if docs_key not in stats or chunks_key not in stats:
        return None

    return np.array(stats[docs_key]), np.array(stats[chunks_key])


def load_avg_doc_length_per_decile(
    stats: dict,
    mode: str,
) -> np.ndarray | None:
    """Return avg document length (chars) per decile from metadata stats.

    Returns None if not stored (older metadata).
    """
    if mode == "unweighted":
        key = "avg_doc_length_per_decile_uw"
    elif mode in ("chunk_weighted", "weighted"):
        key = "avg_doc_length_per_decile_cw"
    else:
        raise ValueError(f"Unknown decile mode {mode!r}")

    if key not in stats:
        return None

    return np.array(stats[key])


# =====================================================================
# 5.  On-the-fly corpus distribution computation
# =====================================================================

def compute_corpus_distributions(
    corpus_path: Path | str,
    boundaries_uw: np.ndarray,
    boundaries_cw: np.ndarray,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    batch_size: int = 50_000,
    n_deciles: int = NUM_DECILES,
    metadata_path: Path | str | None = None,
) -> dict:
    """Scan the corpus and compute per-decile doc & chunk distributions.

    Uses the actual ``RecursiveCharacterTextSplitter`` to count chunks
    (matching ``compute_corpus_boundaries`` exactly) and deduplicates
    documents by ``wikipedia_id``.  This is slower than estimation but
    guarantees that chunk-weighted distributions are perfectly flat when
    plotted against chunk-weighted boundaries.

    Parameters
    ----------
    corpus_path : Path
        Path to the corpus parquet file.
    boundaries_uw, boundaries_cw : np.ndarray
        Boundary arrays (shape ``n_deciles + 1``).
    chunk_size, chunk_overlap : int
        Passed to ``RecursiveCharacterTextSplitter`` (must match the
        values used when computing boundaries).
    metadata_path : Path, optional
        If given, the computed distributions are **saved** into the
        ``corpus_stats`` section of that ``metadata.json`` so future
        runs can load them instantly.

    Returns
    -------
    dist : dict
        Contains ``docs_per_decile_uw``, ``docs_per_decile_cw``,
        ``chunks_per_decile_uw``, ``chunks_per_decile_cw`` (lists of int).
    """
    import pyarrow.parquet as pq
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from tqdm.auto import tqdm

    corpus_path = Path(corpus_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    internal_uw = boundaries_uw[1:-1]
    internal_cw = boundaries_cw[1:-1]

    pf = pq.ParquetFile(corpus_path)
    total_rows = pf.metadata.num_rows
    n_batches = (total_rows + batch_size - 1) // batch_size

    docs_uw = np.zeros(n_deciles, dtype=np.int64)
    docs_cw = np.zeros(n_deciles, dtype=np.int64)
    chunks_uw = np.zeros(n_deciles, dtype=np.int64)
    chunks_cw = np.zeros(n_deciles, dtype=np.int64)

    seen_ids: set[int] = set()

    for batch in tqdm(
        pf.iter_batches(
            batch_size=batch_size,
            columns=["wikipedia_id", COL_POPULARITY, "text"],
        ),
        total=n_batches,
        desc="Corpus distributions",
        unit="batch",
    ):
        df = batch.to_pandas()
        df["wikipedia_id"] = df["wikipedia_id"].astype(int)
        valid = df.dropna(subset=[COL_POPULARITY])
        valid = valid[valid[COL_POPULARITY] >= 0]

        # Deduplicate by wikipedia_id (matches compute_corpus_boundaries)
        valid = valid[~valid["wikipedia_id"].isin(seen_ids)]
        if len(valid) == 0:
            continue
        seen_ids.update(valid["wikipedia_id"].tolist())

        pops = valid[COL_POPULARITY].values
        texts = valid["text"].fillna("").tolist()

        # Actual chunk counting using the same splitter as
        # compute_corpus_boundaries — guarantees exact match
        n_chunks = np.array(
            [max(1, len(splitter.split_text(t))) if t else 1
             for t in texts],
            dtype=np.int64,
        )

        dec_uw = np.searchsorted(internal_uw, pops, side="right")
        dec_cw = np.searchsorted(internal_cw, pops, side="right")

        for d in range(n_deciles):
            m_uw = dec_uw == d
            m_cw = dec_cw == d
            docs_uw[d] += m_uw.sum()
            docs_cw[d] += m_cw.sum()
            chunks_uw[d] += n_chunks[m_uw].sum()
            chunks_cw[d] += n_chunks[m_cw].sum()

        del df, valid, texts
        gc.collect()

    # Average document length (chars) per decile
    avg_doc_len_uw = np.zeros(n_deciles, dtype=np.float64)
    avg_doc_len_cw = np.zeros(n_deciles, dtype=np.float64)
    sum_len_uw = np.zeros(n_deciles, dtype=np.float64)
    sum_len_cw = np.zeros(n_deciles, dtype=np.float64)
    # We already accumulated docs_uw / docs_cw counts; now accumulate
    # text-length sums.  We need a second pass over the corpus for this.
    seen_ids2: set[int] = set()
    pf2 = pq.ParquetFile(corpus_path)
    for batch in pf2.iter_batches(
        batch_size=batch_size,
        columns=["wikipedia_id", COL_POPULARITY, "text"],
    ):
        df2 = batch.to_pandas()
        df2["wikipedia_id"] = df2["wikipedia_id"].astype(int)
        v2 = df2.dropna(subset=[COL_POPULARITY])
        v2 = v2[v2[COL_POPULARITY] >= 0]
        v2 = v2[~v2["wikipedia_id"].isin(seen_ids2)]
        if len(v2) == 0:
            continue
        seen_ids2.update(v2["wikipedia_id"].tolist())
        pops2 = v2[COL_POPULARITY].values
        lens2 = v2["text"].fillna("").str.len().values.astype(np.float64)
        d_uw2 = np.searchsorted(internal_uw, pops2, side="right")
        d_cw2 = np.searchsorted(internal_cw, pops2, side="right")
        for d in range(n_deciles):
            sum_len_uw[d] += lens2[d_uw2 == d].sum()
            sum_len_cw[d] += lens2[d_cw2 == d].sum()
        del df2, v2
    del seen_ids2

    for d in range(n_deciles):
        if docs_uw[d] > 0:
            avg_doc_len_uw[d] = sum_len_uw[d] / docs_uw[d]
        if docs_cw[d] > 0:
            avg_doc_len_cw[d] = sum_len_cw[d] / docs_cw[d]

    dist = {
        "docs_per_decile_uw": docs_uw.tolist(),
        "docs_per_decile_cw": docs_cw.tolist(),
        "chunks_per_decile_uw": chunks_uw.tolist(),
        "chunks_per_decile_cw": chunks_cw.tolist(),
        "avg_doc_length_per_decile_uw": [round(v, 1) for v in avg_doc_len_uw.tolist()],
        "avg_doc_length_per_decile_cw": [round(v, 1) for v in avg_doc_len_cw.tolist()],
    }

    # Persist into metadata so next run is instant
    if metadata_path is not None:
        metadata_path = Path(metadata_path)
        if metadata_path.exists():
            with open(metadata_path) as f:
                meta = json.load(f)
            meta.setdefault(_META_KEY_CORPUS_STATS, {}).update(dist)
            with open(metadata_path, "w") as f:
                json.dump(meta, f, indent=2)
            logger.info("Saved corpus distributions to %s", metadata_path)

    return dist
