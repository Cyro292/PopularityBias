"""TF-IDF per-decile analysis service.

Public API
----------
``load_or_compute_tfidf_stats``
    Cache-aware entry point: reads cached stats from ``metadata.json`` when
    available, otherwise runs the full sampling + vectorisation pipeline and
    writes the results back to the same file.

``plot_tfidf_stats``
    Produces a 3-panel matplotlib figure (vocabulary breadth, keyword
    intensity, term specificity) from the stats dict and saves it to disk.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Keys used inside metadata.json
_CACHE_KEY = "decile_tfidf_stats"
_NUM_DECILES = 10


def _cache_key(chunk_size: int | None, chunk_overlap: int, sample_per_decile: int) -> str:
    """Return the metadata.json key for the given chunking + sample settings."""
    if chunk_size:
        return f"{_CACHE_KEY}_chunks_{chunk_size}_{chunk_overlap}_n{sample_per_decile}"
    return f"{_CACHE_KEY}_n{sample_per_decile}"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _chunk_texts(
    texts: list[str],
    deciles: list[int],
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[list[str], list[int]]:
    """Split each document into chunks using RecursiveCharacterTextSplitter.

    Each chunk inherits the popularity decile of its parent document.

    Returns
    -------
    texts   : flat list of chunk strings
    deciles : parallel list of 0-based decile indices
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    out_texts: list[str] = []
    out_deciles: list[int] = []
    for text, d in zip(texts, deciles):
        chunks = splitter.split_text(text)
        out_texts.extend(chunks)
        out_deciles.extend([d] * len(chunks))
    return out_texts, out_deciles


def _reservoir_sample_corpus(
    corpus_path: Path | str,
    boundaries: np.ndarray,
    sample_per_decile: int,
    *,
    seed: int = 42,
) -> tuple[list[str], list[int]]:
    """Stream *corpus_path* and reservoir-sample up to *sample_per_decile*
    documents per decile.  Never holds more than one batch in RAM.

    Returns
    -------
    texts   : flat list of document strings
    deciles : parallel list of 0-based decile indices
    """
    import pyarrow.parquet as pq
    from tqdm.auto import tqdm
    from helpers.decile_utils import assign_decile

    random.seed(seed)
    reservoirs: dict[int, list[str]] = {d: [] for d in range(_NUM_DECILES)}
    counts: dict[int, int] = {d: 0 for d in range(_NUM_DECILES)}

    pf = pq.ParquetFile(str(corpus_path))
    for batch in tqdm(
        pf.iter_batches(batch_size=100_000, columns=["popularity_avg", "text"]),
        desc="Sampling corpus",
    ):
        batch_df = batch.to_pandas()
        batch_df = batch_df.dropna(subset=["popularity_avg", "text"])
        batch_df = batch_df[batch_df["popularity_avg"] >= 0].copy()

        # assign_decile returns a pandas IntegerArray — convert to plain ndarray
        decile_arr = assign_decile(batch_df["popularity_avg"], boundaries).to_numpy(
            dtype=int, na_value=-1
        )
        batch_df["_d"] = decile_arr

        for d in range(_NUM_DECILES):
            for text in batch_df.loc[batch_df["_d"] == d, "text"]:
                counts[d] += 1
                if len(reservoirs[d]) < sample_per_decile:
                    reservoirs[d].append(text)
                else:
                    j = random.randint(0, counts[d] - 1)
                    if j < sample_per_decile:
                        reservoirs[d][j] = text

        # Early exit: all deciles full and we've seen a reasonable oversample
        if all(len(reservoirs[d]) >= sample_per_decile for d in range(_NUM_DECILES)):
            sampled_so_far = sum(counts.values())
            if sampled_so_far >= sample_per_decile * 50:
                print(f"  Early exit after {sampled_so_far:,} docs seen")
                break

    texts: list[str] = []
    deciles: list[int] = []
    for d in range(_NUM_DECILES):
        for text in reservoirs[d]:
            texts.append(text)
            deciles.append(d)

    n_deciles_covered = len({d for d in deciles})
    print(f"  Sampled {len(texts):,} documents across {n_deciles_covered} deciles")
    return texts, deciles


def _compute_stats(
    texts: list[str],
    deciles: list[int],
    max_features: int,
    source_deciles: list[int] | None = None,
) -> dict[str, dict]:
    """Fit TF-IDF vectorizer and compute per-decile summary statistics.

    Parameters
    ----------
    source_deciles : list[int] | None
        If texts are chunks, pass the decile list of the *original documents*
        so that ``n_source_docs`` is counted correctly.  When ``None``,
        ``n_source_docs`` equals ``n_items``.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    print("  Fitting TF-IDF vectorizer…")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        strip_accents="unicode",
        sublinear_tf=True, 
        min_df=2,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)  # CSR (n_docs, vocab)
    idf_values = vectorizer.idf_                    # shape (vocab,)
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_):,} terms")

    all_deciles_arr = np.array(deciles)
    source_counts = {}
    if source_deciles is not None:
        src_arr = np.array(source_deciles)
        for d in range(_NUM_DECILES):
            source_counts[d] = int((src_arr == d).sum())
    stats: dict[str, dict] = {}

    for d in range(_NUM_DECILES):
        mask = all_deciles_arr == d
        n_docs_d = int(mask.sum())
        if n_docs_d == 0:
            stats[str(d)] = {}
            continue

        sub = tfidf_matrix[mask]  # (n_d, vocab) CSR

        # A. Unique terms per document (non-zero columns per row)
        unique_per_doc = np.diff(sub.indptr).astype(float)

        # B. Mean TF-IDF score of non-zero entries → keyword intensity
        nnz = sub.nnz
        mean_score = float(sub.sum() / nnz) if nnz > 0 else 0.0

        # C. IDF of terms used in this decile → term specificity
        used_idx = np.unique(sub.nonzero()[1])
        mean_idf = float(idf_values[used_idx].mean()) if len(used_idx) > 0 else 0.0
        vocab_cover = float(len(used_idx) / tfidf_matrix.shape[1])

        stats[str(d)] = {
            "n_docs_sampled": n_docs_d,
            "n_source_docs": source_counts.get(d, n_docs_d),
            "mean_unique_terms": float(np.mean(unique_per_doc)),
            "std_unique_terms": float(np.std(unique_per_doc)),
            "mean_tfidf_score": mean_score,
            "mean_idf_of_used_terms": mean_idf,
            "vocab_coverage": vocab_cover,
        }

    return stats


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_or_compute_tfidf_stats(
    metadata_path: Path | str,
    corpus_path: Path | str,
    boundaries: np.ndarray,
    *,
    sample_per_decile: int = 300,
    max_features: int = 20_000,
    chunk_size: int | None = None,
    chunk_overlap: int = 100,
    force_recompute: bool = False,
) -> dict[str, dict[str, Any]]:
    """Return per-decile TF-IDF statistics, using the cache when possible.

    Parameters
    ----------
    metadata_path : Path | str
        Path to the ``metadata.json`` file used to store/retrieve the cache.
    corpus_path : Path | str
        Path to the Parquet corpus file (must have ``popularity_avg`` and
        ``text`` columns).
    boundaries : np.ndarray
        Decile boundary array (length ``NUM_DECILES + 1``) as returned by
        ``rag.decile_utils.boundaries_for``.
    sample_per_decile : int
        Maximum number of documents to sample per decile.  When
        ``chunk_size`` is set this is the number of *documents* sampled;
        the actual TF-IDF corpus will contain more units (one per chunk).
    max_features : int
        Vocabulary cap for ``TfidfVectorizer``.
    chunk_size : int | None
        When set, each sampled document is split into chunks of this many
        characters using LangChain's ``RecursiveCharacterTextSplitter``
        before TF-IDF is computed.  Stats then reflect chunk-level
        vocabulary rather than full-document vocabulary.  ``None`` (default)
        keeps the original document-level behaviour.
    chunk_overlap : int
        Overlap in characters between consecutive chunks (only used when
        ``chunk_size`` is not ``None``).
    force_recompute : bool
        If ``True``, skip the cache and recompute from scratch.

    Returns
    -------
    dict mapping str(decile_index_0_based) → stats dict with keys:
        n_docs_sampled, mean_unique_terms, std_unique_terms,
        mean_tfidf_score, mean_idf_of_used_terms, vocab_coverage
    """
    metadata_path = Path(metadata_path)
    key = _cache_key(chunk_size, chunk_overlap, sample_per_decile)

    # ---- Try cache first ------------------------------------------------
    if not force_recompute and metadata_path.exists():
        with open(metadata_path) as f:
            meta = json.load(f)
        if key in meta:
            level = f"chunk (size={chunk_size}, overlap={chunk_overlap})" if chunk_size else "document"
            print(f"✓ Loaded TF-IDF stats from metadata.json [{level} level, cached]")
            return meta[key]

    level = f"chunk (size={chunk_size}, overlap={chunk_overlap})" if chunk_size else "document"
    print(
        f"No cache found — computing TF-IDF stats [{level} level] "
        f"(sampling {sample_per_decile} docs/decile from corpus)…"
    )

    # ---- Full computation -----------------------------------------------
    texts, deciles = _reservoir_sample_corpus(
        corpus_path,
        boundaries,
        sample_per_decile,
    )

    if chunk_size:
        print(f"  Splitting {len(texts):,} documents into chunks "
              f"(size={chunk_size}, overlap={chunk_overlap})…")
        source_deciles_for_stats = list(deciles)  # save before splitting
        texts, deciles = _chunk_texts(texts, deciles, chunk_size, chunk_overlap)
        print(f"  → {len(texts):,} chunks across all deciles")
    else:
        source_deciles_for_stats = None

    stats = _compute_stats(texts, deciles, max_features, source_deciles_for_stats)

    # ---- Write cache back to metadata.json ------------------------------
    with open(metadata_path) as f:
        meta = json.load(f)
    meta[key] = stats
    with open(metadata_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"✓ Saved TF-IDF stats [{level} level] → metadata.json  (future runs will load instantly)")

    return stats


def print_tfidf_stats(stats: dict[str, dict]) -> None:
    """Print a formatted summary table of per-decile TF-IDF statistics."""
    header = (
        f"{'Decile':>7} {'src_docs':>9} {'n_chunks':>9} {'mean_unique_terms':>18} "
        f"{'mean_tfidf':>12} {'mean_idf':>10} {'vocab_cov':>10}"
    )
    print(f"\n{header}")
    for d in range(_NUM_DECILES):
        s = stats.get(str(d), {})
        if not s:
            continue
        n_chunks = s['n_docs_sampled']
        n_src = s.get('n_source_docs', n_chunks)
        print(
            f"  {d + 1:>5}   {n_src:>9}  {n_chunks:>9}   "
            f"{s['mean_unique_terms']:>16.1f}   "
            f"{s['mean_tfidf_score']:>10.4f}   "
            f"{s['mean_idf_of_used_terms']:>8.4f}   "
            f"{s['vocab_coverage']:>8.4f}"
        )


def plot_tfidf_stats(
    stats: dict[str, dict],
    *,
    decile_mode: str = "",
    sample_per_decile: int = 300,
    max_features: int = 20_000,
    chunk_size: int | None = None,
    chunk_overlap: int = 100,
    out_path: Path | str | None = None,
) -> "plt.Figure":  # type: ignore[name-defined]  # noqa: F821
    """Create and return a 3-panel TF-IDF distribution figure.

    Parameters
    ----------
    stats : dict
        Output of :func:`load_or_compute_tfidf_stats`.
    decile_mode : str
        Label for the figure title (e.g. ``"chunk_weighted"``).
    sample_per_decile : int
        Used only in the subtitle for context.
    max_features : int
        Used only in the subtitle for context.
    chunk_size : int | None
        When set, axis labels refer to chunks rather than documents.
        Should match the value passed to :func:`load_or_compute_tfidf_stats`.
    chunk_overlap : int
        Used only in the subtitle when ``chunk_size`` is set.
    out_path : Path | str | None
        If provided, the figure is saved to this path at 150 dpi.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    unit = f"chunk ({chunk_size} chars)" if chunk_size else "document"
    unit_short = "Chunk" if chunk_size else "Doc"

    d_idx = np.arange(1, _NUM_DECILES + 1)

    mean_unique = [stats.get(str(d), {}).get("mean_unique_terms", np.nan) for d in range(_NUM_DECILES)]
    std_unique  = [stats.get(str(d), {}).get("std_unique_terms",  np.nan) for d in range(_NUM_DECILES)]
    mean_score  = [stats.get(str(d), {}).get("mean_tfidf_score",  np.nan) for d in range(_NUM_DECILES)]
    mean_idf    = [stats.get(str(d), {}).get("mean_idf_of_used_terms", np.nan) for d in range(_NUM_DECILES)]
    n_chunks    = [stats.get(str(d), {}).get("n_docs_sampled", np.nan)    for d in range(_NUM_DECILES)]
    n_src       = [stats.get(str(d), {}).get("n_source_docs",  np.nan)    for d in range(_NUM_DECILES)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel A — Vocabulary breadth
    ax = axes[0]
    ax.bar(d_idx, mean_unique, color="#3498db", alpha=0.85, edgecolor="white", linewidth=1)
    ax.errorbar(
        d_idx, mean_unique, yerr=std_unique,
        fmt="none", color="black", capsize=4, linewidth=1.5, elinewidth=1,
    )
    if chunk_size:
        for xi, nc, ns in zip(d_idx, n_chunks, n_src):
            if not np.isnan(nc):
                ax.text(xi, 0, f"{int(nc):,}\nchunks\n({int(ns):,} docs)",
                        ha="center", va="bottom", fontsize=6.5, color="#555555")
    ax.set_xlabel("Popularity Decile (1=Rare → 10=Famous)", fontsize=11, fontweight="bold")
    ax.set_ylabel(f"Mean Unique Terms / {unit_short}", fontsize=11, fontweight="bold")
    ax.set_title(f"A. Vocabulary Breadth\n(unique TF-IDF terms per {unit})", fontsize=12, fontweight="bold")
    ax.set_xticks(d_idx)
    ax.grid(axis="y", alpha=0.3)

    # Panel B — Keyword intensity
    ax = axes[1]
    ax.bar(d_idx, mean_score, color="#e67e22", alpha=0.85, edgecolor="white", linewidth=1)
    ax.set_xlabel("Popularity Decile (1=Rare → 10=Famous)", fontsize=11, fontweight="bold")
    ax.set_ylabel(f"Mean TF-IDF Score (non-zero entries)", fontsize=11, fontweight="bold")
    ax.set_title(f"B. Keyword Intensity\n(sublinear TF-IDF of matched terms per {unit})", fontsize=12, fontweight="bold")
    ax.set_xticks(d_idx)
    ax.grid(axis="y", alpha=0.3)

    # Panel C — Term specificity
    ax = axes[2]
    ax.bar(d_idx, mean_idf, color="#27ae60", alpha=0.85, edgecolor="white", linewidth=1)
    ax.set_xlabel("Popularity Decile (1=Rare → 10=Famous)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean IDF of Used Terms", fontsize=11, fontweight="bold")
    ax.set_title("C. Term Specificity\n(higher IDF = rarer/more unique terms)", fontsize=12, fontweight="bold")
    ax.set_xticks(d_idx)
    ax.grid(axis="y", alpha=0.3)

    mode_label = f" — {decile_mode} mode" if decile_mode else ""
    chunk_label = (
        f" · chunked {chunk_size} chars / {chunk_overlap} overlap"
        if chunk_size else ""
    )
    plt.suptitle(
        f"TF-IDF Distribution Per Decile{mode_label} [{unit} level]\n"
        f"(sampled ≤{sample_per_decile} docs/decile · vocab cap {max_features:,}{chunk_label})",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    if out_path is not None:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"✓ Saved → {out_path}")

    return fig
