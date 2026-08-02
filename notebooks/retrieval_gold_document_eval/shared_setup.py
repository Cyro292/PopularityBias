"""Common setup for gold-document retrieval evaluation notebooks.

Import this at the top of every eval notebook with:

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))
    from notebooks.retrieval_gold_document_eval.shared_setup import *

It exports:
  - All standard imports (os, Path, pd, np, plt, sns, tqdm, warnings)
  - Config dataclasses: BackendEntry, EvalConfig
  - Derived constants: ALL_STRATEGIES, RESULTS_DIR, IMAGES_DIR, GROUP_COL,
    TOP_K, K_VALUES_DETAILED, DECILE_MODE, EXCLUDED_DATASETS
  - Loaded data: results_by_strategy, metrics_by_strategy,
    decile_metrics_by_strategy, boundaries_uw, boundaries_cw, corpus_stats,
    corpus_docs, corpus_chunks, corpus_avg_doc_length, metadata_path
  - Helper functions: pick_group_col, strategy_colors, markers, strategy_label
  - Metric helpers: compute_metrics, get_found_rank, get_wrong_pops
"""
from __future__ import annotations

import gc
import json
import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

from dotenv import load_dotenv

# ── Repo root on sys.path ─────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import DATA_DIR
from src.metrics.decile_utils import (
    load_boundaries_from_metadata,
    load_corpus_distributions,
    load_avg_doc_length_per_decile,
    assign_decile,
    decile_col_for,
    boundaries_for,
    COL_DECILE_UNWEIGHTED,
    COL_DECILE_CHUNK_WEIGHTED,
    COL_POPULARITY,
)
from src.metrics.metrics import (
    compute_metrics,
    get_found_rank,
    get_wrong_pops,
    pick_group_col,
)

load_dotenv()

# ═════════════════════════════════════════════════════════════════════════════
# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class BackendEntry:
    """One retrieval strategy to load and compare.

    Attributes:
        label:        Human-readable name shown in plots and tables.
        results_path: Path to the ``results_{key}.parquet`` file produced by
                      the pipeline.
        key:          Unique identifier used as the dict key in
                      ``results_by_strategy`` and for cache filenames.
                      Defaults to the results parquet stem (e.g. ``"results_ivfpq"``
                      → ``"ivfpq"``).
        corpus_path:  Path to the corpus parquet (for doc-length enrichment).
                      Defaults to ``wiki_corpus.parquet`` alongside results.
        index_path:   FAISS index directory. Only needed if the notebooks need
                      to query the index directly.

    Example — two FAISS indexes::

        BackendEntry(
            label        = "FAISS high-pop",
            results_path = DATA_DIR / "wiki_full_bil/all_qa_8k/results_ivfpq.parquet",
            key          = "ivfpq_high",
            index_path   = DATA_DIR / "wiki_full_bil/faiss_high",
        ),
        BackendEntry(
            label        = "FAISS low-pop",
            results_path = DATA_DIR / "wiki_full_bil/all_qa_8k/results_ivfpq.parquet",
            key          = "ivfpq_low",
            index_path   = DATA_DIR / "wiki_full_bil/faiss_low",
        ),
    """

    label:        str
    results_path: Path
    key:          str | None  = None
    corpus_path:  Path | None = None
    index_path:   Path | None = None

    def __post_init__(self) -> None:
        self.results_path = Path(self.results_path)
        if self.key is None:
            # Strip leading "results_" from stem → e.g. "results_ivfpq" → "ivfpq"
            stem = self.results_path.stem
            self.key = stem[len("results_"):] if stem.startswith("results_") else stem
        if self.corpus_path is None:
            self.corpus_path = self.results_path.parent.parent / "wiki_corpus.parquet"
        if self.index_path is not None:
            self.index_path = Path(self.index_path)

    @property
    def is_csv(self) -> bool:
        """True when results_path points to a retrieved_docs CSV checkpoint."""
        return self.results_path.suffix.lower() == ".csv"

    @property
    def metadata_path(self) -> Path:
        return self.corpus_path.parent / "metadata.json"


@dataclass
class EvalConfig:
    """Top-level evaluation configuration.

    Attributes:
        backends:          Ordered list of retrieval strategies to compare.
        results_dir:       Where pipeline outputs and eval results live.
                           Defaults to the parent folder of the first backend's
                           results parquet.
        corpus_root:       Root folder for corpus/boundary metadata loading.
                           Defaults to the grandparent of the first results parquet.
        top_k:             Primary recall cut-off used in summary tables.
        k_values_detailed: All K values computed and plotted.
        decile_mode:       Popularity-decile weighting scheme.
        excluded_datasets: Dataset names to drop from all analyses.
        force_recompute:   Ignore caches and recompute everything from scratch.

    Example::

        _ROOT = DATA_DIR / "wiki_full_bil"
        _OUT  = _ROOT / "all_qa_8k"

        CFG = EvalConfig(
            results_dir = _OUT,
            corpus_root = _ROOT,
            backends    = [
                BackendEntry("Dense (ES)",   _OUT / "results_approximation.parquet"),
                BackendEntry("Sparse (BM25)", _OUT / "results_bm25.parquet"),
                BackendEntry("FAISS high",   _OUT / "results_ivfpq.parquet",
                             key="ivfpq_high", index_path=_ROOT / "faiss_high"),
            ],
        )
    """

    backends:    list[BackendEntry] = field(default_factory=list)
    results_dir: Path | None        = None
    corpus_root: Path | None        = None

    top_k:             int       = 1
    k_values_detailed: list[int] = field(default_factory=lambda: [1, 3, 5, 10])
    decile_mode: Literal["chunk_weighted", "unweighted"] = "chunk_weighted"
    excluded_datasets: list[str] = field(default_factory=lambda: ["hotpot_qa", "trex"])
    force_recompute:   bool      = False

    def __post_init__(self) -> None:
        if not self.backends:
            raise ValueError("EvalConfig.backends must contain at least one BackendEntry.")
        if self.results_dir is None:
            self.results_dir = self.backends[0].results_path.parent
        if self.corpus_root is None:
            self.corpus_root = self.backends[0].results_path.parent.parent
        self.results_dir = Path(self.results_dir)
        self.corpus_root = Path(self.corpus_root)

    @property
    def images_dir(self) -> Path:
        return self.results_dir / "eval_results"

    @property
    def all_keys(self) -> list[str]:
        """Unique result keys in declaration order — used as dict keys everywhere."""
        return [b.key for b in self.backends]

    @property
    def entry_by_key(self) -> dict[str, BackendEntry]:
        return {b.key: b for b in self.backends}

    @property
    def label_by_key(self) -> dict[str, str]:
        return {b.key: b.label for b in self.backends}


# ═════════════════════════════════════════════════════════════════════════════
# ── USER SETTINGS — edit here ─────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════

_ROOT = Path(DATA_DIR) / "wiki_full_bil"
_OUT  = _ROOT / "all_qa_8k"

CFG = EvalConfig(
    results_dir = _OUT,
    corpus_root = _ROOT,
    backends    = [
        # ── Baselines ─────────────────────────────────────────────────────────
        BackendEntry(
            label        = "BM25+",
            results_path = _OUT / "retrieved_docs_bm25_plus.csv",
        ),
        BackendEntry(
            label        = "FAISS high",
            results_path = _OUT / "retrieved_docs_ivfpq_high.csv",
        ),
        BackendEntry(
            label        = "FAISS hybrid",
            results_path = _OUT / "retrieved_docs_faiss_hybrid.csv",
        ),
        # ── Anti-overfitting router (frozen BERT, wd=5e-3, drop=0.6, mrr20, s42) ─
        # BackendEntry(
        #     label        = "Router frozen wd5e-3 drop60 s42",
        #     results_path = _OUT / "retrieved_docs_router60k_frozen_wd5e-3_drop60_mrr20_s42.csv",
        #     key          = "router60k_frozen_wd5e-3_drop60_mrr20_s42",
        # ),
        # BackendEntry(
        #     label        = "Router frozen wd5e-3 drop60 s42 hybrid",
        #     results_path = _OUT / "retrieved_docs_router60k_frozen_wd5e-3_drop60_mrr20_s42_hybrid.csv",
        #     key          = "router60k_frozen_wd5e-3_drop60_mrr20_s42_hybrid",
        # ),
        # BackendEntry(
        #     label        = "Router no-pop wd5e-3 drop60 s42",
        #     results_path = _OUT / "retrieved_docs_router60k_nopop_wd5e-3_drop60_mrr20_s42.csv",
        #     key          = "router60k_nopop_wd5e-3_drop60_mrr20_s42",
        # ),
        # BackendEntry(
        #     label        = "Router no-pop wd5e-3 drop60 s42 hybrid",
        #     results_path = _OUT / "retrieved_docs_router60k_nopop_wd5e-3_drop60_mrr20_s42_hybrid.csv",
        #     key          = "router60k_nopop_wd5e-3_drop60_mrr20_s42_hybrid",
        # ),
        # ── Learnable Fusion Weight Predictor ───────────────────────────────
        # BackendEntry(
        #     label        = "Fusion v1 (with pop)",
        #     results_path = _OUT / "retrieved_docs_fusion_v1.csv",
        #     key          = "fusion_v1",
        # ),
        # BackendEntry(
        #     label        = "Fusion v1 60k (no pop)",
        #     results_path = _OUT / "retrieved_docs_fusion_v1_60k.csv",
        #     key          = "fusion_v1_60k",
        # ),
        # BackendEntry(
        #     label        = "Fusion v4 (Approach A, no pop)",
        #     results_path = _OUT / "retrieved_docs_fusion_v4.csv",
        #     key          = "fusion_v4",
        # ),
        
    ],
    top_k             = 10,
    k_values_detailed = [1, 3, 5, 10],
    decile_mode       = "chunk_weighted",
    excluded_datasets = ["hotpot_qa", "trex"],
    force_recompute   = False,
)

# ── Unpack into module-level names (used by notebooks) ────────────────────────

BACKENDS:        list[BackendEntry]      = CFG.backends
ALL_STRATEGIES:  list[str]              = CFG.all_keys
_STRATEGY_ENTRY: dict[str, BackendEntry] = CFG.entry_by_key
_STRATEGY_LABEL: dict[str, str]         = CFG.label_by_key

COLLECTION_ROOT:   Path = CFG.corpus_root
COLLECTION_NAME:   str  = CFG.corpus_root.name
QUESTIONS_PATH:    Path = CFG.corpus_root / "all_qa_8k.parquet"
CORPUS_PATH:       Path = CFG.corpus_root / "wiki_corpus.parquet"
RESULTS_DIR:       Path = CFG.results_dir
IMAGES_DIR:        Path = CFG.images_dir

TOP_K:             int       = CFG.top_k
K_VALUES_DETAILED: list[int] = CFG.k_values_detailed
DECILE_MODE:       str       = CFG.decile_mode
EXCLUDED_DATASETS: list[str] = CFG.excluded_datasets
FORCE_RECOMPUTE:   bool      = CFG.force_recompute

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

_CACHE_DIR = RESULTS_DIR / "_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_EXCLUDED_DATASETS_CACHE_SUFFIX = (
    "all" if not EXCLUDED_DATASETS else "exclude_" + "_".join(sorted(EXCLUDED_DATASETS))
)



def strategy_label(strategy: str) -> str:
    """Human-readable label for a strategy key."""
    return _STRATEGY_LABEL.get(strategy, strategy)

# ═════════════════════════════════════════════════════════════════════════════
# ── Plotting helpers ──────────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════

# Exact-key colour assignments.
_STRATEGY_COLOR_MAP: dict[str, str] = {
    # Baselines
    "retrieved_docs_bm25_plus":                         "#F59E0B",
    "retrieved_docs_ivfpq_high":                        "#3B82F6",
    "retrieved_docs_faiss_hybrid":                      "#06B6D4",
    # Anti-overfitting router — frozen BERT (with popularity)
    "router60k_frozen_wd5e-3_drop60_mrr20_s42":         "#166534",
    "router60k_frozen_wd5e-3_drop60_mrr20_s42_hybrid":  "#86EFAC",
    # Anti-overfitting router — no popularity
    "router60k_nopop_wd5e-3_drop60_mrr20_s42":          "#831843",
    "router60k_nopop_wd5e-3_drop60_mrr20_s42_hybrid":   "#F9A8D4",
    # Fusion
    "fusion_v4":                                         "#DC2626",
}

_FALLBACK_COLORS = [
    "#F59E0B", "#10B981", "#3B82F6", "#EF4444",
    "#8B5CF6", "#06B6D4", "#F97316", "#84CC16",
]

def _assign_colors() -> dict[str, str]:
    result: dict[str, str] = {}
    fallback_idx = 0
    for s in ALL_STRATEGIES:
        if s in _STRATEGY_COLOR_MAP:
            result[s] = _STRATEGY_COLOR_MAP[s]
        else:
            result[s] = _FALLBACK_COLORS[fallback_idx % len(_FALLBACK_COLORS)]
            fallback_idx += 1
    return result

strategy_colors: dict[str, str] = _assign_colors()
_MARKER_CYCLE = ["o", "s", "^", "D", "v", "P", "X", "p", "*", "h"]
markers: dict[str, str] = {s: _MARKER_CYCLE[i % len(_MARKER_CYCLE)] for i, s in enumerate(ALL_STRATEGIES)}

# Backward-compatible alias
_pick_group_col = pick_group_col


# ═════════════════════════════════════════════════════════════════════════════
# ── Load boundaries & corpus metadata ────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════

from src.corpus_handler import ParquetCorpusHandler

k_values_full = list(range(1, max(TOP_K, max(K_VALUES_DETAILED)) + 1))

metadata_path = COLLECTION_ROOT / "metadata.json"
boundaries_uw, boundaries_cw, corpus_stats = load_boundaries_from_metadata(metadata_path)
print(f"✓ Boundaries loaded (UW: {boundaries_uw[0]:.4f}…{boundaries_uw[-1]:.4f})")

corpus_avg_doc_length = load_avg_doc_length_per_decile(corpus_stats, DECILE_MODE)

dists = load_corpus_distributions(corpus_stats, DECILE_MODE)
if dists is None:
    raise RuntimeError(
        "Corpus distributions not found in metadata.json.\n"
        "Rebuild the collection metadata with the current indexing workflow."
    )
corpus_docs, corpus_chunks = dists

decile_col = decile_col_for(DECILE_MODE)

# ═════════════════════════════════════════════════════════════════════════════
# ── Cache validity helpers ────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════

def _cache_is_valid(key: str) -> bool:
    entry      = _STRATEGY_ENTRY[key]
    cache_path = _CACHE_DIR / f"enriched_{key}.parquet"
    if not cache_path.exists() or not entry.results_path.exists():
        return False
    return cache_path.stat().st_mtime >= entry.results_path.stat().st_mtime

def _metrics_cache_is_valid() -> bool:
    cache_path = _CACHE_DIR / f"metrics_by_strategy_{_EXCLUDED_DATASETS_CACHE_SUFFIX}.json"
    if not cache_path.exists():
        return False
    mtime = cache_path.stat().st_mtime
    for k in ALL_STRATEGIES:
        ep = _CACHE_DIR / f"enriched_{k}.parquet"
        if not ep.exists() or ep.stat().st_mtime > mtime:
            return False
    return True


def _decile_metrics_cache_is_valid() -> bool:
    cache_path = _CACHE_DIR / f"decile_metrics_by_strategy_{_EXCLUDED_DATASETS_CACHE_SUFFIX}.json"
    if not cache_path.exists():
        return False
    mtime = cache_path.stat().st_mtime
    for k in ALL_STRATEGIES:
        ep = _CACHE_DIR / f"enriched_{k}.parquet"
        if not ep.exists() or ep.stat().st_mtime > mtime:
            return False
    return True

# ═════════════════════════════════════════════════════════════════════════════
# ── CSV → results-parquet conversion ─────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════

def _csv_to_results_df(csv_path: Path, qa_parquet_path: Path) -> pd.DataFrame:
    """Convert a ``retrieved_docs_<key>.csv`` checkpoint into the per-question
    results DataFrame expected by the rest of shared_setup.

    The CSV has one row per retrieved document.  This function pivots it into
    one row per question, aggregating ``topk_ids``, ``topk_scores``, and
    ``topk_popularities`` as lists, then joins question-level metadata
    (``question``, ``wikipedia_id``, ``dataset``, decile columns) from the QA
    parquet.

    The retrieval runner uses ``cyro_qa_cache.parquet`` (hashed question IDs)
    as its question source, so we prefer that file over ``all_qa_8k.parquet``
    (which uses HuggingFace dataset IDs) when it exists alongside the CSV.

    Args:
        csv_path: Path to the ``retrieved_docs_<key>.csv`` file.
        qa_parquet_path: Path to the ``all_qa_8k.parquet`` (or equivalent) file
            that supplies per-question metadata.  If a ``cyro_qa_cache.parquet``
            exists in the same directory as the CSV, it is used instead.

    Returns:
        DataFrame with the same schema as ``results_<key>.parquet``.

    Raises:
        FileNotFoundError: If either input file is missing.
        KeyError: If expected columns are absent from the CSV or QA parquet.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV checkpoint not found: {csv_path}")

    # Prefer cyro_qa_cache.parquet (same IDs as the retrieval runner uses)
    cyro_cache = csv_path.parent / "cyro_qa_cache.parquet"
    if cyro_cache.exists():
        qa = pd.read_parquet(cyro_cache)
    elif qa_parquet_path.exists():
        qa = pd.read_parquet(qa_parquet_path)
    else:
        raise FileNotFoundError(f"QA parquet not found: {qa_parquet_path}")

    raw = pd.read_csv(csv_path)

    # Ensure question_id types match for merging
    raw["question_id"] = raw["question_id"].astype(str)
    qa["question_id"]  = qa["question_id"].astype(str)

    # Sort by question then rank so list order is correct
    raw = raw.sort_values(["question_id", "doc_rank"])

    # Aggregate per question → lists
    def _agg(g: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "topk_ids":          [str(x) for x in g["metadata_wikipedia_id"].tolist()],
            "topk_scores":       g["metadata_score"].tolist()
                                 if "metadata_score" in g.columns
                                 else [float("nan")] * len(g),
            "topk_popularities": g["metadata_popularity_avg"].tolist(),
        })

    has_score = "metadata_score" in raw.columns

    def _agg(g: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "topk_ids":          [str(x) for x in g["metadata_wikipedia_id"].tolist()],
            "topk_scores":       g["metadata_score"].tolist() if has_score else [float("nan")] * len(g),
            "topk_popularities": g["metadata_popularity_avg"].tolist(),
        })

    agg = raw.groupby("question_id", sort=False).apply(_agg)
    agg = agg.drop(columns=["question_id"], errors="ignore").reset_index()

    # Join with QA metadata
    meta_cols = [
        "question_id", "question_text", "wikipedia_id", "wikipedia_title",
        "popularity_avg", "dataset",
        "pop_decile_unweighted", "pop_decile_chunk_weighted", "decile",
    ]
    qa_sub = qa[[c for c in meta_cols if c in qa.columns]].copy()
    qa_sub["wikipedia_id"] = qa_sub["wikipedia_id"].astype(str)

    merged = agg.merge(qa_sub, on="question_id", how="left")

    # Rename to match parquet schema
    merged = merged.rename(columns={"question_text": "question"})

    col_order = [
        "question", "wikipedia_id", "wikipedia_title", "popularity_avg",
        "dataset", "pop_decile_unweighted", "pop_decile_chunk_weighted", "decile",
        "topk_ids", "topk_scores", "topk_popularities",
    ]
    col_order = [c for c in col_order if c in merged.columns]
    return merged[col_order]




results_by_strategy: dict[str, pd.DataFrame] = {}

for _key in ALL_STRATEGIES:
    entry       = _STRATEGY_ENTRY[_key]
    source_path = entry.results_path
    cache_path  = _CACHE_DIR / f"enriched_{_key}.parquet"
    corpus_path = entry.corpus_path
    meta_path   = entry.metadata_path

    if not source_path.exists():
        print(f"  ⚠ Missing: {source_path.name} — run the pipeline first!")
        continue

    if not FORCE_RECOMPUTE and _cache_is_valid(_key):
        print(f"  ✓ {strategy_label(_key)}: loading from cache")
        _df_cache = pd.read_parquet(cache_path)
        if EXCLUDED_DATASETS:
            _gc = pick_group_col(_df_cache)
            if _gc:
                _before = len(_df_cache)
                _df_cache = _df_cache[~_df_cache[_gc].isin(EXCLUDED_DATASETS)].copy()
                _dropped = _before - len(_df_cache)
                if _dropped:
                    print(f"    excluded {_dropped:,} rows matching EXCLUDED_DATASETS ({EXCLUDED_DATASETS})")
        results_by_strategy[_key] = _df_cache
        continue

    print(f"  computing {strategy_label(_key)}…")
    if entry.is_csv:
        print(f"    converting CSV checkpoint → results format…")
        df = _csv_to_results_df(source_path, QUESTIONS_PATH)
    else:
        df = pd.read_parquet(source_path)
    print(f"    loaded {len(df):,} rows")

    if COL_POPULARITY not in df.columns:
        raise ValueError(f"{_key}: missing '{COL_POPULARITY}' column")

    before = len(df)
    df = df.dropna(subset=[COL_POPULARITY]).copy()
    df = df[df[COL_POPULARITY] >= 0].copy()
    if len(df) < before:
        print(f"    dropped {before - len(df)} rows with missing/invalid popularity")

    df[COL_DECILE_UNWEIGHTED]     = assign_decile(df[COL_POPULARITY], boundaries_uw).astype(int)
    df[COL_DECILE_CHUNK_WEIGHTED] = assign_decile(df[COL_POPULARITY], boundaries_cw).astype(int)

    print("    computing recall@k / MRR…")
    _, df = compute_metrics(df, k_values_full)

    df["found_at_rank"]           = df.apply(get_found_rank, axis=1)
    df["wrong_docs_popularities"] = df.apply(get_wrong_pops, axis=1)

    def _query_entropy(row) -> float:
        scores = row.get("topk_scores")
        if scores is None:
            return float("nan")
        s = np.array(list(scores), dtype=float)
        return float("nan") if len(s) == 0 else s.mean()

    df["query_entropy"] = df.apply(_query_entropy, axis=1)

    _entropy_valid = df["query_entropy"].notna() & df["query_entropy"].rank(method="first").notna()
    if _entropy_valid.sum() >= 5:
        df["entropy_group"] = pd.qcut(
            df["query_entropy"].rank(method="first"),
            q=5,
            labels=["Q1_least", "Q2", "Q3", "Q4", "Q5_most"],
            duplicates="drop",
        ).astype(str)
    else:
        df["entropy_group"] = "N/A"

    print("    fetching doc lengths via corpus handler…")
    _corpus_handler = ParquetCorpusHandler(corpus_path=corpus_path, metadata_path=meta_path)
    question_ids_str: list[str] = df["wikipedia_id"].unique().tolist()
    question_ids_int: list[int] = []
    for _wid in question_ids_str:
        try:
            question_ids_int.append(int(_wid))
        except (ValueError, TypeError):
            pass
    _docs = _corpus_handler.get_documents(question_ids_int)
    doc_length_df = pd.DataFrame({
        "wikipedia_id": [str(d.metadata["wikipedia_id"]) for d in _docs],
        "doc_length":   [len(d.page_content) for d in _docs],
    }).drop_duplicates(subset="wikipedia_id", keep="first")
    print(f"    matched {len(doc_length_df):,} / {len(question_ids_str):,} IDs")
    if "doc_length" in df.columns:
        df = df.drop(columns=["doc_length"])
    df = df.merge(doc_length_df[["wikipedia_id", "doc_length"]], on="wikipedia_id", how="left")
    del _docs, doc_length_df
    gc.collect()

    df.to_parquet(cache_path, index=False)
    print(f"    cached → {cache_path.name}")
    if EXCLUDED_DATASETS:
        _gc = pick_group_col(df)
        if _gc:
            before = len(df)
            df = df[~df[_gc].isin(EXCLUDED_DATASETS)].copy()
            dropped = before - len(df)
            if dropped:
                print(f"    excluded {dropped:,} rows matching EXCLUDED_DATASETS ({EXCLUDED_DATASETS})")
    results_by_strategy[_key] = df

# Restrict ALL_STRATEGIES to only those that actually loaded
ALL_STRATEGIES = [k for k in ALL_STRATEGIES if k in results_by_strategy]

if not ALL_STRATEGIES:
    raise FileNotFoundError("No retrieval results found! Run the pipeline first.")

# Resolve a single GROUP_COL shared across all loaded strategies
GROUP_COL: str = ""
for _s in ALL_STRATEGIES:
    _gc = pick_group_col(results_by_strategy[_s])
    if _gc:
        GROUP_COL = _gc
        break

# ═════════════════════════════════════════════════════════════════════════════
# ── Aggregate metrics ─────────────────────────────────────────────════════════
# ═════════════════════════════════════════════════════════════════════════════

_metrics_cache_path = _CACHE_DIR / f"metrics_by_strategy_{_EXCLUDED_DATASETS_CACHE_SUFFIX}.json"

if not FORCE_RECOMPUTE and _metrics_cache_is_valid():
    with open(_metrics_cache_path) as _f:
        metrics_by_strategy: dict[str, dict] = json.load(_f)
    print("✓ Metrics loaded from cache")
else:
    metrics_by_strategy = {}
    for strategy, rdf in results_by_strategy.items():
        row: dict[str, Any] = {}
        for k in k_values_full:
            col = f"recall@{k}"
            if col in rdf.columns:
                row[f"recall@{k}"] = float(rdf[col].mean())
        if "reciprocal_rank" in rdf.columns:
            row["mrr"] = float(rdf["reciprocal_rank"].mean())
        row["median_rank"] = float(rdf["rank"].dropna().median()) if "rank" in rdf.columns else None
        row["mean_rank"]   = float(rdf["rank"].dropna().mean())   if "rank" in rdf.columns else None
        metrics_by_strategy[strategy] = row
    with open(_metrics_cache_path, "w") as _f:
        json.dump(metrics_by_strategy, _f)

pd.DataFrame(metrics_by_strategy).T.to_csv(IMAGES_DIR / "metrics_comparison.csv")

# ═════════════════════════════════════════════════════════════════════════════
# ── Per-decile metrics ────────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════

_decile_cache_path = _CACHE_DIR / f"decile_metrics_by_strategy_{_EXCLUDED_DATASETS_CACHE_SUFFIX}.json"

if not FORCE_RECOMPUTE and _decile_metrics_cache_is_valid():
    with open(_decile_cache_path) as _f:
        _raw = json.load(_f)
    decile_metrics_by_strategy: dict[str, pd.DataFrame] = {
        s: pd.DataFrame(rows) for s, rows in _raw.items()
    }
    print("✓ Per-decile metrics loaded from cache")
else:
    decile_metrics_by_strategy = {}
    for strategy in ALL_STRATEGIES:
        rdf = results_by_strategy[strategy]
        valid_results = rdf[rdf[decile_col] >= 0]
        decile_metrics = []
        for decile in range(10):
            decile_data = valid_results[valid_results[decile_col] == decile]
            n = len(decile_data)
            if n == 0:
                continue
            metrics_row: dict[str, Any] = {"decile": decile, "count": n}

            for k in k_values_full:
                col = f"recall@{k}"
                vals = decile_data[col].dropna()
                nk = len(vals)
                if nk > 0:
                    p    = vals.mean()
                    se   = np.sqrt(p * (1 - p) / nk)
                    ci95 = 1.96 * se
                else:
                    p, se, ci95 = np.nan, np.nan, np.nan
                metrics_row[col]           = p
                metrics_row[f"{col}_se"]   = se
                metrics_row[f"{col}_ci95"] = ci95

            rr_vals = decile_data["reciprocal_rank"].dropna()
            nr = len(rr_vals)
            if nr > 0:
                metrics_row["mrr"]      = rr_vals.mean()
                metrics_row["mrr_se"]   = rr_vals.std(ddof=1) / np.sqrt(nr)
                metrics_row["mrr_ci95"] = 1.96 * metrics_row["mrr_se"]
            else:
                metrics_row["mrr"] = metrics_row["mrr_se"] = metrics_row["mrr_ci95"] = np.nan

            doc_lens = decile_data["doc_length"].dropna()
            if len(doc_lens) > 0:
                metrics_row["avg_expected_doc_length"]    = doc_lens.mean()
                metrics_row["avg_expected_doc_length_se"] = doc_lens.std(ddof=1) / np.sqrt(len(doc_lens))
            else:
                metrics_row["avg_expected_doc_length"]    = np.nan
                metrics_row["avg_expected_doc_length_se"] = np.nan

            metrics_row["avg_corpus_doc_length"] = (
                corpus_avg_doc_length[decile] if corpus_avg_doc_length is not None else np.nan
            )

            if (corpus_avg_doc_length is not None
                    and corpus_avg_doc_length[decile] > 0
                    and not np.isnan(metrics_row.get("avg_expected_doc_length", np.nan))):
                metrics_row["doc_length_ratio"] = (
                    metrics_row["avg_expected_doc_length"] / corpus_avg_doc_length[decile]
                )
            else:
                metrics_row["doc_length_ratio"] = np.nan

            decile_metrics.append(metrics_row)

        decile_metrics_by_strategy[strategy] = pd.DataFrame(decile_metrics)

    with open(_decile_cache_path, "w") as _f:
        json.dump(
            {s: dm.to_dict(orient="records") for s, dm in decile_metrics_by_strategy.items()},
            _f,
        )

print(f"\n✓ shared_setup complete — {len(ALL_STRATEGIES)} strategies ready")
print(f"  Corpus root: {COLLECTION_ROOT}")
print(f"  Results dir: {RESULTS_DIR}")
for b in BACKENDS:
    idx = f" → index: {b.index_path}" if b.index_path else ""
    print(f"    {b.label!r} (key={b.key}){idx}")
print(f"  Decile mode: {DECILE_MODE}")
print(f"  Group col  : {GROUP_COL or '(none found)'}")
