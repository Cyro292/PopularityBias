"""shared_setup.py — Common setup for all eval/ notebooks.

Import this at the top of every eval notebook with:

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))
    from notebooks.eval.shared_setup import *

It exports:
  - All standard imports (os, Path, pd, np, plt, sns, tqdm, warnings)
  - Config constants: COLLECTION_NAME, COLLECTION_ROOT, QUESTIONS_PATH,
    CORPUS_PATH, BACKENDS, ALL_STRATEGIES, TOP_K, K_VALUES_DETAILED,
    DECILE_MODE, OUTPUT_FOLDER, RESULTS_DIR, GROUP_COL
  - Loaded data: results_by_strategy, metrics_by_strategy,
    decile_metrics_by_strategy, boundaries_uw, boundaries_cw, corpus_stats,
    corpus_docs, corpus_chunks, corpus_avg_doc_length, metadata_path
  - Helper functions: pick_group_col, strategy_colors, markers, strategy_label
  - Metric helper functions: compute_metrics, get_found_rank, get_wrong_pops
"""
from __future__ import annotations

import gc
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

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
# ── CONFIGURATION — edit here to switch backends / strategies ─────────────
# ═════════════════════════════════════════════════════════════════════════════

# Map each backend name to the list of strategy keys it produced.
# Results must exist as:  RESULTS_DIR / f"results_{strategy}.parquet"
# Remove a backend or strategy by commenting it out.
BACKENDS: dict[str, list[str]] = {
    "elasticsearch": ["approximation", "bm25"],
    "faiss":         ["ivfpq"],
}

OUTPUT_NAME   = "all_qa_8k"        # Folder name for results (must match indexing output folder)
COLLECTION_NAME = "wiki_full_bil"
COLLECTION_ROOT = Path(DATA_DIR) / COLLECTION_NAME
QUESTIONS_PATH  = COLLECTION_ROOT / "all_qa_8k.parquet"
CORPUS_PATH     = COLLECTION_ROOT / "wiki_corpus.parquet"

TOP_K              = 1
K_VALUES_DETAILED  = [1, 3, 5, 10]
DECILE_MODE        = "chunk_weighted"   # "chunk_weighted" or "unweighted"
OUTPUT_FOLDER      = "all_qa_8k"            # Folder name for storing all results (must match indexing output folder)
RESULTS_DIR        = COLLECTION_ROOT / OUTPUT_FOLDER
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Flat list of all active strategies (order: backend order, then strategy order)
ALL_STRATEGIES: list[str] = [s for strats in BACKENDS.values() for s in strats]

# Datasets to exclude from all analyses. Set to [] to include everything.
# Example: ["hotpot_qa", "trex"]
EXCLUDED_DATASETS: list[str] = ["hotpot_qa", "trex"]

# Set to True to ignore any cached files and recompute everything from scratch.
FORCE_RECOMPUTE = False

_CACHE_DIR = RESULTS_DIR / "_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════════
# ── Plotting helpers ──────────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════

# Assign a distinct colour per strategy, respecting legacy names.
_STRATEGY_COLOR_MAP: dict[str, str] = {
    "approximation": "#3498db",
    "vector":        "#3498db",
    "bm25":          "#e74c3c",
    "hybrid":        "#27ae60",
    "ivfpq":         "#9b59b6",
    "hnsw":          "#f39c12",
}
_FALLBACK_COLORS = ["#1abc9c", "#e67e22", "#2c3e50", "#8e44ad", "#16a085"]

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

markers: dict[str, str] = {s: ("o" if i % 2 == 0 else "s") for i, s in enumerate(ALL_STRATEGIES)}


def strategy_label(strategy: str) -> str:
    """Human-readable label: 'backend / strategy'."""
    for backend, strats in BACKENDS.items():
        if strategy in strats:
            return f"{backend} / {strategy}"
    return strategy


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
        f"Run: python scripts/patch_corpus_distributions.py {COLLECTION_ROOT}"
    )
corpus_docs, corpus_chunks = dists

decile_col = decile_col_for(DECILE_MODE)

# ═════════════════════════════════════════════════════════════════════════════
# ── Cache validity helpers ────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════

def _cache_is_valid(strategy: str) -> bool:
    cache_path  = _CACHE_DIR / f"enriched_{strategy}.parquet"
    source_path = RESULTS_DIR / f"results_{strategy}.parquet"
    if not cache_path.exists() or not source_path.exists():
        return False
    return cache_path.stat().st_mtime >= source_path.stat().st_mtime


def _metrics_cache_is_valid() -> bool:
    cache_path = _CACHE_DIR / "metrics_by_strategy.json"
    if not cache_path.exists():
        return False
    mtime = cache_path.stat().st_mtime
    for s in ALL_STRATEGIES:
        ep = _CACHE_DIR / f"enriched_{s}.parquet"
        if not ep.exists() or ep.stat().st_mtime > mtime:
            return False
    return True


def _decile_metrics_cache_is_valid() -> bool:
    cache_path = _CACHE_DIR / "decile_metrics_by_strategy.json"
    if not cache_path.exists():
        return False
    mtime = cache_path.stat().st_mtime
    for s in ALL_STRATEGIES:
        ep = _CACHE_DIR / f"enriched_{s}.parquet"
        if not ep.exists() or ep.stat().st_mtime > mtime:
            return False
    return True

# ═════════════════════════════════════════════════════════════════════════════
# ── Load / enrich results ─────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════

results_by_strategy: dict[str, pd.DataFrame] = {}

for strategy in ALL_STRATEGIES:
    source_path = RESULTS_DIR / f"results_{strategy}.parquet"
    cache_path  = _CACHE_DIR / f"enriched_{strategy}.parquet"

    if not source_path.exists():
        print(f"  ⚠ Missing: {source_path.name} — run the pipeline first!")
        continue

    if not FORCE_RECOMPUTE and _cache_is_valid(strategy):
        print(f"  ✓ {strategy_label(strategy)}: loading from cache")
        _df_cache = pd.read_parquet(cache_path)
        if EXCLUDED_DATASETS:
            _gc = pick_group_col(_df_cache)
            if _gc:
                _before = len(_df_cache)
                _df_cache = _df_cache[~_df_cache[_gc].isin(EXCLUDED_DATASETS)].copy()
                _dropped = _before - len(_df_cache)
                if _dropped:
                    print(f"    excluded {_dropped:,} rows matching EXCLUDED_DATASETS ({EXCLUDED_DATASETS})")
        results_by_strategy[strategy] = _df_cache
        continue

    print(f"  computing {strategy_label(strategy)}…")
    df = pd.read_parquet(source_path)
    print(f"    loaded {len(df):,} rows")

    if COL_POPULARITY not in df.columns:
        raise ValueError(f"{strategy}: missing '{COL_POPULARITY}' column")

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
    df["entropy_group"] = pd.qcut(
        df["query_entropy"].rank(method="first"),
        q=5,
        labels=["Q1_least", "Q2", "Q3", "Q4", "Q5_most"],
    ).astype(str)

    print("    fetching doc lengths via corpus handler…")
    _corpus_handler = ParquetCorpusHandler(corpus_path=CORPUS_PATH, metadata_path=metadata_path)
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
    results_by_strategy[strategy] = df

# Restrict ALL_STRATEGIES to only those that actually loaded
ALL_STRATEGIES = [s for s in ALL_STRATEGIES if s in results_by_strategy]

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

_metrics_cache_path = _CACHE_DIR / "metrics_by_strategy.json"

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

pd.DataFrame(metrics_by_strategy).T.to_csv(RESULTS_DIR / "metrics_comparison.csv")

# ═════════════════════════════════════════════════════════════════════════════
# ── Per-decile metrics ────────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════

_decile_cache_path = _CACHE_DIR / "decile_metrics_by_strategy.json"

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
print(f"  Collection : {COLLECTION_NAME}")
print(f"  Results dir: {RESULTS_DIR}")
print(f"  Backends   : {BACKENDS}")
print(f"  Strategies : {ALL_STRATEGIES}")
print(f"  Decile mode: {DECILE_MODE}")
print(f"  Group col  : {GROUP_COL or '(none found)'}")
