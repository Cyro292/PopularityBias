"""shared_setup.py — Common setup for all eval/ notebooks.

Import this at the top of every eval notebook with:

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
    from eval.shared_setup import *

It exports:
  - All standard imports (os, Path, pd, np, plt, sns, tqdm, warnings)
  - Config constants: COLLECTION_NAME, COLLECTION_ROOT, QUESTIONS_PATH,
    CORPUS_PATH, STRATEGIES, TOP_K, K_VALUES_DETAILED, DECILE_MODE,
    OUTPUT_FOLDER, RESULTS_DIR
  - Loaded data: results_by_strategy, ALL_STRATEGIES, metrics_by_strategy,
    decile_metrics_by_strategy, boundaries_uw, boundaries_cw, corpus_stats,
    corpus_docs, corpus_chunks, corpus_avg_doc_length, metadata_path
  - Helper functions: _pick_group_col, strategy_colors, markers
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

# ---------------------------------------------------------------------------
# Path bootstrap: make the repo root importable regardless of cwd
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import DATA_DIR
from helpers.decile_utils import (
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
from helpers.metrics import (
    compute_metrics,
    get_found_rank,
    get_wrong_pops,
    pick_group_col,
)

load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────────

OUTPUT_NAME = "all_qa_8k"

COLLECTION_NAME = "wiki_full_bil"
COLLECTION_ROOT = Path(DATA_DIR) / COLLECTION_NAME
QUESTIONS_PATH  = COLLECTION_ROOT / "all_qa_8k.parquet"
CORPUS_PATH     = COLLECTION_ROOT / "wiki_corpus.parquet"

STRATEGIES         = ["approximation", "bm25"]
TOP_K              = 10
K_VALUES_DETAILED  = [1, 3, 5, 10]

DECILE_MODE   = "chunk_weighted"   # "chunk_weighted" or "unweighted"
OUTPUT_FOLDER = OUTPUT_NAME
RESULTS_DIR   = COLLECTION_ROOT / OUTPUT_FOLDER
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Set to True to ignore any cached files and recompute everything from scratch.
FORCE_RECOMPUTE = False

_CACHE_DIR = RESULTS_DIR / "_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Plotting helpers ──────────────────────────────────────────────────────────

strategy_colors: dict[str, str] = {
    "approximation": "#3498db",
    "vector":        "#3498db",
    "bm25":          "#e74c3c",
    "hybrid":        "#27ae60",
}
markers: dict[str, str] = {
    "approximation": "o",
    "vector":        "o",
    "bm25":          "s",
    "hybrid":        "D",
}

# Backward-compatible alias: notebooks that do `from eval.shared_setup import *`
# still get _pick_group_col under the old name.
_pick_group_col = pick_group_col


# ── Load everything ───────────────────────────────────────────────────────────

import pyarrow.parquet as pq
import pyarrow.compute as pc

k_values_full = list(range(1, max(TOP_K, max(K_VALUES_DETAILED)) + 1))

# --- Boundaries & metadata (always fast) ---
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

# ── Helper: check whether cached enriched results are still fresh ─────────────

def _cache_is_valid(strategy: str) -> bool:
    """Return True if the cached enriched parquet exists and is newer than the source."""
    cache_path  = _CACHE_DIR / f"enriched_{strategy}.parquet"
    source_path = RESULTS_DIR / f"results_{strategy}.parquet"
    if not cache_path.exists() or not source_path.exists():
        return False
    return cache_path.stat().st_mtime >= source_path.stat().st_mtime


def _metrics_cache_is_valid() -> bool:
    cache_path = _CACHE_DIR / "metrics_by_strategy.json"
    if not cache_path.exists():
        return False
    # Valid if every enriched parquet is at least as new as the metrics cache
    mtime = cache_path.stat().st_mtime
    for s in STRATEGIES:
        ep = _CACHE_DIR / f"enriched_{s}.parquet"
        if not ep.exists() or ep.stat().st_mtime > mtime:
            return False
    return True


def _decile_metrics_cache_is_valid() -> bool:
    cache_path = _CACHE_DIR / "decile_metrics_by_strategy.json"
    if not cache_path.exists():
        return False
    mtime = cache_path.stat().st_mtime
    for s in STRATEGIES:
        ep = _CACHE_DIR / f"enriched_{s}.parquet"
        if not ep.exists() or ep.stat().st_mtime > mtime:
            return False
    return True


# ── Load / compute enriched results ──────────────────────────────────────────

results_by_strategy: dict[str, pd.DataFrame] = {}

for strategy in STRATEGIES:
    source_path = RESULTS_DIR / f"results_{strategy}.parquet"
    cache_path  = _CACHE_DIR / f"enriched_{strategy}.parquet"

    if not source_path.exists():
        print(f"  ⚠ Missing: {source_path} — run rag_retrieval.ipynb first!")
        continue

    if not FORCE_RECOMPUTE and _cache_is_valid(strategy):
        print(f"  ✓ {strategy}: loading from cache")
        results_by_strategy[strategy] = pd.read_parquet(cache_path)
        continue

    print(f"  computing {strategy}…")
    df = pd.read_parquet(source_path)
    print(f"    loaded {len(df):,} rows")

    # — decile columns —
    if COL_POPULARITY not in df.columns:
        raise ValueError(f"{strategy}: missing '{COL_POPULARITY}' column")

    before = len(df)
    df = df.dropna(subset=[COL_POPULARITY]).copy()
    df = df[df[COL_POPULARITY] >= 0].copy()
    if len(df) < before:
        print(f"    dropped {before - len(df)} rows with missing/invalid popularity")

    df[COL_DECILE_UNWEIGHTED]     = assign_decile(df[COL_POPULARITY], boundaries_uw).astype(int)
    df[COL_DECILE_CHUNK_WEIGHTED] = assign_decile(df[COL_POPULARITY], boundaries_cw).astype(int)

    # — recall@k / MRR / rank columns —
    print("    computing recall@k / MRR…")
    _, df = compute_metrics(df, k_values_full)

    # — found_at_rank / wrong_docs_popularities —
    df["found_at_rank"]           = df.apply(get_found_rank, axis=1)
    df["wrong_docs_popularities"] = df.apply(get_wrong_pops, axis=1)

    # — entropy —
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

    # — doc_length from corpus —
    print("    scanning corpus for doc lengths…")
    question_ids: set[str] = set(df["wikipedia_id"].unique())
    length_parts: list[pd.DataFrame] = []
    pf = pq.ParquetFile(str(CORPUS_PATH))
    for batch in pf.iter_batches(batch_size=100_000, columns=["wikipedia_id", "text"]):
        lengths = pc.utf8_length(batch.column("text"))
        part = pd.DataFrame({
            "wikipedia_id": batch.column("wikipedia_id").to_pandas().astype(str).str.strip(),
            "doc_length":   lengths.to_pandas(),
        })
        part = part[part["wikipedia_id"].isin(question_ids)]
        length_parts.append(part)
    doc_length_df = pd.concat(length_parts, ignore_index=True).drop_duplicates(
        subset="wikipedia_id", keep="first"
    )
    print(f"    matched {len(doc_length_df):,} / {len(question_ids):,} IDs")
    if "doc_length" in df.columns:
        df = df.drop(columns=["doc_length"])
    df = df.merge(doc_length_df[["wikipedia_id", "doc_length"]], on="wikipedia_id", how="left")
    del length_parts, doc_length_df
    gc.collect()

    # — save cache —
    df.to_parquet(cache_path, index=False)
    print(f"    cached → {cache_path.name}")

    results_by_strategy[strategy] = df

ALL_STRATEGIES = list(results_by_strategy.keys())
if not ALL_STRATEGIES:
    raise FileNotFoundError("No retrieval results found! Run rag_retrieval.ipynb first.")

# ── Aggregate metrics ─────────────────────────────────────────────────────────

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
            found = rdf["reciprocal_rank"].dropna()
        row["median_rank"] = float(rdf["rank"].dropna().median()) if "rank" in rdf.columns else None
        row["mean_rank"]   = float(rdf["rank"].dropna().mean())   if "rank" in rdf.columns else None
        metrics_by_strategy[strategy] = row
    with open(_metrics_cache_path, "w") as _f:
        json.dump(metrics_by_strategy, _f)

pd.DataFrame(metrics_by_strategy).T.to_csv(RESULTS_DIR / "metrics_comparison.csv")

# ── Per-decile metrics ────────────────────────────────────────────────────────

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
print(f"  Strategies : {STRATEGIES}")
print(f"  Decile mode: {DECILE_MODE}")
