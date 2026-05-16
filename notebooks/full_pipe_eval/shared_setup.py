"""shared_setup.py — Common setup for all full_pipe_eval/ notebooks.

Import this at the top of every notebook with:

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))
    from notebooks.full_pipe_eval.shared_setup import *

It exports:
  - All standard imports (Path, pd, np, plt, sns, warnings)
  - Config dataclasses: RunEntry
  - Derived constants: ALL_RUNS, RESULTS_DIR, IMAGES_DIR
  - Loaded data: results_by_run (dict[run_key, DataFrame])
  - Helper functions: run_label, run_colors, pivot_metric
"""
from __future__ import annotations

import gc
import json
import os
import sys
import warnings
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

from dotenv import load_dotenv

# ── Repo root ─────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import DATA_DIR
from src.metrics.decile_utils import (
    load_boundaries_from_metadata,
    COL_DECILE_UNWEIGHTED,
    COL_DECILE_CHUNK_WEIGHTED,
    COL_POPULARITY,
)

load_dotenv()

# ═════════════════════════════════════════════════════════════════════════════
# ── CONFIGURATION — edit here ─────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════

COLLECTION_NAME: str  = "wiki_full_bil"
OUTPUT_DIR:      str  = "all_qa_8k"
DECILE_MODE:     str  = "chunk_weighted"   # "unweighted" | "chunk_weighted"
FORCE_RECOMPUTE: bool = False

# LLM models, retrieval backends, context labels, and evaluator keys that were
# produced by llm_eval_runner.  Adjust to match what you actually ran.
LLM_KEYS:       list[str] = ["neo", "qwen"]
BACKEND_KEYS:   list[str] = ["zero_shot", "bm25_plus", "ivfpq_low", "ivfpq_high"]
CTX_LABELS:     list[str] = ["zero", "top1", "top3"]   # "zero" for zero_shot
EVALUATOR_KEYS: list[str] = ["substring", "binary_mistral"]

# ═════════════════════════════════════════════════════════════════════════════
# ── Derived paths ─────────────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════

COLLECTION_ROOT: Path = DATA_DIR / COLLECTION_NAME
RESULTS_DIR:     Path = COLLECTION_ROOT / OUTPUT_DIR
IMAGES_DIR:      Path = Path(__file__).parent / "images"
_CACHE_DIR:      Path = Path(__file__).parent / ".cache"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

metadata_path = COLLECTION_ROOT / "metadata.json"
boundaries_uw, boundaries_cw, corpus_stats = load_boundaries_from_metadata(metadata_path)
decile_col = COL_DECILE_CHUNK_WEIGHTED if DECILE_MODE == "chunk_weighted" else COL_DECILE_UNWEIGHTED

# The pipeline stores decile values in the metadata dict under a shorter name
# (e.g. "decile_chunk_weighted") rather than the full constant name used by
# decile_utils ("pop_decile_chunk_weighted").  Build a fallback list so notebook
# code can resolve the actual column name after metadata is unpacked.
_DECILE_COL_CANDIDATES: list[str] = [
    decile_col,                     # canonical constant: pop_decile_chunk_weighted / pop_decile_unweighted
    "decile_chunk_weighted",        # short name stored by llm_eval_runner (chunk-weighted)
    "decile_unweighted",            # short name stored by llm_eval_runner (unweighted)
    "decile",                       # legacy fallback
]

# ═════════════════════════════════════════════════════════════════════════════
# ── RunEntry — one parquet file per (llm, backend, ctx, evaluator) ────────────
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class RunEntry:
    """Describes one results parquet produced by llm_eval_runner.

    Attributes:
        llm:       LLM key (e.g. ``"neo"``, ``"qwen"``).
        backend:   Retrieval backend key (e.g. ``"bm25_plus"``, ``"zero_shot"``).
        ctx_label: Context size label (e.g. ``"zero"``, ``"top1"``, ``"top3"``).
        evaluator: Evaluator key (e.g. ``"substring"``, ``"binary_mistral"``).
    """

    llm:       str
    backend:   str
    ctx_label: str
    evaluator: str

    @property
    def key(self) -> str:
        return f"{self.llm}__{self.backend}__{self.ctx_label}__{self.evaluator}"

    @property
    def label(self) -> str:
        ctx = "" if self.ctx_label == "zero" else f" {self.ctx_label}"
        return f"{self.llm} | {self.backend}{ctx} | {self.evaluator}"

    @property
    def results_path(self) -> Path:
        return RESULTS_DIR / f"results_{self.llm}_{self.backend}_{self.ctx_label}_{self.evaluator}.parquet"


def _valid_ctx(backend: str, ctx: str) -> bool:
    """zero_shot only has ctx_label='zero'; retrieval backends skip 'zero'."""
    if backend == "zero_shot":
        return ctx == "zero"
    return ctx != "zero"


ALL_ENTRIES: list[RunEntry] = [
    RunEntry(llm=llm, backend=backend, ctx_label=ctx, evaluator=ev)
    for llm, backend, ctx, ev in product(LLM_KEYS, BACKEND_KEYS, CTX_LABELS, EVALUATOR_KEYS)
    if _valid_ctx(backend, ctx)
]

# ═════════════════════════════════════════════════════════════════════════════
# ── Colours & labels ──────────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════

_BACKEND_COLORS: dict[str, str] = {
    "zero_shot":  "#6B7280",
    "bm25_plus":  "#3B82F6",
    "ivfpq_low":  "#10B981",
    "ivfpq_high": "#F59E0B",
}
_LLM_COLORS: dict[str, str] = {
    "neo":  "#8B5CF6",
    "qwen": "#EF4444",
}
_FALLBACK = ["#64748B", "#0EA5E9", "#22C55E", "#F97316", "#EC4899"]


def run_label(entry: RunEntry) -> str:
    return entry.label


def backend_color(backend: str) -> str:
    return _BACKEND_COLORS.get(backend, "#64748B")


def llm_color(llm: str) -> str:
    return _LLM_COLORS.get(llm, "#64748B")


# ═════════════════════════════════════════════════════════════════════════════
# ── Load results ──────────────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════

results_by_run: dict[str, pd.DataFrame] = {}

print(f"Loading results from: {RESULTS_DIR}")
for entry in ALL_ENTRIES:
    if not entry.results_path.exists():
        print(f"  ⚠ missing: {entry.results_path.name}")
        continue
    df = pd.read_parquet(entry.results_path)

    # ── Unpack the 'metadata' dict column into top-level columns ──────────────
    # The pipeline stores per-row context (dataset, decile, retrieved_doc_ids,
    # popularity_avg, wikipedia_id, …) inside a 'metadata' dict column.
    # Flatten it here so all notebooks can reference columns directly.
    if "metadata" in df.columns:
        meta_df = pd.json_normalize(df["metadata"].where(df["metadata"].notna(), None))
        # Only add columns that don't already exist to avoid clobbering llm/backend etc.
        for col in meta_df.columns:
            if col not in df.columns:
                df[col] = meta_df[col].values

    # Attach run metadata columns for easy filtering/grouping
    df["llm"]       = entry.llm
    df["backend"]   = entry.backend
    df["ctx_label"] = entry.ctx_label
    df["evaluator"] = entry.evaluator
    results_by_run[entry.key] = df
    print(f"  ✓ {entry.label}: {len(df):,} rows")

ALL_RUNS: list[str] = list(results_by_run.keys())

if not ALL_RUNS:
    raise FileNotFoundError(
        "No results parquet files found!\n"
        f"Expected files like: results_neo_bm25_plus_top1_substring.parquet\n"
        f"in: {RESULTS_DIR}\n"
        "Run llm_eval_runner first."
    )

# ── Combined dataframe (all runs stacked) ─────────────────────────────────────
results_all: pd.DataFrame = pd.concat(list(results_by_run.values()), ignore_index=True)
print(f"\n✓ shared_setup complete — {len(ALL_RUNS)} runs loaded, {len(results_all):,} total rows")


# After loading, resolve which actual column holds the decile values and expose
# it as `decile_col` so notebook cells can just use `decile_col` directly.
def _resolve_decile_col(df: pd.DataFrame) -> str:
    """Return the first candidate decile column that exists in *df*.

    Args:
        df: Any results DataFrame (or results_all).

    Returns:
        Column name string.

    Raises:
        KeyError: If none of the candidate columns are present.
    """
    for c in _DECILE_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise KeyError(
        f"No decile column found. Tried: {_DECILE_COL_CANDIDATES}. "
        f"Available columns: {list(df.columns)}"
    )


# Re-resolve decile_col against the actual loaded data so it always points to a
# real column (overrides the constant-based value set earlier).
try:
    decile_col = _resolve_decile_col(results_all)
    print(f"  decile_col resolved → '{decile_col}'")
except KeyError as _e:
    print(f"  WARNING: {_e}")


# ═════════════════════════════════════════════════════════════════════════════
# ── Helper functions ──────────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════

def pivot_metric(
    metric_col: str = "evaluation_score",
    *,
    row: str = "backend",
    col: str = "llm",
    evaluator: str = "substring",
    ctx_label: str | None = None,
    agg: str = "mean",
) -> pd.DataFrame:
    """Pivot mean accuracy (or any metric) into a backend × llm table.

    Args:
        metric_col: Column to aggregate (default ``"evaluation_score"``).
        row:        Column to use as pivot rows.
        col:        Column to use as pivot columns.
        evaluator:  Filter to this evaluator key before pivoting.
        ctx_label:  If given, filter to this context label only.
        agg:        Aggregation function name (``"mean"``, ``"sum"``, etc.).

    Returns:
        Pivoted DataFrame.
    """
    df = results_all[results_all["evaluator"] == evaluator].copy()
    if ctx_label is not None:
        df = df[df["ctx_label"] == ctx_label]
    return df.pivot_table(index=row, columns=col, values=metric_col, aggfunc=agg)


def accuracy_by_decile(
    evaluator: str = "substring",
    *,
    llm: str | None = None,
    backend: str | None = None,
    ctx_label: str | None = None,
) -> pd.DataFrame:
    """Return mean accuracy per decile for the given filters.

    Args:
        evaluator:  Evaluator key to filter on.
        llm:        Optional LLM key filter.
        backend:    Optional backend key filter.
        ctx_label:  Optional context label filter.

    Returns:
        DataFrame with columns ``decile``, ``accuracy``, ``count`` and
        any grouping columns that were not fixed by the filters.
    """
    df = results_all[results_all["evaluator"] == evaluator].copy()
    if llm is not None:
        df = df[df["llm"] == llm]
    if backend is not None:
        df = df[df["backend"] == backend]
    if ctx_label is not None:
        df = df[df["ctx_label"] == ctx_label]

    dcol = _resolve_decile_col(df)

    group_cols = [dcol]
    if llm is None:
        group_cols.append("llm")
    if backend is None:
        group_cols.append("backend")
    if ctx_label is None and "ctx_label" in df.columns:
        group_cols.append("ctx_label")

    agg = (
        df.groupby(group_cols)["evaluation_score"]
        .agg(accuracy="mean", count="count")
        .reset_index()
        .rename(columns={dcol: "decile"})
    )
    return agg
