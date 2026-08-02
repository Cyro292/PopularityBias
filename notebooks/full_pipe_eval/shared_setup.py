"""Shared setup for full-pipeline generation evaluation notebooks.

Loads Stage-3 generation/evaluation parquet files and exposes a small set of
helpers for answering: how much does retrieval quality affect generation?
"""

from __future__ import annotations

import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import DATA_DIR
from src.metrics.decile_utils import (
    COL_DECILE_CHUNK_WEIGHTED,
    COL_DECILE_UNWEIGHTED,
    COL_POPULARITY,
    load_boundaries_from_metadata,
)

warnings.filterwarnings("ignore")
load_dotenv()

# === Configuration ===

COLLECTION_NAME = "wiki_full_bil"
OUTPUT_DIR = "all_qa_8k"
DECILE_MODE = "chunk_weighted"  # "chunk_weighted" | "unweighted"
FORCE_RECOMPUTE = False

# Base full-pipeline comparison only. Add to these lists if you run more files.
LLM_KEYS = ["neo", "qwen"]
BACKEND_KEYS = ["zero_shot", "bm25_plus", "ivfpq_high", "ivfpq_extremely_high"]
CTX_LABELS = ["zero", "top1", "top3"]
EVALUATOR_KEYS = ["substring"]

COLLECTION_ROOT = DATA_DIR / COLLECTION_NAME
RESULTS_DIR = COLLECTION_ROOT / OUTPUT_DIR
IMAGES_DIR = Path(__file__).parent / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

metadata_path = COLLECTION_ROOT / "metadata.json"
boundaries_uw, boundaries_cw, corpus_stats = load_boundaries_from_metadata(metadata_path)


# === Run Metadata ===

@dataclass(frozen=True)
class RunEntry:
    """One generation/evaluation parquet file."""

    llm: str
    backend: str
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


def _parse_results_path(path: Path) -> RunEntry | None:
    """Parse results_<llm>_<backend>_<ctx>_<evaluator>.parquet."""
    stem = path.stem
    if not stem.startswith("results_"):
        return None
    rest = stem.removeprefix("results_")
    ctx_match = re.search(r"_(zero|top\d+)_", rest)
    if ctx_match is None:
        return None
    before_ctx = rest[: ctx_match.start()]
    evaluator = rest[ctx_match.end() :]
    if "_" not in before_ctx or not evaluator:
        return None
    llm, backend = before_ctx.split("_", 1)
    return RunEntry(llm=llm, backend=backend, ctx_label=ctx_match.group(1), evaluator=evaluator)


def _configured_entries() -> list[RunEntry]:
    entries: list[RunEntry] = []
    for llm in LLM_KEYS:
        for backend in BACKEND_KEYS:
            for ctx in CTX_LABELS:
                if backend == "zero_shot" and ctx != "zero":
                    continue
                if backend != "zero_shot" and ctx == "zero":
                    continue
                for evaluator in EVALUATOR_KEYS:
                    entries.append(RunEntry(llm, backend, ctx, evaluator))
    return entries


def _discover_entries() -> list[RunEntry]:
    entries = []
    for path in sorted(RESULTS_DIR.glob("results_*.parquet")):
        entry = _parse_results_path(path)
        if entry is None:
            continue
        if entry.backend not in BACKEND_KEYS:
            continue
        entries.append(entry)
    return entries


_entry_by_key = {entry.key: entry for entry in _configured_entries()}
for entry in _discover_entries():
    _entry_by_key.setdefault(entry.key, entry)
ALL_ENTRIES = list(_entry_by_key.values())


# === Normalization ===

def _score_to_float(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    if value is None or pd.isna(value):
        return float("nan")
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "correct", "1"}:
            return 1.0
        if text in {"false", "no", "incorrect", "0"}:
            return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if value is None or pd.isna(value):
        return []
    return [value]


def _same_id(left: Any, right: Any) -> bool:
    try:
        return str(int(left)) == str(int(right))
    except (TypeError, ValueError):
        return str(left) == str(right)


def _normalise_df(df: pd.DataFrame, entry: RunEntry) -> pd.DataFrame:
    df = df.copy()

    if "metadata" in df.columns:
        meta = pd.json_normalize(df["metadata"].where(df["metadata"].notna(), None))
        for col in meta.columns:
            if col not in df.columns:
                df[col] = meta[col].values

    df["llm"] = entry.llm
    df["backend"] = entry.backend
    df["ctx_label"] = entry.ctx_label
    df["evaluator"] = entry.evaluator
    df["run_key"] = entry.key
    df["run_label"] = entry.label

    if "generated_answer" not in df.columns and "proposed_answer" in df.columns:
        df["generated_answer"] = df["proposed_answer"]
    if "proposed_answer" not in df.columns and "generated_answer" in df.columns:
        df["proposed_answer"] = df["generated_answer"]

    df["generation_correct"] = df.get("evaluation_score", pd.Series(index=df.index, dtype=float)).map(_score_to_float)
    df["is_correct"] = df["generation_correct"] >= 0.5

    answers = df.get("generated_answer", pd.Series("", index=df.index)).fillna("").astype(str)
    df["answer_length_chars"] = answers.str.len()
    df["answer_length_tokens"] = answers.str.split().map(len)
    df["has_generated_answer"] = answers.str.strip().ne("")

    if COL_DECILE_UNWEIGHTED not in df.columns and "decile_unweighted" in df.columns:
        df[COL_DECILE_UNWEIGHTED] = df["decile_unweighted"]
    if COL_DECILE_CHUNK_WEIGHTED not in df.columns and "decile_chunk_weighted" in df.columns:
        df[COL_DECILE_CHUNK_WEIGHTED] = df["decile_chunk_weighted"]

    if "retrieved_doc_ids" in df.columns and "wikipedia_id" in df.columns:
        def _hit(row: pd.Series) -> float:
            ids = _as_list(row.get("retrieved_doc_ids"))
            gold = row.get("wikipedia_id")
            if not ids or gold is None or pd.isna(gold):
                return float("nan")
            return float(any(_same_id(item, gold) for item in ids))

        df["retrieval_hit"] = df.apply(_hit, axis=1)
    else:
        df["retrieval_hit"] = np.nan

    df["uses_retrieval"] = entry.backend != "zero_shot"
    return df


# === Load Results ===

results_by_run: dict[str, pd.DataFrame] = {}

print(f"Loading full-pipeline results from: {RESULTS_DIR}")
for entry in ALL_ENTRIES:
    if not entry.results_path.exists():
        continue
    df = pd.read_parquet(entry.results_path)
    df = _normalise_df(df, entry)
    results_by_run[entry.key] = df
    print(f"  ✓ {entry.label}: {len(df):,} rows")

ALL_RUNS = list(results_by_run)
if not ALL_RUNS:
    raise FileNotFoundError(
        f"No base full-pipeline result files found in {RESULTS_DIR}. "
        "Expected names like results_neo_bm25_plus_top3_substring.parquet."
    )

results_all = pd.concat(results_by_run.values(), ignore_index=True)

AVAILABLE_LLMS = sorted(results_all["llm"].dropna().astype(str).unique().tolist())
AVAILABLE_BACKENDS = sorted(results_all["backend"].dropna().astype(str).unique().tolist())
AVAILABLE_CONTEXTS = sorted(results_all["ctx_label"].dropna().astype(str).unique().tolist())
AVAILABLE_EVALUATORS = sorted(results_all["evaluator"].dropna().astype(str).unique().tolist())
GROUP_COL = "dataset" if "dataset" in results_all.columns else ""

print(f"\n✓ shared_setup complete — {len(ALL_RUNS)} runs, {len(results_all):,} rows")
print(f"  llms      : {AVAILABLE_LLMS}")
print(f"  backends  : {AVAILABLE_BACKENDS}")
print(f"  contexts  : {AVAILABLE_CONTEXTS}")
print(f"  evaluators: {AVAILABLE_EVALUATORS}")


# === Labels / Colors ===

_BACKEND_COLORS = {
    "zero_shot": "#6B7280",
    "bm25_plus": "#F59E0B",
    "ivfpq_high": "#3B82F6",
    "ivfpq_extremely_high": "#1D4ED8",
}
_LLM_COLORS = {"neo": "#8B5CF6", "qwen": "#EF4444"}


def backend_color(backend: str) -> str:
    return _BACKEND_COLORS.get(backend, "#64748B")


def llm_color(llm: str) -> str:
    return _LLM_COLORS.get(llm, "#64748B")


def run_label(entry_or_key: RunEntry | str) -> str:
    if isinstance(entry_or_key, RunEntry):
        return entry_or_key.label
    entry = next((item for item in ALL_ENTRIES if item.key == entry_or_key), None)
    return entry.label if entry is not None else str(entry_or_key)


# === Analysis Helpers ===

def decile_col_for_mode(mode: str = DECILE_MODE) -> str:
    candidates = (
        [COL_DECILE_UNWEIGHTED, "decile_unweighted", "decile"]
        if mode == "unweighted"
        else [COL_DECILE_CHUNK_WEIGHTED, "decile_chunk_weighted", "decile"]
    )
    for col in candidates:
        if col in results_all.columns:
            return col
    raise KeyError(f"No decile column found for mode={mode!r}; tried {candidates}")


decile_col = decile_col_for_mode(DECILE_MODE)


def filter_generation_results(
    *,
    evaluator: str | None = None,
    llm: str | None = None,
    backend: str | None = None,
    ctx_label: str | None = None,
    dataset: str | None = None,
) -> pd.DataFrame:
    df = results_all.copy()
    if evaluator is not None:
        df = df[df["evaluator"] == evaluator]
    if llm is not None:
        df = df[df["llm"] == llm]
    if backend is not None:
        df = df[df["backend"] == backend]
    if ctx_label is not None:
        df = df[df["ctx_label"] == ctx_label]
    if dataset is not None and GROUP_COL:
        df = df[df[GROUP_COL].astype(str) == str(dataset)]
    return df.copy()


def summarise_generation(
    df: pd.DataFrame | None = None,
    *,
    group_cols: tuple[str, ...] | list[str] = ("llm", "backend", "ctx_label"),
    metric_col: str = "generation_correct",
) -> pd.DataFrame:
    source = results_all if df is None else df
    group_cols = [col for col in group_cols if col in source.columns]
    if not group_cols:
        raise ValueError("No valid group columns supplied")
    out = (
        source.dropna(subset=[metric_col])
        .groupby(group_cols)[metric_col]
        .agg(accuracy="mean", count="count", std="std")
        .reset_index()
    )
    out["se"] = out["std"].fillna(0.0) / np.sqrt(out["count"].clip(lower=1))
    out["ci95"] = 1.96 * out["se"]
    return out.drop(columns=["std"])


def generation_by_decile(
    *,
    mode: str = DECILE_MODE,
    evaluator: str | None = None,
    extra_group_cols: tuple[str, ...] | list[str] = ("llm", "backend", "ctx_label"),
) -> pd.DataFrame:
    df = filter_generation_results(evaluator=evaluator)
    dcol = decile_col_for_mode(mode)
    df = df.copy()
    df["decile"] = pd.to_numeric(df[dcol], errors="coerce").astype("Int64")
    out = summarise_generation(df, group_cols=("decile", *extra_group_cols))
    out["decile_label"] = out["decile"].astype(int) + 1
    return out.sort_values([col for col in [*extra_group_cols, "decile"] if col in out.columns])


def retrieval_generation_effect(
    *,
    evaluator: str | None = None,
    group_cols: tuple[str, ...] | list[str] = ("llm", "backend", "ctx_label"),
) -> pd.DataFrame:
    """Compare generation accuracy when retrieval hits vs misses the gold page."""
    df = filter_generation_results(evaluator=evaluator)
    df = df[df["retrieval_hit"].notna()].copy()
    grouped = summarise_generation(df, group_cols=(*group_cols, "retrieval_hit"))
    if grouped.empty:
        return grouped
    pivot = grouped.pivot_table(index=list(group_cols), columns="retrieval_hit", values="accuracy")
    pivot = pivot.rename(columns={0.0: "accuracy_when_retrieval_misses", 1.0: "accuracy_when_retrieval_hits"}).reset_index()
    pivot["retrieval_hit_lift"] = pivot.get("accuracy_when_retrieval_hits", np.nan) - pivot.get("accuracy_when_retrieval_misses", np.nan)
    counts = grouped.pivot_table(index=list(group_cols), columns="retrieval_hit", values="count")
    counts = counts.rename(columns={0.0: "n_retrieval_misses", 1.0: "n_retrieval_hits"}).reset_index()
    return pivot.merge(counts, on=list(group_cols), how="left")


def retrieval_generation_correlation(
    *,
    evaluator: str | None = None,
    mode: str = DECILE_MODE,
    group_cols: tuple[str, ...] | list[str] = ("llm", "backend", "ctx_label"),
) -> pd.DataFrame:
    """Correlate per-decile retrieval hit rate with generation accuracy."""
    df = filter_generation_results(evaluator=evaluator)
    df = df[df["retrieval_hit"].notna()].copy()
    if df.empty:
        return pd.DataFrame()
    dcol = decile_col_for_mode(mode)
    df["decile"] = pd.to_numeric(df[dcol], errors="coerce").astype("Int64")
    per_decile = (
        df.dropna(subset=["decile", "generation_correct", "retrieval_hit"])
        .groupby([*group_cols, "decile"])
        .agg(
            generation_accuracy=("generation_correct", "mean"),
            retrieval_hit_rate=("retrieval_hit", "mean"),
            count=("generation_correct", "count"),
        )
        .reset_index()
    )
    rows = []
    for keys, sub in per_decile.groupby(list(group_cols)):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        if len(sub) >= 2:
            row["pearson_r"] = sub["retrieval_hit_rate"].corr(sub["generation_accuracy"], method="pearson")
            row["spearman_r"] = sub["retrieval_hit_rate"].corr(sub["generation_accuracy"], method="spearman")
        else:
            row["pearson_r"] = np.nan
            row["spearman_r"] = np.nan
        row["n_deciles"] = len(sub)
        rows.append(row)
    return pd.DataFrame(rows)


def pivot_metric(
    metric_col: str = "generation_correct",
    *,
    row: str = "backend",
    col: str = "llm",
    evaluator: str | None = None,
    ctx_label: str | None = None,
) -> pd.DataFrame:
    df = filter_generation_results(evaluator=evaluator, ctx_label=ctx_label)
    return df.pivot_table(index=row, columns=col, values=metric_col, aggfunc="mean")


def accuracy_by_decile(
    evaluator: str | None = None,
    *,
    llm: str | None = None,
    backend: str | None = None,
    ctx_label: str | None = None,
) -> pd.DataFrame:
    df = filter_generation_results(evaluator=evaluator, llm=llm, backend=backend, ctx_label=ctx_label)
    dcol = decile_col_for_mode(DECILE_MODE)
    df = df.copy()
    df["decile"] = pd.to_numeric(df[dcol], errors="coerce").astype("Int64")
    return summarise_generation(df, group_cols=("decile",))
