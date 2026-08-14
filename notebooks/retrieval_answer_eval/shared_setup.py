"""Common setup for answer-containment retrieval evaluation notebooks."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import DATA_DIR


@dataclass(frozen=True)
class BackendEntry:
    """Describe one answer-evaluation result to load."""

    key: str
    label: str
    color: str


COLLECTION_ROOT = DATA_DIR / "wiki_full_bil"
RESULTS_DIR = COLLECTION_ROOT / "all_qa_8k"
IMAGES_DIR = RESULTS_DIR / "answer_eval_results"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

BACKENDS = [
    BackendEntry("bm25_plus", "BM25+", "#E69F00"),
    BackendEntry("ivfpq_high", "FAISS high", "#0072B2"),
    BackendEntry("faiss_hybrid", "FAISS hybrid", "#009E73"),
]
TOP_K = 5
K_VALUES_DETAILED = [1, 3, 5, 10]
EXCLUDED_DATASETS = ["hotpot_qa", "trex"]
DECILE_CANDIDATES = [
    "pop_decile_chunk_weighted",
    "decile_chunk_weighted",
    "decile",
]


def strategy_label(key: str) -> str:
    """Return the display label for a backend key."""
    return next((entry.label for entry in BACKENDS if entry.key == key), key)


def strategy_color(key: str) -> str:
    """Return the plot color for a backend key."""
    return next((entry.color for entry in BACKENDS if entry.key == key), "#666666")


def _load_backend(entry: BackendEntry) -> tuple[pd.DataFrame, dict[str, int]]:
    path = RESULTS_DIR / f"retrieval_answer_eval_{entry.key}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run retrieval_answer_eval_runner before these notebooks."
        )
    frame = pd.read_parquet(path)
    required = {
        "question_id",
        "is_evaluable",
        "answer_rank",
        "answer_reciprocal_rank",
        "dataset",
        "popularity_avg",
    }
    required.update(f"answer_recall@{k}" for k in K_VALUES_DETAILED)
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"{path.name} is missing columns: {sorted(missing)}")

    decile_col = next((column for column in DECILE_CANDIDATES if column in frame), None)
    if decile_col is None:
        raise KeyError(f"{path.name} has no supported popularity-decile column")

    counts = {
        "total": len(frame),
        "non_evaluable": int((~frame["is_evaluable"].fillna(False)).sum()),
    }
    frame = frame.loc[frame["is_evaluable"].fillna(False)].copy()
    counts["excluded_dataset"] = int(frame["dataset"].isin(EXCLUDED_DATASETS).sum())
    frame = frame.loc[~frame["dataset"].isin(EXCLUDED_DATASETS)].copy()
    frame["question_id"] = frame["question_id"].astype(str)
    frame["decile"] = pd.to_numeric(frame[decile_col], errors="coerce")
    frame["popularity_avg"] = pd.to_numeric(frame["popularity_avg"], errors="coerce")
    frame = frame.dropna(subset=["decile", "popularity_avg"])

    frame["rank"] = pd.to_numeric(frame["answer_rank"], errors="coerce")
    frame["reciprocal_rank"] = pd.to_numeric(
        frame["answer_reciprocal_rank"], errors="coerce"
    )
    for k in K_VALUES_DETAILED:
        frame[f"recall@{k}"] = pd.to_numeric(
            frame[f"answer_recall@{k}"], errors="coerce"
        )
    counts["analysed"] = len(frame)
    return frame, counts


results_by_strategy: dict[str, pd.DataFrame] = {}
cohort_rows: list[dict[str, object]] = []
for backend in BACKENDS:
    result_path = RESULTS_DIR / f"retrieval_answer_eval_{backend.key}.parquet"
    if not result_path.exists():
        continue
    result, counts = _load_backend(backend)
    results_by_strategy[backend.key] = result
    cohort_rows.append({"backend": backend.label, **counts})

if not results_by_strategy:
    raise FileNotFoundError(
        f"No retrieval_answer_eval_*.parquet files found in {RESULTS_DIR}"
    )

ALL_STRATEGIES = list(results_by_strategy)
cohort_summary = pd.DataFrame(cohort_rows)
question_sets = {
    key: set(frame["question_id"]) for key, frame in results_by_strategy.items()
}
reference_questions = question_sets[ALL_STRATEGIES[0]]
cohorts_match = all(values == reference_questions for values in question_sets.values())

metrics_rows: list[dict[str, object]] = []
decile_rows: list[dict[str, object]] = []
for key, frame in results_by_strategy.items():
    metric_row: dict[str, object] = {
        "backend": strategy_label(key),
        "n_questions": len(frame),
        "answer_mrr": frame["reciprocal_rank"].mean(),
    }
    for k in K_VALUES_DETAILED:
        metric_row[f"answer_recall@{k}"] = frame[f"recall@{k}"].mean()
    metrics_rows.append(metric_row)

    grouped = frame.groupby("decile", as_index=False).agg(
        n_questions=("question_id", "size"),
        answer_mrr=("reciprocal_rank", "mean"),
        answer_mrr_sem=("reciprocal_rank", "sem"),
        **{
            f"answer_recall_at_{k}": (f"recall@{k}", "mean")
            for k in K_VALUES_DETAILED
        },
    )
    grouped["backend_key"] = key
    grouped["backend"] = strategy_label(key)
    decile_rows.extend(grouped.to_dict(orient="records"))

metrics_by_strategy = pd.DataFrame(metrics_rows).set_index("backend")
decile_metrics = pd.DataFrame(decile_rows)
metrics_by_strategy.to_csv(IMAGES_DIR / "answer_metrics_comparison.csv")

sns.set_theme(style="whitegrid", context="notebook")
