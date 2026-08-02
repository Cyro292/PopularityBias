"""Generate concise paper figures for answer-containment retrieval quality.

The script consumes ``retrieval_answer_eval_<backend>.parquet`` outputs from
``retrieval_answer_eval_runner``. It creates a per-dataset popularity-decile
comparison and a paired FAISS-high minus BM25+ delta figure.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import DATA_DIR

logger = logging.getLogger(__name__)

BACKENDS: dict[str, tuple[str, str]] = {
    "bm25_plus": ("BM25+", "#F59E0B"),
    "ivfpq_high": ("FAISS high", "#3B82F6"),
}
DATASET_STYLES = [
    ("#0072B2", "o", "-"),
    ("#E69F00", "s", "--"),
    ("#CC79A7", "D", "-."),
    ("#56B4E9", "^", ":"),
    ("#332288", "P", (0, (3, 1, 1, 1))),
    ("#666666", "X", (0, (5, 1))),
]


def _metric_column(metric: Literal["recall", "mrr"], k: int) -> str:
    return f"answer_recall@{k}" if metric == "recall" else "answer_reciprocal_rank"


def _decile_column(frame: pd.DataFrame, requested: str | None) -> str:
    candidates = [
        requested,
        "pop_decile_chunk_weighted",
        "decile_chunk_weighted",
        "decile",
    ]
    for candidate in candidates:
        if candidate and candidate in frame.columns:
            return candidate
    raise KeyError("No popularity-decile column found in answer evaluation output")


def load_answer_results(
    results_dir: Path,
    backend_key: str,
    *,
    metric: Literal["recall", "mrr"],
    k: int,
    decile_col: str | None,
    excluded_datasets: set[str],
) -> pd.DataFrame:
    """Load one backend's evaluable answer-retrieval rows.

    Args:
        results_dir: Directory containing answer evaluation parquets.
        backend_key: Retrieval backend key.
        metric: Answer Recall@k or answer MRR.
        k: Recall cutoff when ``metric`` is ``"recall"``.
        decile_col: Preferred popularity-decile column.
        excluded_datasets: Dataset names to omit.

    Returns:
        Normalized per-question rows with ``decile`` and ``metric_value``.
    """
    path = results_dir / f"retrieval_answer_eval_{backend_key}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Answer evaluation not found: {path}. Run "
            "python -m src.process.pipeline.retrieval_answer_eval_runner first."
        )
    frame = pd.read_parquet(path)
    metric_col = _metric_column(metric, k)
    if metric_col not in frame.columns:
        raise KeyError(f"{path.name} does not contain {metric_col!r}")
    if "dataset" not in frame.columns:
        raise KeyError(f"{path.name} does not contain 'dataset'")

    source_decile_col = _decile_column(frame, decile_col)
    frame = frame.loc[frame["is_evaluable"].fillna(False)].copy()
    frame = frame[~frame["dataset"].isin(excluded_datasets)]
    frame["decile"] = pd.to_numeric(frame[source_decile_col], errors="coerce")
    frame["metric_value"] = pd.to_numeric(frame[metric_col], errors="coerce")
    return frame.dropna(subset=["decile", "metric_value"])


def summarize_by_dataset_and_decile(
    frame: pd.DataFrame,
    *,
    min_questions_per_point: int,
) -> pd.DataFrame:
    """Aggregate per-question metric values with normal 95% intervals."""
    summary = frame.groupby(["dataset", "decile"], as_index=False).agg(
        metric_value=("metric_value", "mean"),
        metric_sem=("metric_value", "sem"),
        n_questions=("metric_value", "size"),
    )
    summary["metric_ci95"] = 1.96 * summary["metric_sem"].fillna(0.0)
    return summary[summary["n_questions"] >= min_questions_per_point].copy()


def plot_metric_panels(
    results: dict[str, pd.DataFrame],
    output_path: Path,
    *,
    metric: Literal["recall", "mrr"],
    k: int,
    cohort_label: str,
    min_questions_per_point: int,
) -> None:
    """Plot one per-dataset popularity curve panel per retrieval backend."""
    summaries = {
        key: summarize_by_dataset_and_decile(
            frame,
            min_questions_per_point=min_questions_per_point,
        )
        for key, frame in results.items()
    }
    datasets = sorted(
        set().union(*(summary["dataset"].unique() for summary in summaries.values()))
    )
    styles = {
        dataset: DATASET_STYLES[index % len(DATASET_STYLES)]
        for index, dataset in enumerate(datasets)
    }

    fig, axes = plt.subplots(
        1,
        len(results),
        figsize=(6.5 * len(results), 4.6),
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    for panel_label, (backend_key, summary), ax in zip(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ", summaries.items(), axes
    ):
        backend_label = BACKENDS.get(backend_key, (backend_key, "#333333"))[0]
        for dataset in datasets:
            subset = summary[summary["dataset"] == dataset].sort_values("decile")
            if subset.empty:
                continue
            color, marker, linestyle = styles[dataset]
            ax.errorbar(
                subset["decile"] + 1,
                subset["metric_value"] * 100,
                yerr=subset["metric_ci95"] * 100,
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.7,
                markersize=4,
                capsize=2,
                label=dataset.replace("_", " ").title(),
            )
        ax.set_title(f"{panel_label}. {backend_label}", fontweight="bold")
        ax.set_xlabel("Popularity Decile (1=Rare to 10=Famous)", fontweight="bold")
        ax.set_xticks(range(1, 11))
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    metric_label = f"Answer Recall@{k}" if metric == "recall" else "Answer MRR"
    axes[0].set_ylabel(f"{metric_label} (%)", fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        title="Dataset",
        fontsize=8,
    )
    fig.suptitle(
        f"{metric_label} by Dataset and Popularity\n{cohort_label}",
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)


def plot_paired_delta(
    baseline: pd.DataFrame,
    comparison: pd.DataFrame,
    output_path: Path,
    *,
    metric: Literal["recall", "mrr"],
    k: int,
    cohort_label: str,
    min_questions_per_point: int,
) -> None:
    """Plot paired comparison-minus-baseline metric differences."""
    paired = baseline[["question_id", "dataset", "decile", "metric_value"]].merge(
        comparison[["question_id", "metric_value"]],
        on="question_id",
        suffixes=("_baseline", "_comparison"),
        validate="one_to_one",
    )
    paired["metric_delta"] = (
        paired["metric_value_comparison"] - paired["metric_value_baseline"]
    )
    summary = paired.groupby(["dataset", "decile"], as_index=False).agg(
        metric_delta=("metric_delta", "mean"),
        metric_sem=("metric_delta", "sem"),
        n_questions=("metric_delta", "size"),
    )
    summary = summary[summary["n_questions"] >= min_questions_per_point].copy()
    summary["metric_ci95"] = 1.96 * summary["metric_sem"].fillna(0.0)

    datasets = sorted(summary["dataset"].unique())
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.axhline(0, color="black", linewidth=1.1, linestyle="--", alpha=0.7)
    for index, dataset in enumerate(datasets):
        subset = summary[summary["dataset"] == dataset].sort_values("decile")
        color, marker, linestyle = DATASET_STYLES[index % len(DATASET_STYLES)]
        ax.errorbar(
            subset["decile"] + 1,
            subset["metric_delta"] * 100,
            yerr=subset["metric_ci95"] * 100,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.7,
            markersize=4,
            capsize=2,
            label=dataset.replace("_", " ").title(),
        )
    metric_label = f"Answer Recall@{k}" if metric == "recall" else "Answer MRR"
    ax.set_xlabel("Popularity Decile (1=Rare to 10=Famous)", fontweight="bold")
    ax.set_ylabel(
        f"FAISS high minus BM25+ {metric_label} (pp)",
        fontweight="bold",
    )
    ax.set_title(
        f"Paired Answer-Retrieval Difference\n{cohort_label}",
        fontweight="bold",
    )
    ax.set_xticks(range(1, 11))
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        title="Dataset",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and generate answer-retrieval paper figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DATA_DIR / "wiki_full_bil" / "all_qa_8k",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "paper_figures",
    )
    parser.add_argument("--backends", nargs="+", default=list(BACKENDS))
    parser.add_argument("--metric", choices=["recall", "mrr"], default="mrr")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--decile-col", default=None)
    parser.add_argument("--cohort-label", default="8k Question Pool")
    parser.add_argument("--exclude-datasets", nargs="*", default=["hotpot_qa"])
    parser.add_argument("--min-questions-per-point", type=int, default=40)
    parser.add_argument(
        "--figure",
        choices=["all", "metric", "delta"],
        default="all",
    )
    args = parser.parse_args(argv)
    if args.k <= 0 or args.min_questions_per_point <= 0:
        raise ValueError("--k and --min-questions-per-point must be positive")

    results = {
        backend_key: load_answer_results(
            args.results_dir,
            backend_key,
            metric=args.metric,
            k=args.k,
            decile_col=args.decile_col,
            excluded_datasets=set(args.exclude_datasets),
        )
        for backend_key in args.backends
    }
    metric_slug = f"recall_at_{args.k}" if args.metric == "recall" else "mrr"
    if args.figure in {"all", "metric"}:
        plot_metric_panels(
            results,
            args.output_dir / f"answer_{metric_slug}_by_dataset_and_decile.png",
            metric=args.metric,
            k=args.k,
            cohort_label=args.cohort_label,
            min_questions_per_point=args.min_questions_per_point,
        )
    if args.figure in {"all", "delta"}:
        required = {"bm25_plus", "ivfpq_high"}
        if not required.issubset(results):
            raise ValueError("Delta figure requires --backends bm25_plus ivfpq_high")
        plot_paired_delta(
            results["bm25_plus"],
            results["ivfpq_high"],
            args.output_dir
            / f"answer_{metric_slug}_delta_faiss_high_vs_bm25_plus.png",
            metric=args.metric,
            k=args.k,
            cohort_label=args.cohort_label,
            min_questions_per_point=args.min_questions_per_point,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
