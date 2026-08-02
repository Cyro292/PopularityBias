"""Plot retrieval metrics by popularity decile for BM25+ and FAISS high.

The script reads retrieval checkpoint CSVs and their matching QA metadata,
then plots one Recall@k or MRR curve per source dataset. It can render both
methods as a two-panel figure or either method alone, and can also plot the
FAISS-high difference relative to BM25+.
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

METHODS: dict[str, tuple[str, str, str]] = {
    "bm25": ("retrieved_docs_bm25_plus.csv", "BM25+", "#F59E0B"),
    "faiss-high": ("retrieved_docs_ivfpq_high.csv", "FAISS high", "#3B82F6"),
}

# Colors are colorblind-safe; markers and line styles remain distinguishable in grayscale.
DATASET_STYLES = [
    ("#0072B2", "o", "-"),
    ("#E69F00", "s", "--"),
    ("#CC79A7", "D", "-."),
    ("#56B4E9", "^", ":"),
    ("#332288", "P", (0, (3, 1, 1, 1))),
    ("#666666", "X", (0, (5, 1))),
]


def _normalise_id(value: object) -> str:
    """Return a stable string representation of a Wikipedia identifier."""
    value_str = str(value).strip()
    return value_str[:-2] if value_str.endswith(".0") else value_str


def compute_metric_by_dataset_and_decile(
    results_path: Path,
    qa_path: Path,
    *,
    excluded_datasets: set[str],
    metric: Literal["recall", "mrr"],
    k: int,
    min_questions_per_point: int,
) -> pd.DataFrame:
    """Compute article-level Recall@k or MRR for every dataset and decile.

    Args:
        results_path: Retrieval checkpoint CSV with one row per retrieved chunk.
        qa_path: QA metadata parquet with target article and decile columns.
        excluded_datasets: Dataset names to omit from the figure.
        metric: ``"recall"`` for Recall@k or ``"mrr"`` for reciprocal rank.
        k: Number of ranked chunks inspected per question.
        min_questions_per_point: Minimum questions required for a plotted estimate.

    Returns:
        A DataFrame with dataset, decile, metric, confidence interval, and count.

    Raises:
        FileNotFoundError: If either input file is absent.
        KeyError: If required input columns are absent.
    """
    if not results_path.exists():
        raise FileNotFoundError(f"Results checkpoint not found: {results_path}")
    if not qa_path.exists():
        raise FileNotFoundError(f"QA metadata not found: {qa_path}")

    qa = pd.read_parquet(qa_path)
    raw = pd.read_csv(results_path)
    qa_required = {"question_id", "wikipedia_id", "dataset", "pop_decile_chunk_weighted"}
    raw_required = {"question_id", "doc_rank", "metadata_wikipedia_id"}
    missing_qa = qa_required - set(qa.columns)
    missing_raw = raw_required - set(raw.columns)
    if missing_qa:
        raise KeyError(f"QA metadata missing columns: {sorted(missing_qa)}")
    if missing_raw:
        raise KeyError(f"Results checkpoint missing columns: {sorted(missing_raw)}")

    qa = qa.loc[
        ~qa["dataset"].isin(excluded_datasets)
        & qa["pop_decile_chunk_weighted"].between(0, 9),
        ["question_id", "wikipedia_id", "dataset", "pop_decile_chunk_weighted"],
    ].copy()
    qa["question_id"] = qa["question_id"].astype(str)
    qa["target_id"] = qa["wikipedia_id"].map(_normalise_id)

    raw = raw[["question_id", "doc_rank", "metadata_wikipedia_id"]].copy()
    raw["question_id"] = raw["question_id"].astype(str)
    raw["retrieved_id"] = raw["metadata_wikipedia_id"].map(_normalise_id)
    merged = raw.merge(qa[["question_id", "target_id"]], on="question_id", how="inner")

    target_ranks = (
        merged.loc[merged["retrieved_id"] == merged["target_id"]]
        .groupby("question_id", as_index=False)["doc_rank"]
        .min()
        .rename(columns={"doc_rank": "target_rank"})
    )
    question_metrics = qa.merge(target_ranks, on="question_id", how="left")
    if metric == "recall":
        question_metrics["metric_value"] = (
            question_metrics["target_rank"].notna() & (question_metrics["target_rank"] < k)
        ).astype(float)
    else:
        question_metrics["metric_value"] = 1.0 / (question_metrics["target_rank"] + 1)
        question_metrics["metric_value"] = question_metrics["metric_value"].fillna(0.0)

    summary = question_metrics.groupby(
        ["dataset", "pop_decile_chunk_weighted"], as_index=False
    ).agg(
        metric_value=("metric_value", "mean"),
        n_questions=("metric_value", "size"),
        metric_sem=("metric_value", "sem"),
    ).rename(
        columns={"pop_decile_chunk_weighted": "decile"}
    )
    if metric == "recall":
        summary["metric_ci95"] = 1.96 * np.sqrt(
            summary["metric_value"] * (1.0 - summary["metric_value"]) / summary["n_questions"]
        )
    else:
        summary["metric_ci95"] = 1.96 * summary["metric_sem"].fillna(0.0)
    return summary[summary["n_questions"] >= min_questions_per_point].copy()


def plot_recall_by_dataset_and_decile(
    results_dir: Path,
    qa_path: Path,
    output_path: Path,
    *,
    cohort_label: str,
    metric: Literal["recall", "mrr"],
    panel: Literal["both", "bm25", "faiss-high"],
    excluded_datasets: set[str],
    ci_style: Literal["band", "bars"],
    k: int,
    min_questions_per_point: int,
) -> None:
    """Render selected retrieval-method panels with one curve per dataset.

    Args:
        results_dir: Directory containing the retrieval checkpoint CSVs.
        qa_path: QA metadata parquet corresponding to the checkpoint questions.
        output_path: PNG destination.
        cohort_label: Human-readable evaluation cohort label for the figure title.
        metric: ``"recall"`` for Recall@k or ``"mrr"`` for reciprocal rank.
        panel: ``"both"`` for two panels, otherwise the method to show.
        excluded_datasets: Dataset names to omit.
        ci_style: Render confidence intervals as shaded bands or error bars.
        k: Recall cutoff.
        min_questions_per_point: Minimum questions required for a plotted estimate.
    """
    selected_methods = list(METHODS) if panel == "both" else [panel]
    summaries = {
        method: compute_metric_by_dataset_and_decile(
            results_dir / METHODS[method][0],
            qa_path,
            excluded_datasets=excluded_datasets,
            metric=metric,
            k=k,
            min_questions_per_point=min_questions_per_point,
        )
        for method in selected_methods
    }
    datasets = sorted(set().union(*(summary["dataset"].unique() for summary in summaries.values())))
    dataset_styles = {
        dataset: DATASET_STYLES[index % len(DATASET_STYLES)]
        for index, dataset in enumerate(datasets)
    }

    fig, axes = plt.subplots(1, len(selected_methods), figsize=(7 * len(selected_methods), 4.8), sharey=True)
    if len(selected_methods) == 1:
        axes = [axes]

    for panel_label, method, ax in zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", selected_methods, axes):
        summary = summaries[method]
        _, method_label, _ = METHODS[method]
        for dataset in datasets:
            subset = summary[summary["dataset"] == dataset].sort_values("decile")
            if subset.empty:
                continue
            color, marker, linestyle = dataset_styles[dataset]
            x_values = subset["decile"] + 1
            y_values = subset["metric_value"] * 100
            ci_values = subset["metric_ci95"] * 100
            if ci_style == "band":
                ax.plot(x_values, y_values, marker=marker, markersize=4, linewidth=1.8,
                        linestyle=linestyle, color=color, label=dataset.replace("_", " ").title())
                ax.fill_between(x_values, y_values - ci_values, y_values + ci_values, color=color, alpha=0.14)
            else:
                ax.errorbar(x_values, y_values, yerr=ci_values, marker=marker, markersize=4,
                            linewidth=1.8, linestyle=linestyle, color=color, capsize=2,
                            label=dataset.replace("_", " ").title())

        ax.set_title(f"{panel_label}. {method_label}", fontweight="bold")
        ax.set_xlabel("Popularity Decile (1=Rare to 10=Famous)", fontweight="bold")
        ax.set_xticks(range(1, 11))
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    metric_label = f"Recall@{k}" if metric == "recall" else "MRR"
    axes[0].set_ylabel(f"{metric_label} (%)", fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Dataset", loc="lower center", ncol=3, fontsize=8, title_fontsize=9)
    fig.suptitle(
        f"{metric_label} by Dataset and Chunk-Weighted Popularity Decile\n{cohort_label}",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info("Saved figure to %s", output_path)


def plot_delta_vs_bm25(
    results_dir: Path,
    qa_path: Path,
    output_path: Path,
    *,
    cohort_label: str,
    metric: Literal["recall", "mrr"],
    excluded_datasets: set[str],
    ci_style: Literal["band", "bars"],
    k: int,
    min_questions_per_point: int,
) -> None:
    """Render FAISS-high minus BM25+ Recall@k or MRR per dataset and decile.

    Args:
        results_dir: Directory containing the BM25+ and FAISS-high checkpoints.
        qa_path: QA metadata parquet corresponding to the checkpoint questions.
        output_path: PNG destination.
        cohort_label: Human-readable evaluation cohort label for the figure title.
        metric: ``"recall"`` for Recall@k or ``"mrr"`` for reciprocal rank.
        excluded_datasets: Dataset names to omit.
        ci_style: Render confidence intervals as shaded bands or error bars.
        k: Recall cutoff.
        min_questions_per_point: Minimum questions required for a plotted estimate.
    """
    bm25 = compute_metric_by_dataset_and_decile(
        results_dir / METHODS["bm25"][0],
        qa_path,
        excluded_datasets=excluded_datasets,
        metric=metric,
        k=k,
        min_questions_per_point=min_questions_per_point,
    ).rename(columns={"metric_value": "bm25_metric", "metric_ci95": "bm25_ci95"})
    faiss = compute_metric_by_dataset_and_decile(
        results_dir / METHODS["faiss-high"][0],
        qa_path,
        excluded_datasets=excluded_datasets,
        metric=metric,
        k=k,
        min_questions_per_point=min_questions_per_point,
    ).rename(columns={"metric_value": "faiss_metric", "metric_ci95": "faiss_ci95"})
    delta = bm25.merge(
        faiss[["dataset", "decile", "faiss_metric", "faiss_ci95"]],
        on=["dataset", "decile"],
    )
    delta["metric_delta"] = (delta["faiss_metric"] - delta["bm25_metric"]) * 100
    delta["metric_delta_ci95"] = np.sqrt(
        delta["bm25_ci95"] ** 2 + delta["faiss_ci95"] ** 2
    ) * 100

    datasets = sorted(delta["dataset"].unique())
    dataset_styles = {
        dataset: DATASET_STYLES[index % len(DATASET_STYLES)]
        for index, dataset in enumerate(datasets)
    }
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.axhline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.7)
    for dataset in datasets:
        subset = delta[delta["dataset"] == dataset].sort_values("decile")
        color, marker, linestyle = dataset_styles[dataset]
        x_values = subset["decile"] + 1
        y_values = subset["metric_delta"]
        ci_values = subset["metric_delta_ci95"]
        if ci_style == "band":
            ax.plot(x_values, y_values, marker=marker, markersize=4, linewidth=1.8,
                    linestyle=linestyle, color=color, label=dataset.replace("_", " ").title())
            ax.fill_between(x_values, y_values - ci_values, y_values + ci_values, color=color, alpha=0.14)
        else:
            ax.errorbar(x_values, y_values, yerr=ci_values, marker=marker, markersize=4,
                        linewidth=1.8, linestyle=linestyle, color=color, capsize=2,
                        label=dataset.replace("_", " ").title())

    ax.set_xlabel("Popularity Decile (1=Rare to 10=Famous)", fontweight="bold")
    metric_label = f"Recall@{k}" if metric == "recall" else "MRR"
    ax.set_ylabel(f"FAISS High minus BM25+ {metric_label} (pp)", fontweight="bold")
    ax.set_title(
        f"FAISS High {metric_label} Advantage over BM25+ by Dataset\n{cohort_label}",
        fontweight="bold",
    )
    ax.set_xticks(range(1, 11))
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, title="Dataset", loc="lower center", ncol=3, fontsize=8, title_fontsize=9)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info("Saved figure to %s", output_path)


def main() -> None:
    """Parse arguments and generate the requested retrieval-metric figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DATA_DIR / "wiki_full_bil" / "all_qa_8k",
        help="Directory containing retrieved_docs_bm25_plus.csv and retrieved_docs_ivfpq_high.csv.",
    )
    parser.add_argument(
        "--qa-path",
        type=Path,
        default=None,
        help="QA metadata parquet. Defaults to cyro_qa_cache.parquet in --results-dir.",
    )
    parser.add_argument("--figure", choices=["metric", "delta"], default="metric")
    parser.add_argument("--panel", choices=["both", *METHODS], default="both")
    parser.add_argument("--metric", choices=["recall", "mrr"], default="recall")
    parser.add_argument("--ci-style", choices=["band", "bars"], default="bars")
    parser.add_argument("--cohort-label", default="8k Question Pool")
    parser.add_argument("--k", type=int, default=5, help="Recall cutoff (default: 5).")
    parser.add_argument(
        "--min-questions-per-point",
        type=int,
        default=40,
        help="Minimum questions required for a dataset-decile point (default: 40).",
    )
    parser.add_argument(
        "--exclude-datasets",
        nargs="*",
        default=["hotpot_qa"],
        help="Dataset names to omit (default: hotpot_qa).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    if args.k <= 0 or args.min_questions_per_point <= 0:
        raise ValueError("--k and --min-questions-per-point must be positive")

    qa_path = args.qa_path or args.results_dir / "cyro_qa_cache.parquet"
    output_path = args.output_path
    if output_path is None:
        filename = (
            "delta_vs_bm25_retrieved_docs_ivfpq_high.png"
            if args.figure == "delta"
            else f"{args.metric}{f'_at_{args.k}' if args.metric == 'recall' else ''}_by_dataset_and_decile_all_qa_8k.png"
        )
        output_path = ROOT_DIR / "paper_figures" / filename

    if args.figure == "delta":
        plot_delta_vs_bm25(
            args.results_dir,
            qa_path,
            output_path,
            cohort_label=args.cohort_label,
            metric=args.metric,
            excluded_datasets=set(args.exclude_datasets),
            ci_style=args.ci_style,
            k=args.k,
            min_questions_per_point=args.min_questions_per_point,
        )
        return

    plot_recall_by_dataset_and_decile(
        args.results_dir,
        qa_path,
        output_path,
        cohort_label=args.cohort_label,
        metric=args.metric,
        panel=args.panel,
        excluded_datasets=set(args.exclude_datasets),
        ci_style=args.ci_style,
        k=args.k,
        min_questions_per_point=args.min_questions_per_point,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
