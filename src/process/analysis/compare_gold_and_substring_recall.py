"""Compare gold-document and answer-substring retrieval recall by popularity."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import DATA_DIR
from src.process.pipeline.retrieval_answer_eval_runner import (
    find_answer_rank,
    parse_answer_texts,
)

logger = logging.getLogger(__name__)

BACKENDS = {
    "bm25_plus": ("BM25+", "#D97706", "o"),
    "ivfpq_high": ("FAISS high", "#2563EB", "s"),
}
DEFAULT_EXCLUSIONS = ("fever", "hotpot_qa", "trex", "trivia_qa")


def _normalise_id(value: object) -> str:
    """Return a stable string representation of a Wikipedia identifier."""
    value_str = str(value).strip()
    return value_str[:-2] if value_str.endswith(".0") else value_str


def evaluate_backend(
    retrieval_path: Path,
    questions: pd.DataFrame,
    *,
    k_values: Sequence[int],
) -> pd.DataFrame:
    """Calculate both recall definitions for one backend.

    Args:
        retrieval_path: Retrieval checkpoint CSV with ranked chunks.
        questions: Unique question rows with target IDs and answer aliases.
        k_values: Retrieval depths to evaluate.

    Returns:
        One row per question with gold-document and substring Recall@k.
    """
    max_k = max(k_values)
    retrieved = pd.read_csv(
        retrieval_path,
        usecols=[
            "question_id",
            "doc_rank",
            "page_content",
            "metadata_wikipedia_id",
        ],
        dtype={"question_id": str},
    )
    retrieved = retrieved.loc[retrieved["doc_rank"] < max_k].copy()
    retrieved["retrieved_id"] = retrieved["metadata_wikipedia_id"].map(_normalise_id)
    grouped = {
        question_id: group.sort_values("doc_rank")
        for question_id, group in retrieved.groupby("question_id", sort=False)
    }

    rows: list[dict[str, object]] = []
    for question in questions.itertuples(index=False):
        chunks = grouped.get(question.question_id)
        if chunks is None:
            gold_rank = None
            substring_rank = None
        else:
            ids = chunks["retrieved_id"].tolist()
            gold_rank = next(
                (rank for rank, article_id in enumerate(ids, start=1) if article_id == question.target_id),
                None,
            )
            substring_rank = find_answer_rank(
                chunks["page_content"].tolist(),
                question.answer_aliases,
            )
        row: dict[str, object] = {
            "question_id": question.question_id,
            "dataset": question.dataset,
            "decile": int(question.pop_decile_unweighted),
        }
        for k in k_values:
            row[f"gold_document_recall@{k}"] = float(gold_rank is not None and gold_rank <= k)
            row[f"substring_recall@{k}"] = float(
                substring_rank is not None and substring_rank <= k
            )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_recall(
    results: dict[str, pd.DataFrame],
    *,
    k_values: Sequence[int],
) -> pd.DataFrame:
    """Aggregate backend recall and question-clustered uncertainty by decile."""
    rows: list[dict[str, object]] = []
    for backend, frame in results.items():
        for metric in ("gold_document", "substring"):
            for k in k_values:
                column = f"{metric}_recall@{k}"
                for decile, group in frame.groupby("decile"):
                    recall = float(group[column].mean())
                    n_questions = len(group)
                    rows.append(
                        {
                            "backend": backend,
                            "metric": metric,
                            "k": k,
                            "decile": int(decile),
                            "n_questions": n_questions,
                            "recall": recall,
                            "ci95": 1.96 * np.sqrt(recall * (1.0 - recall) / n_questions),
                        }
                    )
    return pd.DataFrame(rows)


def summarize_disagreement(
    results: dict[str, pd.DataFrame],
    *,
    k_values: Sequence[int],
) -> pd.DataFrame:
    """Summarize agreement between gold-document and substring recall.

    Args:
        results: Per-question results keyed by backend.
        k_values: Retrieval depths to summarize.

    Returns:
        Category shares by backend, cutoff, and popularity decile.
    """
    rows: list[dict[str, object]] = []
    for backend, frame in results.items():
        for k in k_values:
            gold = frame[f"gold_document_recall@{k}"].astype(bool)
            substring = frame[f"substring_recall@{k}"].astype(bool)
            categories = np.select(
                [gold & substring, gold & ~substring, ~gold & substring],
                ["both", "gold_only", "substring_only"],
                default="neither",
            )
            categorized = frame.assign(category=categories)
            counts = (
                categorized.groupby(["decile", "category"])
                .size()
                .unstack(fill_value=0)
                .reindex(columns=["both", "gold_only", "substring_only", "neither"], fill_value=0)
            )
            shares = counts.div(counts.sum(axis=1), axis=0)
            for decile, values in shares.iterrows():
                for category, share in values.items():
                    rows.append(
                        {
                            "backend": backend,
                            "k": k,
                            "decile": int(decile),
                            "category": category,
                            "share": float(share),
                            "n_questions": int(counts.loc[decile].sum()),
                        }
                    )
    return pd.DataFrame(rows)


def plot_comparison(summary: pd.DataFrame, output_path: Path) -> None:
    """Plot Recall@1/3/5/10 for both relevance definitions and backends."""
    figure, axes = plt.subplots(2, 2, figsize=(12.4, 8.2), sharex=True, sharey=True)
    panels = [
        ("gold_document", 5, "A. Gold-document Recall@5"),
        ("gold_document", 10, "B. Gold-document Recall@10"),
        ("substring", 5, "C. Answer-substring Recall@5"),
        ("substring", 10, "D. Answer-substring Recall@10"),
    ]
    for axis, (metric, k, title) in zip(axes.flat, panels):
        subset = summary[(summary["metric"] == metric) & (summary["k"] == k)]
        for backend, (label, color, marker) in BACKENDS.items():
            values = subset[subset["backend"] == backend].sort_values("decile")
            axis.errorbar(
                values["decile"] + 1,
                values["recall"] * 100,
                yerr=values["ci95"] * 100,
                label=label,
                color=color,
                marker=marker,
                linewidth=2.1,
                markersize=5,
                capsize=2.5,
            )
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xticks(range(1, 11))
        axis.set_ylim(0, 100)
        axis.grid(axis="y", color="#D7D2C8", linewidth=0.8, alpha=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    for axis in axes[:, 0]:
        axis.set_ylabel("Question Recall (%)", fontweight="bold")
    for axis in axes[-1, :]:
        axis.set_xlabel("Target Popularity Decile (1=Rare, 10=Famous)", fontweight="bold")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.955),
    )
    figure.suptitle(
        "Retrieval Success Depends on the Relevance Definition",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    figure.patch.set_facecolor("#FAF8F2")
    for axis in axes.flat:
        axis.set_facecolor("#FAF8F2")
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def plot_disagreement(summary: pd.DataFrame, output_path: Path, *, k: int = 10) -> None:
    """Plot the four paired gold-document/substring outcomes by decile."""
    colors = {
        "both": "#2F6B4F",
        "gold_only": "#D97706",
        "substring_only": "#2563EB",
        "neither": "#C9C3B8",
    }
    labels = {
        "both": "Both",
        "gold_only": "Gold article only",
        "substring_only": "Substring only",
        "neither": "Neither",
    }
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), sharex=True, sharey=True)
    for axis, (backend, (backend_label, _, _)) in zip(axes, BACKENDS.items()):
        subset = summary[(summary["backend"] == backend) & (summary["k"] == k)]
        pivot = subset.pivot(index="decile", columns="category", values="share")
        bottom = np.zeros(len(pivot))
        for category in ("both", "gold_only", "substring_only", "neither"):
            values = pivot[category].to_numpy() * 100
            axis.bar(
                pivot.index + 1,
                values,
                bottom=bottom,
                color=colors[category],
                width=0.72,
                label=labels[category],
            )
            bottom += values
        axis.set_title(backend_label, loc="left", fontweight="bold")
        axis.set_xlabel("Target Popularity Decile", fontweight="bold")
        axis.set_xticks(range(1, 11))
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_facecolor("#FAF8F2")
    axes[0].set_ylabel("Questions (%)", fontweight="bold")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="upper center", ncol=4, frameon=False)
    figure.suptitle(
        f"Gold-Document and Answer-Substring Recall@{k} Agreement",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    figure.patch.set_facecolor("#FAF8F2")
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def run_analysis(
    results_dir: Path,
    output_dir: Path,
    *,
    excluded_datasets: Sequence[str] = DEFAULT_EXCLUSIONS,
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> pd.DataFrame:
    """Run the paired recall analysis and write tabular and graphical outputs."""
    questions = pd.read_parquet(results_dir / "cyro_qa_cache.parquet")
    questions["question_id"] = questions["question_id"].astype(str)
    questions = questions.loc[~questions["dataset"].isin(excluded_datasets)].copy()
    questions["answer_aliases"] = questions["answer_texts"].apply(parse_answer_texts)
    questions = questions.loc[questions["answer_aliases"].str.len() > 0]
    questions = questions.drop_duplicates("question_id", keep="first")
    questions["target_id"] = questions["wikipedia_id"].map(_normalise_id)
    questions = questions.loc[questions["pop_decile_unweighted"].between(0, 9)].copy()

    results = {
        backend: evaluate_backend(
            results_dir / f"retrieved_docs_{backend}.csv",
            questions,
            k_values=k_values,
        )
        for backend in BACKENDS
    }
    summary = summarize_recall(results, k_values=k_values)
    disagreement = summarize_disagreement(results, k_values=k_values)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "gold_document_vs_substring_recall_by_decile.csv", index=False)
    disagreement.to_csv(
        output_dir / "gold_document_vs_substring_disagreement_by_decile.csv",
        index=False,
    )
    plot_comparison(
        summary,
        output_dir / "gold_document_vs_substring_recall_by_decile.png",
    )
    plot_disagreement(
        disagreement,
        output_dir / "gold_document_vs_substring_disagreement_by_decile.png",
    )
    return summary


def main() -> None:
    """Run the recall comparison from command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DATA_DIR / "wiki_full_bil" / "all_qa_60k_balanced",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output_dir = args.output_dir or args.results_dir / "answer_eval_results"
    run_analysis(args.results_dir, output_dir)
    logger.info("Saved comparison outputs to %s", output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
