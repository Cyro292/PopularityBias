"""Create paired analogue-pair boxplot and score-distribution figures.

The input is produced by ``src.process.analysis.build_analogue_similarity_scores``. It contains
manually selected pairs where ``*_1`` is a new 2026 article and ``*_2`` is its
known counterpart from the old corpus. A zero score means the target article
was not returned in the retrieval depth, so it is excluded from the plots and
statistical comparisons as in the original analysis.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import DATA_DIR

logger = logging.getLogger(__name__)

NEW_LABEL = "New (2026)"
OLD_LABEL = "Known (old corpus)"
PALETTE = {NEW_LABEL: "#E05C5C", OLD_LABEL: "#4C8BBF"}
SCORE_COLUMNS = ("bm25_score", "cosine_score")
PAPER_FIGURES_DIR = ROOT_DIR / "paper_figures"


def load_scores(path: Path) -> pd.DataFrame:
    """Load the wide pair table and reshape it for the original analysis.

    Args:
        path: Parquet output from the analogue similarity score builder.

    Returns:
        One row per article side of each manually selected pair.

    Raises:
        KeyError: If the score parquet lacks required columns.
    """
    scores = pd.read_parquet(path)
    required_columns = {
        "question_1", "question_2", "wiki_title_1", "wiki_title_2",
        "bm25_score_1", "bm25_score_2", "cosine_score_1", "cosine_score_2",
    }
    missing_columns = required_columns - set(scores.columns)
    if missing_columns:
        raise KeyError(f"Similarity scores missing required columns: {sorted(missing_columns)}")

    new_scores = pd.DataFrame({
        "pair_index": range(len(scores)),
        "question": scores["question_1"],
        "wiki_title": scores["wiki_title_1"],
        "bm25_score": scores["bm25_score_1"],
        "cosine_score": scores["cosine_score_1"],
        "source": NEW_LABEL,
    })
    old_scores = pd.DataFrame({
        "pair_index": range(len(scores)),
        "question": scores["question_2"],
        "wiki_title": scores["wiki_title_2"],
        "bm25_score": scores["bm25_score_2"],
        "cosine_score": scores["cosine_score_2"],
        "source": OLD_LABEL,
    })
    long_scores = pd.concat([new_scores, old_scores], ignore_index=True)

    # A zero is the scorer's sentinel for a target absent from the top-k results.
    for score_column in SCORE_COLUMNS:
        long_scores[score_column] = pd.to_numeric(long_scores[score_column], errors="coerce").replace(0.0, np.nan)
    return long_scores


def _paired_annotation(long_scores: pd.DataFrame, score_column: str) -> str:
    """Return a paired Wilcoxon annotation for one retrieval-score method."""
    paired = long_scores.pivot(index="pair_index", columns="source", values=score_column).dropna()
    if len(paired) < 2:
        return "n/a"
    differences = paired[OLD_LABEL] - paired[NEW_LABEL]
    if np.allclose(differences, 0):
        return f"paired n={len(paired)}, all differences = 0"
    _, p_value = stats.wilcoxon(paired[OLD_LABEL], paired[NEW_LABEL], method="auto")
    p_text = f"p={p_value:.3f}" if p_value >= 0.001 else "p<0.001"
    return f"paired n={len(paired)}, median Δ={differences.median():+.3f}, {p_text}"


def plot_boxplot(long_scores: pd.DataFrame, output_path: Path) -> None:
    """Plot BM25 and dense score boxplots with the original strip overlays.

    Args:
        long_scores: Reshaped score table returned by :func:`load_scores`.
        output_path: PNG destination.
    """
    sns.set_theme(style="whitegrid", font_scale=1.05)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    panels = (
        ("bm25_score", "BM25 Score Distribution"),
        ("cosine_score", "Dense (Cosine) Score Distribution"),
    )
    for axis, (score_column, title) in zip(axes, panels):
        sns.boxplot(
            data=long_scores,
            x="source",
            y=score_column,
            hue="source",
            palette=PALETTE,
            width=0.45,
            linewidth=1.2,
            fliersize=0,
            legend=False,
            order=[NEW_LABEL, OLD_LABEL],
            ax=axis,
        )
        sns.stripplot(
            data=long_scores,
            x="source",
            y=score_column,
            hue="source",
            palette=PALETTE,
            size=5,
            alpha=0.55,
            jitter=True,
            legend=False,
            order=[NEW_LABEL, OLD_LABEL],
            ax=axis,
        )
        new_scores = long_scores.loc[long_scores["source"] == NEW_LABEL, score_column]
        old_scores = long_scores.loc[long_scores["source"] == OLD_LABEL, score_column]
        axis.set_title(f"{title}\n({_paired_annotation(long_scores, score_column)})", fontweight="bold")
        axis.set_xlabel("")
        axis.set_ylabel(score_column.replace("_", " ").title())
    figure.suptitle("New 2026 Articles vs. Known Articles", fontweight="bold")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    logger.info("Saved similarity boxplot to %s", output_path)


def plot_score_distribution(long_scores: pd.DataFrame, output_path: Path) -> None:
    """Plot empirical score distributions without small-sample smoothing.

    Args:
        long_scores: Reshaped score table returned by :func:`load_scores`.
        output_path: PNG destination.
    """
    sns.set_theme(style="whitegrid", font_scale=1.05)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    panels = (
        ("bm25_score", "BM25 Score Distribution"),
        ("cosine_score", "Dense (Cosine) Score Distribution"),
    )
    for axis, (score_column, title) in zip(axes, panels):
        for label, color in PALETTE.items():
            values = long_scores.loc[long_scores["source"] == label, score_column].dropna()
            if len(values) < 2:
                continue
            sns.ecdfplot(values, ax=axis, label=label, color=color, linewidth=2)
            axis.scatter(
                values,
                np.zeros(len(values)),
                color=color,
                alpha=0.5,
                s=16,
                clip_on=False,
                zorder=3,
            )
            axis.axvline(values.median(), color=color, linestyle="--", linewidth=1.2, alpha=0.8)
        axis.set_title(title, fontweight="bold")
        axis.set_xlabel(score_column.replace("_", " ").title())
        axis.set_ylabel("Empirical cumulative proportion")
        axis.set_ylim(-0.04, 1.04)
        axis.legend(frameon=False)
    figure.suptitle("Similarity Score Distributions (All Scores Shown)", fontweight="bold")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    logger.info("Saved score-distribution figure to %s", output_path)


def main() -> None:
    """Create the two analogue-pair similarity figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-path", type=Path, default=DATA_DIR / "similarity_scores.parquet")
    parser.add_argument("--output-dir", type=Path, default=PAPER_FIGURES_DIR)
    args = parser.parse_args()
    if not args.scores_path.exists():
        raise FileNotFoundError(f"Similarity score file not found: {args.scores_path}")

    long_scores = load_scores(args.scores_path)
    plot_boxplot(long_scores, args.output_dir / "analogue_similarity_boxplot.png")
    plot_score_distribution(long_scores, args.output_dir / "analogue_similarity_score_distribution.png")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
