"""Plot wrong-retrieval popularity preference under equal-article deciles.

The figure compares BM25+ and FAISS-high wrong retrieved chunks against random
article and random chunk baselines. Popularity deciles are unweighted, so each
decile contains an equal number of corpus articles while chunk exposure remains
uneven across those deciles.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import DATA_DIR
from src.metrics.decile_utils import load_corpus_distributions

logger = logging.getLogger(__name__)


def _normalise_id(value: object) -> str:
    """Return a stable string representation of a Wikipedia identifier."""
    value_str = str(value).strip()
    return value_str[:-2] if value_str.endswith(".0") else value_str


def compute_wrong_retrieval_preference(
    results_path: Path,
    targets: pd.DataFrame,
    *,
    k: int,
) -> pd.DataFrame:
    """Calculate wrong-retrieval preference for more popular articles by decile.

    Args:
        results_path: Retrieval checkpoint CSV for one retrieval method.
        targets: Target question metadata with article IDs and unweighted deciles.
        k: Number of retrieved chunks inspected per question.

    Returns:
        Per-decile preference probabilities, 95% confidence intervals, and counts.
    """
    retrieved = pd.read_csv(
        results_path,
        usecols=["question_id", "doc_rank", "metadata_wikipedia_id", "metadata_popularity_avg"],
    )
    retrieved = retrieved[retrieved["doc_rank"] < k].copy()
    retrieved["question_id"] = retrieved["question_id"].astype(str)
    retrieved["retrieved_id"] = retrieved["metadata_wikipedia_id"].map(_normalise_id)
    retrieved["metadata_popularity_avg"] = pd.to_numeric(
        retrieved["metadata_popularity_avg"], errors="coerce"
    )
    merged = retrieved.merge(targets, on="question_id", how="inner")
    wrong = merged[
        (merged["retrieved_id"] != merged["target_id"])
        & merged["metadata_popularity_avg"].notna()
        & (merged["metadata_popularity_avg"] >= 0)
    ].copy()
    wrong["more_popular"] = wrong["metadata_popularity_avg"] > wrong["popularity_avg"]

    rows = []
    for decile in range(10):
        values = wrong.loc[wrong["pop_decile_unweighted"] == decile, "more_popular"]
        count = len(values)
        probability = float(values.mean()) if count else np.nan
        ci95 = float(1.96 * np.sqrt(probability * (1.0 - probability) / count)) if count else np.nan
        rows.append({"decile": decile, "preference": probability, "ci95": ci95, "n_wrong_chunks": count})
    return pd.DataFrame(rows)


def random_more_popular_baseline(probabilities: np.ndarray) -> np.ndarray:
    """Calculate per-decile P(random item is more popular than the target)."""
    return np.array(
        [probabilities[decile + 1 :].sum() + 0.5 * probabilities[decile] for decile in range(10)]
    )


def plot_preference_curve(
    bm25: pd.DataFrame,
    faiss: pd.DataFrame,
    article_baseline: np.ndarray,
    chunk_baseline: np.ndarray,
    output_path: Path,
) -> None:
    """Render observed wrong-retrieval preferences and exposure baselines.

    Args:
        bm25: Per-decile BM25+ preference summary.
        faiss: Per-decile FAISS-high preference summary.
        article_baseline: Random-article preference probability by decile.
        chunk_baseline: Random-chunk preference probability by decile.
        output_path: PNG destination.
    """
    x_values = np.arange(1, 11)
    figure, axis = plt.subplots(figsize=(7.6, 4.9))
    axis.plot(x_values, article_baseline * 100, color="#222222", linewidth=1.8, linestyle="--",
              label="Random article baseline")
    axis.plot(x_values, chunk_baseline * 100, color="#666666", linewidth=1.8, linestyle=":",
              label="Random chunk baseline")
    for summary, color, marker, label in [
        (bm25, "#E69F00", "o", "BM25+"),
        (faiss, "#0072B2", "s", "FAISS high"),
    ]:
        axis.errorbar(
            x_values,
            summary["preference"] * 100,
            yerr=summary["ci95"] * 100,
            color=color,
            marker=marker,
            linewidth=1.9,
            markersize=5,
            capsize=2.5,
            label=label,
        )
    axis.set_xlabel("Expected Article Popularity Decile (1=Rare to 10=Famous)", fontweight="bold")
    axis.set_ylabel("Wrong Retrieval More Popular Than Target (%)", fontweight="bold")
    axis.set_title("Wrong-Retrieval Popularity Preference\nEqual-Article Deciles, 60k Balanced Cohort", fontweight="bold")
    axis.set_xticks(x_values)
    axis.set_ylim(0, 100)
    axis.grid(axis="y", alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.35))
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    logger.info("Saved preference curve to %s", output_path)


def main() -> None:
    """Generate the equal-article 60k wrong-retrieval preference curve."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DATA_DIR / "wiki_full_bil" / "all_qa_60k_balanced",
    )
    parser.add_argument(
        "--targets-path",
        type=Path,
        default=DATA_DIR / "wiki_full_bil" / "all_qa_60k_balanced" / "cyro_qa_cache.parquet",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=DATA_DIR / "wiki_full_bil" / "metadata.json",
    )
    parser.add_argument("--output-path", type=Path,
                        default=ROOT_DIR / "paper_figures" / "wrong_retrieval_preference_equal_article_60k.png")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    if args.k <= 0:
        raise ValueError("--k must be positive")

    bm25_path = args.results_dir / "retrieved_docs_bm25_plus.csv"
    faiss_path = args.results_dir / "retrieved_docs_ivfpq_high.csv"
    required_paths = [bm25_path, faiss_path, args.targets_path, args.metadata_path]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Required input(s) not found: {missing_paths}")

    targets = pd.read_parquet(
        args.targets_path,
        columns=["question_id", "wikipedia_id", "popularity_avg", "pop_decile_unweighted"],
    )
    targets["question_id"] = targets["question_id"].astype(str)
    targets["target_id"] = targets["wikipedia_id"].map(_normalise_id)
    with args.metadata_path.open(encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    distributions = load_corpus_distributions(metadata.get("corpus_stats", {}), "unweighted")
    if distributions is None:
        raise ValueError("Metadata does not contain unweighted article and chunk distributions")
    documents_per_decile, chunks_per_decile = distributions
    article_baseline = random_more_popular_baseline(documents_per_decile / documents_per_decile.sum())
    chunk_baseline = random_more_popular_baseline(chunks_per_decile / chunks_per_decile.sum())

    bm25 = compute_wrong_retrieval_preference(bm25_path, targets, k=args.k)
    faiss = compute_wrong_retrieval_preference(faiss_path, targets, k=args.k)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_preference_curve(bm25, faiss, article_baseline, chunk_baseline, args.output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
