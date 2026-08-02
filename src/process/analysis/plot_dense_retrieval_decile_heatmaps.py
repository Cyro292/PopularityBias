"""Plot FAISS-high wrong-retrieval deviations from uniform exposure for 60k.

The same retrieved chunks are assigned popularity deciles in two ways:
chunk-weighted boundaries make each corpus decile contain an equal number of
chunks, while unweighted boundaries make each decile contain an equal number
of articles. Each heatmap row is normalized over wrong top-k chunks retrieved
for questions whose target article belongs to that expected decile, then has a
uniform 10% random-exposure baseline subtracted from every cell.
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
from src.metrics.decile_utils import assign_decile

logger = logging.getLogger(__name__)

DECILE_COLUMNS = {
    "chunk_weighted": "pop_decile_chunk_weighted",
    "unweighted": "pop_decile_unweighted",
}


def _normalise_id(value: object) -> str:
    """Return a stable string representation of a Wikipedia identifier."""
    value_str = str(value).strip()
    return value_str[:-2] if value_str.endswith(".0") else value_str


def compute_retrieval_heatmap(
    retrieved: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    target_decile_column: str,
    boundaries: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a row-normalized target-to-retrieved decile heatmap.

    Args:
        retrieved: Top-k FAISS-high chunks with retrieved popularity values.
        targets: Question metadata with target IDs and popularity deciles.
        target_decile_column: Target decile column for this boundary definition.
        boundaries: Popularity boundaries used to assign retrieved chunks.

    Returns:
        A 10-by-10 matrix of retrieved-chunk percentages by target decile and
        the number of wrong retrieved chunks contributing to each row.
    """
    merged = retrieved.merge(
        targets[["question_id", "target_id", target_decile_column]],
        on="question_id",
        how="inner",
    )
    merged = merged[merged["retrieved_id"] != merged["target_id"]].copy()
    merged["retrieved_decile"] = assign_decile(
        merged["metadata_popularity_avg"], boundaries
    ).astype(int)
    counts = pd.crosstab(
        merged[target_decile_column],
        merged["retrieved_decile"],
    ).reindex(index=range(10), columns=range(10), fill_value=0)
    row_totals = counts.sum(axis=1).to_numpy(dtype=float)
    safe_row_totals = pd.Series(row_totals).replace(0, np.nan)
    matrix = counts.div(safe_row_totals, axis=0).fillna(0.0).to_numpy() * 100.0
    return matrix, row_totals


def plot_heatmap(
    matrix: np.ndarray,
    row_totals: np.ndarray,
    output_path: Path,
    *,
    title: str,
    vmax: float,
) -> None:
    """Render one dense-retrieval deviation heatmap.

    Args:
        matrix: Retrieved chunk percentage-point deviations from a uniform 10% baseline.
        row_totals: Wrong retrieved-chunk count for each expected-decile row.
        output_path: PNG destination.
        title: Figure title.
        vmax: Shared absolute diverging-scale maximum across both figures.
    """
    figure, axis = plt.subplots(figsize=(7.2, 5.8))
    image = axis.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("Percentage Points vs. Uniform 10% Baseline", fontweight="bold")

    for row in range(10):
        for column in range(10):
            value = matrix[row, column]
            observed_share = (value + 10.0) / 100.0
            ci95 = 1.96 * np.sqrt(observed_share * (1.0 - observed_share) / row_totals[row]) * 100.0
            if abs(value) < 3.0 or abs(value) <= ci95:
                continue
            color = "white" if abs(value) >= vmax * 0.58 else "black"
            axis.text(column, row, f"{value:+.1f}", ha="center", va="center", fontsize=7, color=color)

    axis.set_xticks(range(10), labels=range(1, 11))
    axis.set_yticks(range(10), labels=range(1, 11))
    axis.set_xlabel("Retrieved Chunk Popularity Decile", fontweight="bold")
    axis.set_ylabel("Expected Article Popularity Decile", fontweight="bold")
    axis.set_title(title, fontweight="bold")
    axis.set_xticks(np.arange(-0.5, 10, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, 10, 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=0.8)
    axis.tick_params(which="minor", bottom=False, left=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    logger.info("Saved heatmap to %s", output_path)


def main() -> None:
    """Generate observed FAISS-high decile heatmaps for both decile definitions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-path",
        type=Path,
        default=DATA_DIR / "wiki_full_bil" / "all_qa_60k_balanced" / "retrieved_docs_ivfpq_high.csv",
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
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "paper_figures")
    parser.add_argument("--k", type=int, default=10, help="Number of retrieved chunks per question.")
    args = parser.parse_args()
    if args.k <= 0:
        raise ValueError("--k must be positive")
    required_paths = [args.results_path, args.targets_path, args.metadata_path]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Required input(s) not found: {missing_paths}")

    with args.metadata_path.open(encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    boundaries = {
        "chunk_weighted": np.asarray(metadata["decile_boundaries_chunk_weighted"], dtype=float),
        "unweighted": np.asarray(metadata["decile_boundaries_unweighted"], dtype=float),
    }
    targets = pd.read_parquet(
        args.targets_path,
        columns=["question_id", "wikipedia_id", *DECILE_COLUMNS.values()],
    )
    targets["question_id"] = targets["question_id"].astype(str)
    targets["target_id"] = targets["wikipedia_id"].map(_normalise_id)
    retrieved = pd.read_csv(
        args.results_path,
        usecols=["question_id", "doc_rank", "metadata_wikipedia_id", "metadata_popularity_avg"],
    )
    retrieved = retrieved[retrieved["doc_rank"] < args.k].copy()
    retrieved["question_id"] = retrieved["question_id"].astype(str)
    retrieved["retrieved_id"] = retrieved["metadata_wikipedia_id"].map(_normalise_id)
    retrieved["metadata_popularity_avg"] = pd.to_numeric(
        retrieved["metadata_popularity_avg"], errors="coerce"
    )
    retrieved = retrieved.dropna(subset=["metadata_popularity_avg"])
    retrieved = retrieved[retrieved["metadata_popularity_avg"] >= 0].copy()

    observed_results = {
        mode: compute_retrieval_heatmap(
            retrieved,
            targets,
            target_decile_column=column,
            boundaries=boundaries[mode],
        )
        for mode, column in DECILE_COLUMNS.items()
    }
    matrices = {mode: matrix - 10.0 for mode, (matrix, _) in observed_results.items()}
    vmax = float(np.ceil(max(np.abs(matrix).max() for matrix in matrices.values()) / 5.0) * 5.0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_heatmap(
        matrices["chunk_weighted"],
        observed_results["chunk_weighted"][1],
        args.output_dir / "faiss_high_wrong_retrieval_delta_uniform_chunk_weighted_60k.png",
        title="FAISS High Wrong Retrievals vs. Uniform (Chunk-Weighted Deciles)",
        vmax=vmax,
    )
    plot_heatmap(
        matrices["unweighted"],
        observed_results["unweighted"][1],
        args.output_dir / "faiss_high_wrong_retrieval_delta_uniform_article_weighted_60k.png",
        title="FAISS High Wrong Retrievals vs. Uniform (Equal-Article Deciles)",
        vmax=vmax,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
