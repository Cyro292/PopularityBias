"""Measure scored BM25+ and FAISS-high neighborhood density by popularity.

For a balanced sample of target questions from each chunk-weighted popularity
decile, this analysis retrieves 100 candidates from both services. It excludes
the target article and measures: (1) how many wrong candidates occupy the
closest 5% of the query's observed score range and (2) the same-decile share
among the top 10 wrong neighbors. These are local density proxies, not a global
corpus clustering algorithm.
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
from src.rag.bm25_rag_service import BM25RagService
from src.rag.faiss_rag_service import FaissRagService
from src.rag.utils import IndexingConfig

logger = logging.getLogger(__name__)

METHODS = {
    "bm25": {"label": "BM25+", "color": "#E69F00", "marker": "o", "higher_is_better": True},
    "faiss": {"label": "FAISS high", "color": "#0072B2", "marker": "s", "higher_is_better": False},
}


def _normalise_id(value: object) -> str:
    """Return a stable string representation of a Wikipedia identifier."""
    value_str = str(value).strip()
    return value_str[:-2] if value_str.endswith(".0") else value_str


def _sample_questions(questions: pd.DataFrame, *, sample_per_decile: int, seed: int) -> pd.DataFrame:
    """Draw an equal-sized deterministic question sample from every decile."""
    samples = []
    for decile, group in questions.groupby("pop_decile_chunk_weighted", sort=True):
        if len(group) < sample_per_decile:
            raise ValueError(f"Decile {decile + 1} has only {len(group)} questions")
        samples.append(group.sample(n=sample_per_decile, random_state=seed + int(decile)))
    return pd.concat(samples, ignore_index=True)


def _metrics_from_results(
    sample: pd.DataFrame,
    results: list[list[tuple[dict[str, object], float]]],
    *,
    method: str,
    boundaries: np.ndarray,
) -> list[dict[str, float | int | str]]:
    """Compute local wrong-neighborhood metrics from ranked scored candidates."""
    method_config = METHODS[method]
    rows: list[dict[str, float | int | str]] = []
    for question, scored_candidates in zip(sample.itertuples(index=False), results):
        wrong = []
        for metadata, score in scored_candidates:
            if _normalise_id(metadata.get("wikipedia_id")) == question.target_id:
                continue
            popularity = pd.to_numeric(metadata.get("popularity_avg"), errors="coerce")
            if pd.isna(popularity) or popularity < 0:
                continue
            wrong.append((float(score), int(assign_decile(float(popularity), boundaries))))
        if len(wrong) < 2:
            continue
        scores = np.asarray([score for score, _ in wrong], dtype=float)
        best = scores.max() if method_config["higher_is_better"] else scores.min()
        worst = scores.min() if method_config["higher_is_better"] else scores.max()
        score_range = abs(best - worst)
        if method_config["higher_is_better"]:
            near_best = scores >= best - 0.05 * score_range
        else:
            near_best = scores <= best + 0.05 * score_range
        top_ten_deciles = [decile for _, decile in wrong[:10]]
        rows.append({
            "method": method,
            "question_id": str(question.question_id),
            "decile": int(question.pop_decile_chunk_weighted),
            "near_best_wrong_count": int(near_best.sum()),
            "same_decile_share_top10": float(np.mean(np.asarray(top_ten_deciles) == question.pop_decile_chunk_weighted)),
        })
    return rows


def _summarise(metrics: pd.DataFrame, column: str) -> pd.DataFrame:
    """Calculate decile means and normal-approximation 95% intervals."""
    summary = metrics.groupby(["method", "decile"], as_index=False).agg(
        value=(column, "mean"),
        sem=(column, "sem"),
        n=(column, "size"),
    )
    summary["ci95"] = 1.96 * summary["sem"].fillna(0.0)
    return summary


def plot_neighborhood_density(metrics: pd.DataFrame, output_path: Path) -> None:
    """Plot local score density and same-decile composition by method and decile."""
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.6))
    panels = [
        ("near_best_wrong_count", "A. Near-Best Wrong Neighbors", "Wrong Candidates in Closest 5% of Score Range"),
        ("same_decile_share_top10", "B. Same-Decile Locality", "Same-Decile Share of Top 10 Wrong Neighbors (%)"),
    ]
    for axis, (column, title, ylabel) in zip(axes, panels):
        summary = _summarise(metrics, column)
        for method, config in METHODS.items():
            subset = summary[summary["method"] == method].sort_values("decile")
            y_values = subset["value"] * 100 if column.endswith("share_top10") else subset["value"]
            ci_values = subset["ci95"] * 100 if column.endswith("share_top10") else subset["ci95"]
            axis.errorbar(
                subset["decile"] + 1,
                y_values,
                yerr=ci_values,
                color=config["color"],
                marker=config["marker"],
                linewidth=1.8,
                markersize=4.5,
                capsize=2.5,
                label=config["label"],
            )
        axis.set_title(title, fontweight="bold")
        axis.set_xlabel("Target Popularity Decile (1=Rare to 10=Famous)", fontweight="bold")
        axis.set_ylabel(ylabel, fontweight="bold")
        axis.set_xticks(range(1, 11))
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="upper left")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    logger.info("Saved neighborhood-density figure to %s", output_path)


def main() -> None:
    """Run balanced scored-neighborhood analysis for BM25+ and FAISS high."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions-path",
        type=Path,
        default=DATA_DIR / "wiki_full_bil" / "all_qa_60k_balanced" / "cyro_qa_cache.parquet",
    )
    parser.add_argument("--bm25-index-path", type=Path, default=DATA_DIR / "wiki_full_bil" / "bm25_bm25plus")
    parser.add_argument("--faiss-index-path", type=Path, default=DATA_DIR / "wiki_full_bil" / "faiss_high")
    parser.add_argument("--metadata-path", type=Path, default=DATA_DIR / "wiki_full_bil" / "metadata.json")
    parser.add_argument("--sample-per-decile", type=int, default=250)
    parser.add_argument("--depth", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=DATA_DIR / "wiki_full_bil" / "all_qa_60k_balanced" / "sampled_neighborhood_density_60k.parquet",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT_DIR / "paper_figures" / "retrieval_neighborhood_density_60k.png",
    )
    args = parser.parse_args()
    if args.sample_per_decile <= 0 or args.depth < 10 or args.batch_size <= 0:
        raise ValueError("--sample-per-decile, --depth, and --batch-size must be positive; --depth >= 10")
    required_paths = [args.questions_path, args.bm25_index_path, args.faiss_index_path, args.metadata_path]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Required input(s) not found: {missing_paths}")

    questions = pd.read_parquet(
        args.questions_path,
        columns=["question_id", "question_text", "wikipedia_id", "pop_decile_chunk_weighted"],
    )
    questions = questions[questions["pop_decile_chunk_weighted"].between(0, 9)].copy()
    questions["question_id"] = questions["question_id"].astype(str)
    questions["target_id"] = questions["wikipedia_id"].map(_normalise_id)
    sample = _sample_questions(questions, sample_per_decile=args.sample_per_decile, seed=args.seed)
    with args.metadata_path.open(encoding="utf-8") as metadata_file:
        boundaries = np.asarray(json.load(metadata_file)["decile_boundaries_chunk_weighted"], dtype=float)

    bm25 = BM25RagService(method="bm25+")
    bm25.load_index(args.bm25_index_path)
    faiss_config = IndexingConfig(
        embedding_provider="huggingface",
        embedding_model="Lajavaness/bilingual-embedding-small",
        normalise_embeddings=True,
        trust_remote_code=True,
    )
    faiss = FaissRagService(
        config=faiss_config,
        strategy="ivfpq",
        distance_strategy="cosine",
        ivfpq_nprobe=256,
    )
    faiss.load_index(args.faiss_index_path)

    metric_rows: list[dict[str, float | int | str]] = []
    for start in range(0, len(sample), args.batch_size):
        batch = sample.iloc[start : start + args.batch_size]
        queries = batch["question_text"].tolist()
        bm25_results = bm25.batch_retrieve_metadata_with_scores(
            queries, top_k=args.depth, progress_bar=False, batch_size=args.batch_size
        )
        faiss_results = faiss.batch_retrieve_metadata_with_scores(queries, top_k=args.depth, progress_bar=False)
        metric_rows.extend(_metrics_from_results(batch, bm25_results, method="bm25", boundaries=boundaries))
        metric_rows.extend(_metrics_from_results(batch, faiss_results, method="faiss", boundaries=boundaries))
        logger.info("Scored %d of %d sampled questions", min(start + len(batch), len(sample)), len(sample))

    metrics = pd.DataFrame(metric_rows)
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_parquet(args.metrics_path, index=False)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_neighborhood_density(metrics, args.output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
