"""Compute direct BM25 score-competition diagnostics for an evaluation cohort.

The process uses :class:`src.rag.bm25_rag_service.BM25RagService` to re-run a
persisted local BM25 index at a chosen depth and records, per query, the best
target score, best non-target score, score margin, and near-tie counts. These
metrics distinguish direct lexical competition from target-side lexical proxies
such as raw TF or lexical overlap.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import DATA_DIR
from src.rag.bm25_rag_service import BM25RagService

logger = logging.getLogger(__name__)


def _article_id(value: object) -> str:
    """Normalize Wikipedia identifiers for comparisons across stored formats."""
    return str(value).strip()


def _query_metrics(
    *,
    decile: int,
    question_id: str,
    target_id: str,
    results: list[tuple[object, float]],
) -> dict[str, float | int | str]:
    """Compute competition metrics for one scored BM25 result list.

    Args:
        decile: Target popularity decile.
        question_id: Unique query identifier.
        target_id: Target article identifier.
        results: Retrieved chunks and BM25 scores in descending score order.

    Returns:
        A flat row containing target presence/rank, margin, near-tie counts,
        and top-result score-distribution summaries.
    """
    result_ids = [_article_id(doc.metadata.get("wikipedia_id")) for doc, _ in results]
    target_indices = [i for i, result_id in enumerate(result_ids) if result_id == target_id]
    non_target_indices = [i for i, result_id in enumerate(result_ids) if result_id != target_id]
    score_array = np.asarray([score for _, score in results], dtype=float)
    score_sum = float(score_array.sum())
    score_share = score_array / score_sum if score_sum > 0 else np.array([])

    row: dict[str, float | int | str] = {
        "question_id": question_id,
        "decile": decile,
        "target_in_depth": int(bool(target_indices)),
        "target_rank": target_indices[0] + 1 if target_indices else np.nan,
        "score_entropy": float(-(score_share * np.log(score_share)).sum()) if len(score_share) else np.nan,
        "top1_score_share": float(score_share[0]) if len(score_share) else np.nan,
        "top1_top10_gap": float(score_array[0] - score_array[min(9, len(score_array) - 1)]),
    }
    if not target_indices or not non_target_indices:
        row.update({
            "best_target_score": np.nan,
            "best_non_target_score": np.nan,
            "target_margin": np.nan,
            "near_ties_1pct": np.nan,
            "near_ties_5pct": np.nan,
            "near_ties_10pct": np.nan,
        })
        return row

    best_target = float(score_array[target_indices].max())
    best_non_target = float(score_array[non_target_indices].max())
    non_target_scores = score_array[non_target_indices]
    row.update({
        "best_target_score": best_target,
        "best_non_target_score": best_non_target,
        "target_margin": best_target - best_non_target,
        "near_ties_1pct": int((non_target_scores >= best_target * 0.99).sum()),
        "near_ties_5pct": int((non_target_scores >= best_target * 0.95).sum()),
        "near_ties_10pct": int((non_target_scores >= best_target * 0.90).sum()),
    })
    return row


def analyse_competition(
    questions: pd.DataFrame,
    *,
    rag_service: BM25RagService,
    depth: int,
    batch_size: int,
) -> pd.DataFrame:
    """Re-run BM25 and calculate direct score competition for each question.

    Args:
        questions: Query rows containing question text, target IDs, and deciles.
        rag_service: Loaded BM25+ RAG service used for retrieval.
        depth: Number of ranked chunks to inspect per query.
        batch_size: Number of queries processed per BM25 batch.

    Returns:
        One competition-diagnostic row per question.
    """
    rows: list[dict[str, float | int | str]] = []

    for start in tqdm(range(0, len(questions), batch_size), desc="Scoring BM25 queries"):
        batch = questions.iloc[start : start + batch_size]
        scored_results = rag_service.batch_retrieve_with_scores(
            batch["question_text"].tolist(),
            top_k=depth,
            progress_bar=False,
            batch_size=batch_size,
        )
        for question, query_results in zip(batch.itertuples(index=False), scored_results):
            rows.append(_query_metrics(
                decile=int(question.pop_decile_chunk_weighted),
                question_id=str(question.question_id),
                target_id=_article_id(question.wikipedia_id),
                results=query_results,
            ))

    return pd.DataFrame(rows)


def main() -> None:
    """Run direct BM25 competition analysis for the standard 8k cohort."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions-path",
        type=Path,
        default=DATA_DIR / "wiki_full_bil" / "all_qa_8k" / "cyro_qa_cache.parquet",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=DATA_DIR / "wiki_full_bil" / "bm25_bm25plus_recursive",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DATA_DIR / "wiki_full_bil" / "all_qa_8k" / "bm25_competition_top100.parquet",
    )
    parser.add_argument("--depth", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--exclude-datasets",
        nargs="*",
        default=["hotpot_qa"],
        help="Datasets excluded when the lexical-factor cohort was created.",
    )
    args = parser.parse_args()

    if args.depth < 10:
        raise ValueError("depth must be at least 10 to calculate the top-1/top-10 gap")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if not args.questions_path.exists():
        raise FileNotFoundError(f"Questions file not found: {args.questions_path}")
    if not args.index_path.exists():
        raise FileNotFoundError(f"BM25 index not found: {args.index_path}")

    questions = pd.read_parquet(args.questions_path)
    questions = questions[~questions["dataset"].isin(args.exclude_datasets)].copy()
    required = {"question_id", "question_text", "wikipedia_id", "pop_decile_chunk_weighted"}
    missing = required - set(questions.columns)
    if missing:
        raise KeyError(f"Questions file missing required columns: {sorted(missing)}")

    logger.info("Analysing %d questions at depth %d", len(questions), args.depth)
    rag_service = BM25RagService(method="bm25+")
    rag_service.load_index(args.index_path)
    competition = analyse_competition(
        questions,
        rag_service=rag_service,
        depth=args.depth,
        batch_size=args.batch_size,
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    competition.to_parquet(args.output_path, index=False)
    logger.info("Saved per-query competition diagnostics to %s", args.output_path)

    summary = competition.groupby("decile").agg(
        n=("question_id", "size"),
        target_top100=("target_in_depth", "mean"),
        mean_target_rank=("target_rank", "mean"),
        mean_margin=("target_margin", "mean"),
        mean_near_ties_5pct=("near_ties_5pct", "mean"),
        mean_entropy=("score_entropy", "mean"),
    )
    print(summary.round(4).to_string())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
