#!/usr/bin/env python3
"""Retrieve and evaluate a FAISS index against a QA dataset.

Loads a pre-built FAISS index, runs batch retrieval over a QA question set,
computes Recall@K and MRR (overall and per-decile), and saves the raw
retrieval results as Parquet files compatible with ``rag_evaluation.ipynb``.

Usage:
    # Basic run with defaults
    python scripts/run_faiss_retrieval.py

    # Specify index, QA dataset, and output
    python scripts/run_faiss_retrieval.py \
        --index-dir data/faiss_migrated \
        --questions data/wiki_full_l/all_qa_8k.parquet \
        --output-dir data/faiss_migrated/eval_results \
        --top-k 10

    # Override embedding settings (must match what the index was built with)
    python scripts/run_faiss_retrieval.py \
        --embedding-provider modal \
        --embedding-model intfloat/multilingual-e5-large

    # Run with BM25 strategy as well
    python scripts/run_faiss_retrieval.py --strategies vector bm25

    # Limit to a small sample for testing
    python scripts/run_faiss_retrieval.py --max-questions 200

    python scripts/run_faiss_retrieval.py --help
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATA_DIR
from helpers.decile_utils import (
    COL_DECILE_CHUNK_WEIGHTED,
    COL_DECILE_UNWEIGHTED,
    COL_POPULARITY,
    assign_decile,
    decile_col_for,
    load_boundaries_from_metadata,
)
from rag.faiss_rag_service import FaissRagService, MemoryConfig
from rag.utils import IndexingConfig

logger = logging.getLogger(__name__)


# === Metric Helpers ==========================================================

def compute_recall_at_k(
    expected_id: str,
    topk_ids: list[str],
    k: int,
) -> float:
    """Return 1.0 if *expected_id* appears in the first *k* of *topk_ids*."""
    return 1.0 if expected_id in topk_ids[:k] else 0.0


def compute_reciprocal_rank(
    expected_id: str,
    topk_ids: list[str],
) -> float:
    """Return the reciprocal of the rank at which *expected_id* is found."""
    try:
        rank = topk_ids.index(expected_id) + 1
        return 1.0 / rank
    except ValueError:
        return 0.0


def compute_metrics(
    results_df: pd.DataFrame,
    k_values: list[int],
) -> dict[str, float]:
    """Compute Recall@K and MRR over the full results DataFrame.

    Args:
        results_df: Must contain columns ``wikipedia_id`` and ``topk_ids``.
        k_values: List of K cut-offs to evaluate.

    Returns:
        Dict mapping metric names to their values.
    """
    metrics: dict[str, float] = {}

    for k in k_values:
        col = f"recall@{k}"
        results_df[col] = results_df.apply(
            lambda row, _k=k: compute_recall_at_k(
                row["wikipedia_id"], row["topk_ids"], _k
            ),
            axis=1,
        )
        metrics[col] = results_df[col].mean()

    results_df["reciprocal_rank"] = results_df.apply(
        lambda row: compute_reciprocal_rank(
            row["wikipedia_id"], row["topk_ids"]
        ),
        axis=1,
    )
    metrics["mrr"] = results_df["reciprocal_rank"].mean()

    # Rank of the correct document (None when not found)
    def _rank(row):
        try:
            return row["topk_ids"].index(row["wikipedia_id"]) + 1
        except ValueError:
            return None

    results_df["rank"] = results_df.apply(_rank, axis=1)
    found = results_df["rank"].dropna()
    metrics["median_rank"] = found.median() if len(found) else None
    metrics["mean_rank"] = found.mean() if len(found) else None

    return metrics


def compute_per_decile_metrics(
    results_df: pd.DataFrame,
    decile_col: str,
    k_values: list[int],
) -> pd.DataFrame:
    """Return a DataFrame with per-decile Recall@K and MRR + confidence.

    Args:
        results_df: Results with recall/mrr columns already computed.
        decile_col: Name of the decile column to group by.
        k_values: K cut-offs.

    Returns:
        DataFrame indexed by decile (0-9).
    """
    rows: list[dict] = []
    valid = results_df[results_df[decile_col] >= 0]

    for decile in range(10):
        chunk = valid[valid[decile_col] == decile]
        n = len(chunk)
        if n == 0:
            continue

        row: dict = {"decile": decile, "count": n}

        for k in k_values:
            col = f"recall@{k}"
            vals = chunk[col].dropna()
            nk = len(vals)
            if nk > 0:
                p = vals.mean()
                se = np.sqrt(p * (1 - p) / nk)
                ci95 = 1.96 * se
            else:
                p, se, ci95 = np.nan, np.nan, np.nan
            row[col] = p
            row[f"{col}_se"] = se
            row[f"{col}_ci95"] = ci95

        rr = chunk["reciprocal_rank"].dropna()
        nr = len(rr)
        if nr > 0:
            row["mrr"] = rr.mean()
            row["mrr_se"] = rr.std(ddof=1) / np.sqrt(nr)
            row["mrr_ci95"] = 1.96 * row["mrr_se"]
        else:
            row["mrr"] = np.nan
            row["mrr_se"] = np.nan
            row["mrr_ci95"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


# === Result Processing =======================================================

def process_retrieval_results(
    qa_df: pd.DataFrame,
    all_results: list[list[tuple]],
) -> pd.DataFrame:
    """Convert raw retrieval output into a results DataFrame.

    The output schema matches what ``rag_evaluation.ipynb`` expects.
    """
    rows: list[dict] = []
    for question_row, retrieved_docs in zip(qa_df.itertuples(), all_results):
        expected_id = str(question_row.wikipedia_id).strip()

        retrieved_ids: list[str] = []
        retrieved_scores: list[float] = []
        retrieved_popularities: list[float | None] = []

        for doc, score in retrieved_docs:
            raw_id = doc.metadata.get("wikipedia_id", doc.metadata.get("id", ""))
            doc_id = str(int(float(raw_id))) if raw_id not in (None, "") else ""
            retrieved_ids.append(doc_id)
            retrieved_scores.append(float(score))
            retrieved_popularities.append(doc.metadata.get("popularity_avg"))

        rows.append({
            "question": question_row.question_text,
            "wikipedia_id": expected_id,
            "wikipedia_title": getattr(question_row, "wikipedia_title", None),
            "popularity_avg": getattr(question_row, "popularity_avg", None),
            "dataset": getattr(question_row, "dataset", None),
            COL_DECILE_UNWEIGHTED: getattr(
                question_row, COL_DECILE_UNWEIGHTED, -1
            ),
            COL_DECILE_CHUNK_WEIGHTED: getattr(
                question_row, COL_DECILE_CHUNK_WEIGHTED, -1
            ),
            "decile": getattr(question_row, "decile", -1),
            "topk_ids": retrieved_ids,
            "topk_scores": retrieved_scores,
            "topk_popularities": retrieved_popularities,
        })

    return pd.DataFrame(rows)


# === Printing Helpers ========================================================

def print_overall_metrics(
    strategy: str,
    metrics: dict[str, float],
    k_values: list[int],
) -> None:
    """Pretty-print overall metrics for a strategy."""
    print(f"\n{'=' * 60}")
    print(f"  {strategy.upper()} — Overall Metrics")
    print(f"{'=' * 60}")
    for k in k_values:
        val = metrics.get(f"recall@{k}", 0)
        print(f"  Recall@{k:<3d}  {val:.4f}")
    print(f"  MRR        {metrics.get('mrr', 0):.4f}")
    median = metrics.get("median_rank")
    mean = metrics.get("mean_rank")
    if median is not None:
        print(f"  Median rank  {median:.1f}")
    if mean is not None:
        print(f"  Mean rank    {mean:.1f}")


def print_per_decile_table(
    strategy: str,
    dm: pd.DataFrame,
    k_values: list[int],
    decile_mode: str,
) -> None:
    """Pretty-print a per-decile metrics table."""
    print(f"\n{'=' * 60}")
    print(f"  {strategy.upper()} — Per-Decile Metrics ({decile_mode})")
    print(f"{'=' * 60}")

    primary_k = k_values[-1]
    header = f"  {'Decile':>6}  {'Count':>7}"
    for k in k_values:
        header += f"  {'R@' + str(k):>8}"
    header += f"  {'MRR':>8}"
    print(header)
    print(f"  {'-' * (len(header) - 2)}")

    for _, row in dm.iterrows():
        d = int(row["decile"])
        line = f"  {d + 1:>6}  {int(row['count']):>7}"
        for k in k_values:
            col = f"recall@{k}"
            val = row.get(col, np.nan)
            line += f"  {val:>8.4f}" if not np.isnan(val) else f"  {'N/A':>8}"
        mrr_val = row.get("mrr", np.nan)
        line += f"  {mrr_val:>8.4f}" if not np.isnan(mrr_val) else f"  {'N/A':>8}"
        print(line)


# === Main ====================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    p = argparse.ArgumentParser(
        description="Retrieve and evaluate a FAISS index against a QA dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # -- Paths ----------------------------------------------------------------
    p.add_argument(
        "--index-dir",
        type=Path,
        default=DATA_DIR / "faiss_migrated",
        help="Directory containing the FAISS index (expects a faiss/ subdirectory). "
             "Default: data/faiss_migrated",
    )
    p.add_argument(
        "--questions",
        type=Path,
        default=DATA_DIR / "wiki_full_l" / "all_qa_8k.parquet",
        help="Path to the QA dataset parquet file. "
             "Default: data/wiki_full_l/all_qa_8k.parquet",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where results are written. "
             "Default: <index-dir>/eval_results",
    )
    p.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Path to metadata.json with decile boundaries. "
             "If not set, looks for metadata.json next to the questions file.",
    )

    # -- Retrieval ------------------------------------------------------------
    p.add_argument(
        "--strategies",
        nargs="+",
        default=["vector"],
        choices=["vector", "bm25"],
        help="Retrieval strategies to evaluate. Default: vector",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of documents to retrieve per query. Default: 10",
    )
    p.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10],
        help="K values to compute Recall@K for. Default: 1 3 5 10",
    )
    p.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Limit the number of questions (for testing). Default: all",
    )

    # -- FAISS strategy -------------------------------------------------------
    p.add_argument(
        "--faiss-strategy",
        default="ivfpq",
        choices=["vector", "hnsw", "ivfpq", "ivfpq_disk", "opq_ivfpq"],
        help="FAISS index type (must match how the index was built). "
             "Default: ivfpq",
    )
    p.add_argument(
        "--use-mmap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Memory-map the FAISS index for lower RAM usage. Default: on",
    )

    # -- Embedding config (must match the indexed embeddings) -----------------
    p.add_argument(
        "--embedding-provider",
        default="modal",
        help="Embedding provider. Default: modal",
    )
    p.add_argument(
        "--embedding-model",
        default="intfloat/multilingual-e5-large",
        help="Embedding model name. Default: intfloat/multilingual-e5-large",
    )
    p.add_argument(
        "--gpu-batch-size",
        type=int,
        default=512,
        help="Batch size for GPU embedding. Default: 512",
    )
    p.add_argument(
        "--request-batch-size",
        type=int,
        default=100,
        help="Number of texts per Modal request batch. Default: 100",
    )
    p.add_argument(
        "--normalise-embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalise embedding vectors. Default: on",
    )

    # -- Decile config --------------------------------------------------------
    p.add_argument(
        "--decile-mode",
        default="chunk_weighted",
        choices=["chunk_weighted", "unweighted"],
        help="Decile mode for per-decile metrics. Default: chunk_weighted",
    )

    # -- Misc -----------------------------------------------------------------
    p.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip strategies whose result parquet already exists. Default: on",
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )

    return p


def main() -> None:
    """Entry point."""
    args = build_arg_parser().parse_args()

    # -- Logging --------------------------------------------------------------
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    # -- Resolve paths --------------------------------------------------------
    index_dir: Path = args.index_dir
    questions_path: Path = args.questions
    output_dir: Path = args.output_dir or (index_dir / "eval_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path: Path = args.metadata or (questions_path.parent / "metadata.json")

    strategies: list[str] = args.strategies
    top_k: int = args.top_k
    k_values: list[int] = sorted(set(args.k_values) | {top_k})
    retrieve_k: int = max(k_values)
    decile_mode: str = args.decile_mode

    logger.info("Configuration:")
    logger.info("  Index dir:     %s", index_dir)
    logger.info("  Questions:     %s", questions_path)
    logger.info("  Output dir:    %s", output_dir)
    logger.info("  Strategies:    %s", strategies)
    logger.info("  Top-K:         %d", top_k)
    logger.info("  K values:      %s", k_values)
    logger.info("  Decile mode:   %s", decile_mode)

    # -- Load QA dataset ------------------------------------------------------
    logger.info("Loading QA dataset from %s ...", questions_path)
    if not questions_path.exists():
        logger.error("Questions file not found: %s", questions_path)
        sys.exit(1)

    qa_df = pd.read_parquet(questions_path)
    qa_df = qa_df.dropna(subset=["question_text"])
    qa_df["wikipedia_id"] = qa_df["wikipedia_id"].astype(str).str.strip()

    if args.max_questions:
        qa_df = qa_df.sample(
            n=min(args.max_questions, len(qa_df)), random_state=42
        )
        logger.info("  Limited to %d questions for testing", len(qa_df))

    logger.info("  Loaded %d questions (%d unique docs)",
                len(qa_df), qa_df["wikipedia_id"].nunique())

    # -- Load FAISS service ---------------------------------------------------
    logger.info("Loading FAISS index from %s ...", index_dir)
    if not index_dir.exists():
        logger.error("Index directory not found: %s", index_dir)
        sys.exit(1)

    config = IndexingConfig(
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        gpu_batch_size=args.gpu_batch_size,
        request_batch_size=args.request_batch_size,
        normalise_embeddings=args.normalise_embeddings,
        trust_remote_code=True,
        use_progress=False,
    )

    service = FaissRagService(
        config=config,
        strategy=args.faiss_strategy,
        distance_strategy="cosine",
        memory_config=MemoryConfig(use_mmap=args.use_mmap),
    )

    store = service.load_faiss_store(index_dir, use_mmap=args.use_mmap)
    if store is None:
        logger.error("Failed to load FAISS index from %s", index_dir)
        sys.exit(1)

    stats = service.get_index_stats()
    logger.info("  Index loaded — %d vectors, strategy=%s, mmap=%s",
                stats.get("n_vectors", 0), stats.get("strategy", "?"),
                stats.get("is_mmap", "?"))

    # -- Load decile boundaries (optional) ------------------------------------
    boundaries_uw = None
    boundaries_cw = None
    has_boundaries = False

    if metadata_path.exists():
        try:
            boundaries_uw, boundaries_cw, _ = load_boundaries_from_metadata(
                metadata_path
            )
            has_boundaries = True
            logger.info("  Loaded decile boundaries from %s", metadata_path)
        except Exception as e:
            logger.warning("Could not load boundaries from %s: %s",
                           metadata_path, e)
    else:
        logger.warning("Metadata file not found at %s — "
                       "per-decile metrics will be skipped.", metadata_path)

    # -- Run retrieval per strategy -------------------------------------------
    results_by_strategy: dict[str, pd.DataFrame] = {}

    for strategy in strategies:
        result_path = output_dir / f"results_{strategy}.parquet"

        if args.skip_existing and result_path.exists():
            logger.info("Skipping %s — results already exist at %s",
                        strategy, result_path)
            results_by_strategy[strategy] = pd.read_parquet(result_path)
            continue

        logger.info("Running retrieval: strategy=%s, top_k=%d, questions=%d",
                    strategy, retrieve_k, len(qa_df))
        t0 = time.perf_counter()

        all_results = service.batch_retrieve(
            questions=qa_df["question_text"].tolist(),
            top_k=retrieve_k,
            strategy=strategy,
            progress_bar=True,
        )

        elapsed = time.perf_counter() - t0
        logger.info("  Retrieval complete in %.1f s (%.1f q/s)",
                    elapsed, len(qa_df) / elapsed if elapsed > 0 else 0)

        # Process into DataFrame
        results_df = process_retrieval_results(qa_df, all_results)
        results_by_strategy[strategy] = results_df

        # Save immediately
        results_df.to_parquet(result_path)
        logger.info("  Saved %d rows -> %s", len(results_df), result_path)

    # -- Compute & print metrics ----------------------------------------------
    all_metrics: dict[str, dict] = {}

    for strategy, results_df in results_by_strategy.items():
        metrics = compute_metrics(results_df, k_values)
        all_metrics[strategy] = metrics
        print_overall_metrics(strategy, metrics, k_values)

        # Per-decile breakdown (if boundaries available)
        if has_boundaries:
            pop = results_df[COL_POPULARITY]
            results_df[COL_DECILE_UNWEIGHTED] = assign_decile(
                pop, boundaries_uw
            ).astype(int)
            results_df[COL_DECILE_CHUNK_WEIGHTED] = assign_decile(
                pop, boundaries_cw
            ).astype(int)

            decile_col = decile_col_for(decile_mode)
            dm = compute_per_decile_metrics(results_df, decile_col, k_values)
            print_per_decile_table(strategy, dm, k_values, decile_mode)

            # Save per-decile CSV
            dm_path = output_dir / f"decile_metrics_{strategy}.csv"
            dm.to_csv(dm_path, index=False)
            logger.info("  Saved per-decile metrics -> %s", dm_path)

    # -- Save summary CSV -----------------------------------------------------
    summary_path = output_dir / "metrics_comparison.csv"
    pd.DataFrame(all_metrics).T.to_csv(summary_path)
    logger.info("Saved metrics summary -> %s", summary_path)

    print(f"\nAll results written to: {output_dir}")


if __name__ == "__main__":
    main()
