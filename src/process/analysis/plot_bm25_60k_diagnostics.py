"""Generate BM25+ candidate and lexical-competition diagnostics for the 60k cohort.

The process uses :class:`src.rag.bm25_rag_service.BM25RagService` to re-run
the persisted BM25+ index to depth 100, so candidate recall and score
competition are measured beyond the stored top-10 checkpoint. It also streams
target chunks from the indexed corpus to measure query IDF and best
target-chunk query-term frequency using the index's tokenizer.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import DATA_DIR
from src.rag.bm25_rag_service import BM25RagService

logger = logging.getLogger(__name__)

RECALL_STYLES = [
    (1, "#0072B2", "o", "-"),
    (10, "#E69F00", "s", "--"),
    (50, "#009E73", "D", "-."),
    (100, "#CC79A7", "^", ":"),
]


def _normalise_id(value: object) -> str:
    """Return a stable string representation of a Wikipedia identifier."""
    value_str = str(value).strip()
    return value_str[:-2] if value_str.endswith(".0") else value_str


def _mean_ci(values: pd.Series) -> tuple[float, float, int]:
    """Calculate a mean and normal-approximation 95% confidence interval."""
    clean_values = values.replace([np.inf, -np.inf], np.nan).dropna()
    count = len(clean_values)
    if count == 0:
        return np.nan, np.nan, 0
    return float(clean_values.mean()), float(1.96 * clean_values.sem()), count


def _summarise(question_metrics: pd.DataFrame, column: str) -> pd.DataFrame:
    """Summarise a per-question metric by zero-based popularity decile."""
    rows = []
    for decile, group in question_metrics.groupby("decile", sort=True):
        mean, ci95, count = _mean_ci(group[column])
        rows.append({"decile": int(decile), "mean": mean, "ci95": ci95, "n": count})
    return pd.DataFrame(rows)


def compute_retrieval_diagnostics(
    questions: pd.DataFrame,
    *,
    rag_service: BM25RagService,
    depth: int,
    batch_size: int,
    checkpoint_path: Path | None = None,
    checkpoint_every_batches: int = 5,
) -> pd.DataFrame:
    """Retrieve BM25+ candidates and calculate per-question ranking metrics.

    Args:
        questions: Questions with IDs, target Wikipedia IDs, and deciles.
        rag_service: Loaded BM25+ RAG service used for retrieval.
        depth: Number of BM25+ candidates to inspect.
        batch_size: Number of queries per BM25+ batch.
        checkpoint_path: Optional parquet file updated during analysis and used to resume.
        checkpoint_every_batches: Number of batches between checkpoint writes.

    Returns:
        Per-question candidate recall and competition diagnostics.
    """
    if checkpoint_every_batches <= 0:
        raise ValueError("checkpoint_every_batches must be positive")

    existing = pd.DataFrame()
    if checkpoint_path and checkpoint_path.exists():
        existing = pd.read_parquet(checkpoint_path)
        completed_question_ids = set(existing["question_id"].astype(str))
        questions = questions[~questions["question_id"].astype(str).isin(completed_question_ids)].copy()
        logger.info("Resuming retrieval diagnostics with %d questions remaining", len(questions))
    rows: list[dict[str, float | int | str]] = existing.to_dict("records")

    for start in tqdm(range(0, len(questions), batch_size), desc="Retrieving BM25+ candidates"):
        batch = questions.iloc[start : start + batch_size]
        scored_results = rag_service.batch_retrieve_metadata_with_scores(
            batch["question_text"].tolist(),
            top_k=depth,
            progress_bar=False,
            batch_size=batch_size,
        )
        for question, scored_documents in zip(batch.itertuples(index=False), scored_results):
            target_id = _normalise_id(question.wikipedia_id)
            retrieved_ids = [
                _normalise_id(metadata.get("wikipedia_id"))
                for metadata, _ in scored_documents
            ]
            target_indices = [index for index, result_id in enumerate(retrieved_ids) if result_id == target_id]
            target_rank = target_indices[0] if target_indices else np.nan
            row: dict[str, float | int | str] = {
                "question_id": str(question.question_id),
                "decile": int(question.pop_decile_chunk_weighted),
                "recall_at_1": float(bool(target_indices and target_rank < 1)),
                "recall_at_10": float(bool(target_indices and target_rank < 10)),
                "recall_at_50": float(bool(target_indices and target_rank < 50)),
                "recall_at_100": float(bool(target_indices and target_rank < 100)),
                "target_rank": target_rank,
                "near_ties_5pct": np.nan,
            }
            if target_indices:
                score_array = np.asarray([score for _, score in scored_documents], dtype=float)
                best_target_score = float(score_array[target_indices].max())
                non_target_scores = score_array[
                    np.array([result_id != target_id for result_id in retrieved_ids])
                ]
                row["near_ties_5pct"] = float((non_target_scores >= best_target_score * 0.95).sum())
            rows.append(row)

        if checkpoint_path and (start // batch_size + 1) % checkpoint_every_batches == 0:
            pd.DataFrame(rows).to_parquet(checkpoint_path, index=False)
            logger.info("Checkpointed %d retrieval diagnostics to %s", len(rows), checkpoint_path)

    diagnostics = pd.DataFrame(rows)
    if checkpoint_path:
        diagnostics.to_parquet(checkpoint_path, index=False)
    return diagnostics


def compute_target_lexical_metrics(
    questions: pd.DataFrame,
    *,
    index_path: Path,
    corpus_path: Path,
    chunk_batch_size: int,
) -> pd.DataFrame:
    """Calculate query IDF and best target-chunk query-term frequency.

    Args:
        questions: Questions with target Wikipedia IDs and popularity deciles.
        index_path: Persisted ``bm25s`` index directory.
        corpus_path: BM25+ corpus JSONL containing indexed chunks.
        chunk_batch_size: Target chunks tokenized together while streaming the corpus.

    Returns:
        Per-question query-IDF and best target-chunk TF metrics.
    """
    import Stemmer
    import bm25s

    retriever = bm25s.BM25.load(str(index_path), load_corpus=False, mmap=True)
    stemmer = Stemmer.Stemmer("english")
    document_frequencies = np.diff(retriever.scores["indptr"])
    document_count = int(retriever.scores["num_docs"])
    query_tokens = bm25s.tokenize(
        questions["question_text"].tolist(),
        stopwords="en",
        stemmer=stemmer,
        show_progress=False,
    ).ids
    query_token_sets = [set(token_ids) for token_ids in query_tokens]
    target_to_rows: dict[str, list[int]] = defaultdict(list)
    for index, target_id in enumerate(questions["wikipedia_id"]):
        target_to_rows[_normalise_id(target_id)].append(index)

    query_idf_means = []
    for token_ids in query_token_sets:
        idfs = [
            float(np.log((document_count + 1) / document_frequencies[token_id]))
            for token_id in token_ids
        ]
        query_idf_means.append(float(np.mean(idfs)) if idfs else np.nan)
    best_chunk_tf = np.zeros(len(questions), dtype=float)

    def process_chunk_batch(chunks: list[tuple[str, str]]) -> None:
        """Update each target question's maximum query-term TF from one chunk batch."""
        if not chunks:
            return
        tokenized_chunks = bm25s.tokenize(
            [text for _, text in chunks],
            stopwords="en",
            stemmer=stemmer,
            show_progress=False,
        ).ids
        for (target_id, _), token_ids in zip(chunks, tokenized_chunks):
            token_counts = Counter(token_ids)
            for question_index in target_to_rows[target_id]:
                query_tf = sum(token_counts[token_id] for token_id in query_token_sets[question_index])
                best_chunk_tf[question_index] = max(best_chunk_tf[question_index], query_tf)

    chunks: list[tuple[str, str]] = []
    with corpus_path.open(encoding="utf-8") as corpus_file:
        for line in tqdm(corpus_file, desc="Scanning target chunks", unit="chunks"):
            document = json.loads(line)
            target_id = _normalise_id(document["wikipedia_id"])
            if target_id not in target_to_rows:
                continue
            chunks.append((target_id, document["text"]))
            if len(chunks) >= chunk_batch_size:
                process_chunk_batch(chunks)
                chunks = []
    process_chunk_batch(chunks)

    return pd.DataFrame({
        "question_id": questions["question_id"].astype(str),
        "decile": questions["pop_decile_chunk_weighted"].astype(int),
        "query_idf_mean": query_idf_means,
        "best_target_chunk_query_tf": best_chunk_tf,
    })


def _style_axis(axis: plt.Axes) -> None:
    """Apply common paper-figure axis styling."""
    axis.set_xticks(range(1, 11))
    axis.grid(axis="y", alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)


def plot_candidate_recall(diagnostics: pd.DataFrame, output_path: Path) -> None:
    """Plot BM25+ candidate Recall@1/10/50/100 by popularity decile."""
    figure, axis = plt.subplots(figsize=(7.4, 4.6))
    for cutoff, color, marker, linestyle in RECALL_STYLES:
        summary = _summarise(diagnostics, f"recall_at_{cutoff}")
        x_values = summary["decile"] + 1
        axis.errorbar(
            x_values,
            summary["mean"] * 100,
            yerr=summary["ci95"] * 100,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.8,
            markersize=4.5,
            capsize=2.5,
            label=f"Recall@{cutoff}",
        )
    axis.set_xlabel("Popularity Decile (1=Rare to 10=Famous)", fontweight="bold")
    axis.set_ylabel("Target Article Retrieved (%)", fontweight="bold")
    axis.set_ylim(0, 100)
    axis.set_title("BM25+ Candidate Retrieval Remains Broad", fontweight="bold")
    _style_axis(axis)
    axis.legend(title="Candidate cutoff", frameon=False, loc="lower center", ncol=4,
                bbox_to_anchor=(0.5, -0.31))
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_lexical_competition(diagnostics: pd.DataFrame, output_path: Path) -> None:
    """Plot non-target chunks within 5% of the best target score by decile."""
    summary = _summarise(diagnostics, "near_ties_5pct")
    figure, axis = plt.subplots(figsize=(7.4, 4.6))
    axis.errorbar(
        summary["decile"] + 1,
        summary["mean"],
        yerr=summary["ci95"],
        color="#7B3294",
        marker="o",
        linewidth=1.8,
        markersize=4.5,
        capsize=2.5,
    )
    axis.set_xlabel("Popularity Decile (1=Rare to 10=Famous)", fontweight="bold")
    axis.set_ylabel("Non-target Chunks Within 5% of Best Target Score", fontweight="bold")
    axis.set_title("BM25+ Lexical Competition Increases with Popularity", fontweight="bold")
    _style_axis(axis)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_target_lexical_evidence(metrics: pd.DataFrame, output_path: Path) -> None:
    """Plot query IDF and maximum target-chunk TF as paper-figure panels."""
    idf_summary = _summarise(metrics, "query_idf_mean")
    tf_summary = _summarise(metrics, "best_target_chunk_query_tf")
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), sharex=True)
    panels = [
        (axes[0], idf_summary, "#0072B2", "A. Mean Query-Term IDF", "Mean Query-Term IDF"),
        (axes[1], tf_summary, "#D55E00", "B. Maximum Target-Chunk TF", "Maximum Target-Chunk Query-Term TF"),
    ]
    for axis, summary, color, title, ylabel in panels:
        axis.errorbar(
            summary["decile"] + 1,
            summary["mean"],
            yerr=summary["ci95"],
            color=color,
            marker="o",
            linewidth=1.8,
            markersize=4.5,
            capsize=2.5,
        )
        axis.set_title(title, fontweight="bold")
        axis.set_xlabel("Popularity Decile (1=Rare to 10=Famous)", fontweight="bold")
        axis.set_ylabel(ylabel, fontweight="bold")
        _style_axis(axis)
    # A zero baseline makes the modest absolute IDF change visually proportionate.
    axes[0].set_ylim(bottom=0)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Generate all 60k BM25+ candidate and lexical diagnostic figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions-path",
        type=Path,
        default=DATA_DIR / "wiki_full_bil" / "all_qa_60k_balanced" / "cyro_qa_cache.parquet",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=DATA_DIR / "wiki_full_bil" / "bm25_bm25plus_recursive",
    )
    parser.add_argument(
        "--corpus-path",
        type=Path,
        default=DATA_DIR / "wiki_full_bil" / "bm25_bm25plus_recursive" / "corpus.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "paper_figures")
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=DATA_DIR / "wiki_full_bil" / "all_qa_60k_balanced",
        help="Directory where reusable per-question diagnostic metrics are stored.",
    )
    parser.add_argument(
        "--reuse-cached-metrics",
        action="store_true",
        help="Regenerate figures from saved metrics instead of re-running BM25+ analysis.",
    )
    parser.add_argument(
        "--lexical-only",
        action="store_true",
        help="Generate only the query-IDF and target-chunk-TF figure without retrieval.",
    )
    parser.add_argument("--depth", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--chunk-batch-size", type=int, default=2_000)
    parser.add_argument(
        "--checkpoint-every-batches",
        type=int,
        default=5,
        help="Persist retrieval diagnostics every N batches for resumable execution.",
    )
    parser.add_argument("--exclude-datasets", nargs="*", default=[])
    args = parser.parse_args()
    if args.depth < 100:
        raise ValueError("--depth must be at least 100 for Recall@100")
    if args.batch_size <= 0 or args.chunk_batch_size <= 0:
        raise ValueError("--batch-size and --chunk-batch-size must be positive")
    if args.checkpoint_every_batches <= 0:
        raise ValueError("--checkpoint-every-batches must be positive")

    required_paths = [args.questions_path, args.index_path, args.corpus_path]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Required input(s) not found: {missing_paths}")

    questions = pd.read_parquet(args.questions_path)
    required_columns = {"question_id", "question_text", "wikipedia_id", "pop_decile_chunk_weighted"}
    missing_columns = required_columns - set(questions.columns)
    if missing_columns:
        raise KeyError(f"Questions missing required columns: {sorted(missing_columns)}")
    if args.exclude_datasets:
        questions = questions[~questions["dataset"].isin(args.exclude_datasets)].copy()
    questions = questions[questions["pop_decile_chunk_weighted"].between(0, 9)].copy()
    retrieval_metrics_path = args.metrics_dir / "bm25_60k_retrieval_diagnostics.parquet"
    lexical_metrics_path = args.metrics_dir / "bm25_60k_lexical_metrics.parquet"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.lexical_only:
        if lexical_metrics_path.exists():
            lexical_metrics = pd.read_parquet(lexical_metrics_path)
        else:
            lexical_metrics = compute_target_lexical_metrics(
                questions,
                index_path=args.index_path,
                corpus_path=args.corpus_path,
                chunk_batch_size=args.chunk_batch_size,
            )
            args.metrics_dir.mkdir(parents=True, exist_ok=True)
            lexical_metrics.to_parquet(lexical_metrics_path, index=False)
            logger.info("Saved reusable lexical metrics to %s", lexical_metrics_path)
        plot_target_lexical_evidence(
            lexical_metrics,
            args.output_dir / "bm25_target_lexical_evidence_by_decile_60k_balanced.png",
        )
        return

    if args.reuse_cached_metrics:
        missing_metrics = [
            str(path)
            for path in (retrieval_metrics_path, lexical_metrics_path)
            if not path.exists()
        ]
        if missing_metrics:
            raise FileNotFoundError(f"Cached metrics not found: {missing_metrics}")
        retrieval_diagnostics = pd.read_parquet(retrieval_metrics_path)
        lexical_metrics = pd.read_parquet(lexical_metrics_path)
    else:
        logger.info("Analysing %d questions at BM25+ depth %d", len(questions), args.depth)
        rag_service = BM25RagService(method="bm25+")
        rag_service.load_index(args.index_path)
        retrieval_diagnostics = compute_retrieval_diagnostics(
            questions,
            rag_service=rag_service,
            depth=args.depth,
            batch_size=args.batch_size,
            checkpoint_path=retrieval_metrics_path,
            checkpoint_every_batches=args.checkpoint_every_batches,
        )
        lexical_metrics = compute_target_lexical_metrics(
            questions,
            index_path=args.index_path,
            corpus_path=args.corpus_path,
            chunk_batch_size=args.chunk_batch_size,
        )
        args.metrics_dir.mkdir(parents=True, exist_ok=True)
        retrieval_diagnostics.to_parquet(retrieval_metrics_path, index=False)
        lexical_metrics.to_parquet(lexical_metrics_path, index=False)
        logger.info("Saved reusable metrics to %s", args.metrics_dir)
    plot_candidate_recall(
        retrieval_diagnostics,
        args.output_dir / "bm25_candidate_recall_by_decile_60k_balanced.png",
    )
    plot_lexical_competition(
        retrieval_diagnostics,
        args.output_dir / "bm25_lexical_competition_by_decile_60k_balanced.png",
    )
    plot_target_lexical_evidence(
        lexical_metrics,
        args.output_dir / "bm25_target_lexical_evidence_by_decile_60k_balanced.png",
    )
    logger.info("Saved 60k BM25+ candidate and lexical diagnostic figures to %s", args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
