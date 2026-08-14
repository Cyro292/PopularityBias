"""Plot a compact 2x2 summary of the BM25 popularity-ranking failure.

The figure combines the target-side Hit@1 result with direct BM25 score
diagnostics: target candidate coverage, target rank, target-versus-distractor
margin, near-tie density, and discriminability of query terms shared with the
target title.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import DATA_DIR

logger = logging.getLogger(__name__)


def _token_sets(tokenized: object) -> list[set[str]]:
    """Convert a bm25s tokenized batch into one set of terms per input text."""
    vocabulary = tokenized.vocab
    id_to_term = {term_id: term for term, term_id in vocabulary.items()}
    return [{id_to_term[token_id] for token_id in token_ids} for token_ids in tokenized.ids]


def compute_title_anchor_stats(
    questions: pd.DataFrame,
    *,
    index_path: Path,
) -> pd.DataFrame:
    """Compute BM25 IDF mass for query terms shared with the target title.

    Args:
        questions: Evaluation questions with target titles and popularity deciles.
        index_path: Persisted BM25 index that defines the term IDF values.

    Returns:
        Per-decile means for title-term coverage and shared-title IDF mass.
    """
    import bm25s
    import Stemmer

    retriever = bm25s.BM25.load(str(index_path), load_corpus=False, mmap=True)
    document_frequencies = np.diff(retriever.scores["indptr"])
    document_count = int(retriever.scores["num_docs"])
    stemmer = Stemmer.Stemmer("english")
    query_terms = _token_sets(bm25s.tokenize(
        questions["question_text"].tolist(),
        stopwords="en",
        stemmer=stemmer,
        show_progress=False,
    ))
    title_terms = _token_sets(bm25s.tokenize(
        questions["wikipedia_title"].tolist(),
        stopwords="en",
        stemmer=stemmer,
        show_progress=False,
    ))

    rows: list[dict[str, float | int]] = []
    for decile, query_set, title_set in zip(
        questions["pop_decile_chunk_weighted"], query_terms, title_terms
    ):
        shared_terms = query_set & title_set
        shared_idf = sum(
            float(np.log((document_count + 1) / document_frequencies[retriever.vocab_dict[term]]))
            for term in shared_terms
            if term in retriever.vocab_dict
        )
        rows.append({
            "decile": int(decile),
            "title_mentioned": float(bool(shared_terms)),
            "shared_title_idf_sum": shared_idf,
        })

    return pd.DataFrame(rows).groupby("decile", as_index=False).mean()


def plot_diagnostics(
    *,
    lexical_factors: pd.DataFrame,
    competition: pd.DataFrame,
    title_stats: pd.DataFrame,
    output_path: Path,
) -> None:
    """Render the compact four-panel BM25 ranking-failure figure.

    Args:
        lexical_factors: Per-query target lexical-factor diagnostics.
        competition: Per-query direct BM25 score diagnostics.
        title_stats: Per-decile query-title discriminability statistics.
        output_path: PNG destination.
    """
    import matplotlib.pyplot as plt

    hit_rates = lexical_factors.groupby("decile", as_index=False)["hit_at_1"].mean()
    score_stats = competition.groupby("decile", as_index=False).agg(
        target_top100=("target_in_depth", "mean"),
        target_rank=("target_rank", "mean"),
        target_margin=("target_margin", "mean"),
        near_ties_5pct=("near_ties_5pct", "mean"),
    )
    stats = hit_rates.merge(score_stats, on="decile").merge(title_stats, on="decile")
    deciles = stats["decile"].to_numpy() + 1

    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    figure, axes = plt.subplots(2, 2, figsize=(10, 6.8), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(deciles, stats["hit_at_1"] * 100, "o-", color="#C2410C", label="Hit@1")
    ax.plot(deciles, stats["target_top100"] * 100, "s--", color="#2563EB", label="Target in top 100")
    ax.set(title="A. Candidate Retrieval vs. Top Rank", xlabel="Popularity decile", ylabel="Queries (%)")
    ax.set_xticks(deciles)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="best")

    ax = axes[0, 1]
    bars = ax.bar(deciles, stats["target_rank"], color="#60A5FA", label="Mean target rank")
    ax.set(title="B. Target Is Found but Ranked Lower", xlabel="Popularity decile", ylabel="Mean target rank")
    ax.set_xticks(deciles)
    ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
    margin_ax = ax.twinx()
    margin_ax.plot(deciles, stats["target_margin"], "o-", color="#B91C1C", label="Target margin")
    margin_ax.axhline(0, color="#475569", linewidth=0.8, linestyle=":")
    margin_ax.set_ylabel("Target score minus best distractor")
    handles, labels = ax.get_legend_handles_labels()
    margin_handles, margin_labels = margin_ax.get_legend_handles_labels()
    ax.legend(handles + margin_handles, labels + margin_labels, frameon=False, loc="upper left")

    ax = axes[1, 0]
    bars = ax.bar(deciles, stats["near_ties_5pct"], color="#7C3AED")
    ax.set(
        title="C. Lexical Competition Increases",
        xlabel="Popularity decile",
        ylabel="Non-target chunks within 5% of target",
    )
    ax.set_xticks(deciles)
    ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)

    ax = axes[1, 1]
    bars = ax.bar(deciles, stats["shared_title_idf_sum"], color="#0F766E", label="Shared title-term IDF")
    ax.set(
        title="D. Entity Terms Become Less Distinctive",
        xlabel="Popularity decile",
        ylabel="BM25 IDF mass of query-title terms",
    )
    ax.set_xticks(deciles)
    ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
    mention_ax = ax.twinx()
    mention_ax.plot(deciles, stats["title_mentioned"] * 100, "D--", color="#F59E0B", label="Title mentioned")
    mention_ax.set_ylabel("Queries mentioning title (%)")
    mention_ax.set_ylim(0, 100)
    handles, labels = ax.get_legend_handles_labels()
    mention_handles, mention_labels = mention_ax.get_legend_handles_labels()
    ax.legend(handles + mention_handles, labels + mention_labels, frameon=False, loc="upper right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(figure)
    logger.info("Saved BM25 ranking-failure figure to %s", output_path)


def main() -> None:
    """Load analysis artifacts and create the compact diagnostics figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--factors-path",
        type=Path,
        default=DATA_DIR / "wiki_full_bil" / "all_qa_8k" / "bm25_lexical_factors_v2_retrieved_docs_bm25_plus_chunk_weighted.parquet",
    )
    parser.add_argument(
        "--competition-path",
        type=Path,
        default=DATA_DIR / "wiki_full_bil" / "all_qa_8k" / "bm25_competition_top100.parquet",
    )
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
        default=ROOT_DIR / "paper_figures" / "bm25_ranking_failure_diagnostics.png",
    )
    args = parser.parse_args()

    required_paths = [args.factors_path, args.competition_path, args.questions_path, args.index_path]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Required analysis artifact(s) not found: {missing_paths}")

    questions = pd.read_parquet(args.questions_path)
    questions = questions[questions["dataset"] != "hotpot_qa"].copy()
    lexical_factors = pd.read_parquet(args.factors_path)
    competition = pd.read_parquet(args.competition_path)
    if len(questions) != len(lexical_factors) or len(questions) != len(competition):
        raise ValueError(
            "Questions, lexical factors, and competition diagnostics must use the same cohort. "
            f"Received {len(questions)}, {len(lexical_factors)}, and {len(competition)} rows."
        )

    title_stats = compute_title_anchor_stats(questions, index_path=args.index_path)
    plot_diagnostics(
        lexical_factors=lexical_factors,
        competition=competition,
        title_stats=title_stats,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
