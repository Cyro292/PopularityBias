"""Analyze answer-bearing chunks and popularity preference of non-answer chunks."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.process.pipeline.retrieval_answer_eval_runner import (
    _normalise_text,
    parse_answer_texts,
)

logger = logging.getLogger(__name__)


def load_answer_chunk_matches(
    retrieval_path: Path,
    questions: pd.DataFrame,
    *,
    k: int,
    excluded_datasets: Sequence[str] = ("fever", "hotpot_qa", "trex"),
    decile_column: str = "pop_decile_unweighted",
) -> pd.DataFrame:
    """Load top-k chunks and mark literal answer-alias containment.

    Args:
        retrieval_path: Retrieval checkpoint containing chunk text and metadata.
        questions: QA metadata containing answer aliases and popularity fields.
        k: Number of retrieved chunks to inspect per question.
        excluded_datasets: Datasets omitted from answer-containment analysis.
        decile_column: Target popularity-decile column to use.

    Returns:
        One row per retrieved chunk with target metadata and ``contains_answer``.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    required_questions = {
        "question_id",
        "answer_texts",
        "dataset",
        "popularity_avg",
        decile_column,
    }
    missing_questions = required_questions - set(questions.columns)
    if missing_questions:
        raise KeyError(f"Questions missing columns: {sorted(missing_questions)}")

    qa = questions.loc[~questions["dataset"].isin(excluded_datasets)].copy()
    qa["question_id"] = qa["question_id"].astype(str)
    qa["answer_aliases"] = qa["answer_texts"].apply(parse_answer_texts)
    qa = qa.loc[qa["answer_aliases"].str.len() > 0].drop_duplicates("question_id")

    retrieved = pd.read_csv(
        retrieval_path,
        usecols=[
            "question_id",
            "doc_rank",
            "page_content",
            "metadata_popularity_avg",
        ],
        dtype={"question_id": str},
    )
    retrieved = retrieved.loc[retrieved["doc_rank"] < k].copy()
    merged = retrieved.merge(
        qa[
            [
                "question_id",
                "dataset",
                "popularity_avg",
                decile_column,
                "answer_aliases",
            ]
        ],
        on="question_id",
        how="inner",
        validate="many_to_one",
    )

    def contains_answer(row: pd.Series) -> bool:
        chunk = _normalise_text(row["page_content"], case_sensitive=False)
        return any(
            _normalise_text(alias, case_sensitive=False) in chunk
            for alias in row["answer_aliases"]
        )

    merged["contains_answer"] = merged.apply(contains_answer, axis=1)
    merged["metadata_popularity_avg"] = pd.to_numeric(
        merged["metadata_popularity_avg"], errors="coerce"
    )
    merged["popularity_avg"] = pd.to_numeric(merged["popularity_avg"], errors="coerce")
    merged["target_decile"] = pd.to_numeric(merged[decile_column], errors="coerce")
    return merged


def summarize_right_chunks(matches: pd.DataFrame) -> pd.DataFrame:
    """Summarize answer-bearing chunk counts per target popularity decile."""
    per_question = matches.groupby(
        ["question_id", "target_decile"], as_index=False
    ).agg(right_chunks=("contains_answer", "sum"))
    return per_question.groupby("target_decile", as_index=False).agg(
        n_questions=("question_id", "size"),
        mean_right_chunks=("right_chunks", "mean"),
        median_right_chunks=("right_chunks", "median"),
        questions_with_answer=("right_chunks", lambda values: (values > 0).mean()),
    )


def summarize_non_answer_preference(matches: pd.DataFrame) -> pd.DataFrame:
    """Summarize whether non-answer chunks are more popular than their targets."""
    wrong = matches.loc[
        ~matches["contains_answer"]
        & matches["metadata_popularity_avg"].notna()
        & matches["popularity_avg"].notna()
    ].copy()
    wrong["more_popular"] = wrong["metadata_popularity_avg"] > wrong["popularity_avg"]
    rows: list[dict[str, float | int]] = []
    for decile in range(10):
        values = wrong.loc[
            wrong["target_decile"] == decile, "more_popular"
        ]
        count = len(values)
        preference = float(values.mean()) if count else np.nan
        ci95 = (
            float(1.96 * np.sqrt(preference * (1.0 - preference) / count))
            if count
            else np.nan
        )
        rows.append(
            {
                "decile": decile,
                "preference": preference,
                "ci95": ci95,
                "n_non_answer_chunks": count,
            }
        )
    return pd.DataFrame(rows)


def compute_non_answer_transition_matrix(
    matches: pd.DataFrame,
    boundaries: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute target-to-retrieved decile percentages for non-answer chunks.

    Args:
        matches: Chunk-level answer matching results.
        boundaries: Eleven popularity boundaries for the selected decile mode.

    Returns:
        Row-normalized 10-by-10 percentage matrix and row sample counts.
    """
    boundary_values = np.asarray(boundaries, dtype=float)
    if boundary_values.shape != (11,):
        raise ValueError("boundaries must contain 11 decile edges")
    wrong = matches.loc[
        ~matches["contains_answer"] & matches["metadata_popularity_avg"].notna()
    ].copy()
    wrong["retrieved_decile"] = np.searchsorted(
        boundary_values[1:-1],
        wrong["metadata_popularity_avg"].to_numpy(),
        side="right",
    )
    counts = pd.crosstab(wrong["target_decile"], wrong["retrieved_decile"])
    counts = counts.reindex(index=range(10), columns=range(10), fill_value=0)
    row_counts = counts.sum(axis=1).to_numpy(dtype=int)
    matrix = counts.to_numpy(dtype=float)
    matrix = np.divide(
        matrix,
        row_counts[:, None],
        out=np.zeros_like(matrix),
        where=row_counts[:, None] > 0,
    )
    return matrix * 100.0, row_counts
