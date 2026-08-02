"""Tests for direct BM25 score-competition diagnostics."""

from __future__ import annotations

import pandas as pd

from src.metrics.bm25_analysis import compute_ranked_score_competition_by_decile


def test_ranked_score_competition_reports_margin_and_near_ties() -> None:
    """Aggregate target-vs-non-target score competition by decile."""
    df = pd.DataFrame({
        "decile": [0, 0],
        "wikipedia_id": ["gold", "gold"],
        "topk_ids": [
            ["gold", "wrong-a", "wrong-b"],
            ["wrong-a", "gold", "wrong-b"],
        ],
        "topk_scores": [
            [4.0, 3.95, 1.0],
            [5.0, 4.9, 4.85],
        ],
    })

    result = compute_ranked_score_competition_by_decile(df, "decile", depth=3, epsilon=0.1)
    row = result.iloc[0]

    assert row["scored_candidate_coverage"] == 1.0
    assert row["target_candidate_coverage"] == 1.0
    assert row["comparable_candidate_coverage"] == 1.0
    assert abs(row["mean_target_margin"] - (-0.025)) < 1e-12
    assert row["mean_near_tie_count"] == 1.5
    assert row["fraction_target_outscored"] == 0.5
