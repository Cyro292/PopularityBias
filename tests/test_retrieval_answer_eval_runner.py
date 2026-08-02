"""Tests for answer-containment retrieval evaluation."""

from __future__ import annotations

import pandas as pd
import pytest

from src.process.pipeline.retrieval_answer_eval_runner import (
    evaluate_retrieval_answers,
    find_answer_rank,
    parse_answer_texts,
)


def test_find_answer_rank_uses_answer_content_not_document_id() -> None:
    chunks = ["An unrelated passage.", "The capital of France is Paris."]

    assert find_answer_rank(chunks, ["PARIS"]) == 2
    assert find_answer_rank(chunks, ["London"]) is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (["Paris", "Paris", ""], ["Paris"]),
        ('["Paris", "City of Paris"]', ["Paris", "City of Paris"]),
        ("Paris", ["Paris"]),
        (None, []),
        (pd.NA, []),
    ],
)
def test_parse_answer_texts(value: object, expected: list[str]) -> None:
    assert parse_answer_texts(value) == expected


def test_evaluate_retrieval_answers_excludes_questions_without_answers() -> None:
    retrieved = pd.DataFrame(
        {
            "question_id": ["q1", "q1", "q2"],
            "doc_rank": [0, 1, 0],
            "page_content": ["wrong", "Contains the right answer", "anything"],
        }
    )
    questions = pd.DataFrame(
        {
            "question_id": ["q1", "q2", "q3"],
            "answer_texts": [["right answer"], [], ["missing"]],
            "dataset": ["nq", "nq", "trivia"],
        }
    )

    result = evaluate_retrieval_answers(retrieved, questions, k_values=[1, 2])
    by_id = result.set_index("question_id")

    assert by_id.loc["q1", "answer_rank"] == 2
    assert by_id.loc["q1", "answer_recall@1"] == 0.0
    assert by_id.loc["q1", "answer_recall@2"] == 1.0
    assert bool(by_id.loc["q2", "is_evaluable"]) is False
    assert pd.isna(by_id.loc["q2", "answer_recall@2"])
    assert by_id.loc["q3", "answer_reciprocal_rank"] == 0.0


def test_evaluate_retrieval_answers_caps_matching_at_largest_k() -> None:
    retrieved = pd.DataFrame(
        {
            "question_id": ["q1", "q1", "q1"],
            "doc_rank": [0, 1, 10],
            "page_content": ["wrong", "also wrong", "contains answer"],
        }
    )
    questions = pd.DataFrame(
        {"question_id": ["q1"], "answer_texts": [["answer"]]}
    )

    result = evaluate_retrieval_answers(retrieved, questions, k_values=[1, 2])

    assert pd.isna(result.loc[0, "answer_rank"])
    assert result.loc[0, "answer_reciprocal_rank"] == 0.0
