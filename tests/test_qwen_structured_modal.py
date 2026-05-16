"""Integration test — Qwen structured output via Modal.

Requires Modal to be authenticated and the Qwen app deployed.
Run with:
    pytest tests/test_qwen_structured_modal.py -v -s
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dotenv
dotenv.load_dotenv()

from src.llm.qwenLLMService import QwenLLMService
from src.evaluator.binary_evaluator import BinaryJudgement


@pytest.mark.integration
def test_qwen_structured_correct_answer():
    svc = QwenLLMService()

    prompt = (
        "Gold standard document: Paris is the capital of France.\n"
        "Question: What is the capital of France?\n"
        "Proposed answer: Paris"
    )

    results = svc.batch_generate_structured([prompt], BinaryJudgement)

    print("\nRaw result:", results[0])
    assert len(results) == 1
    assert results[0] is not None, "Parse returned None — check LLM output"
    assert isinstance(results[0], BinaryJudgement)
    assert results[0].verdict is True, f"Expected True, got: {results[0]}"
    print(f"verdict={results[0].verdict}, reasoning={results[0].reasoning}")


@pytest.mark.integration
def test_qwen_structured_wrong_answer():
    svc = QwenLLMService()

    prompt = (
        "Gold standard document: Paris is the capital of France.\n"
        "Question: What is the capital of France?\n"
        "Proposed answer: Berlin"
    )

    results = svc.batch_generate_structured([prompt], BinaryJudgement)

    print("\nRaw result:", results[0])
    assert len(results) == 1
    assert results[0] is not None
    assert isinstance(results[0], BinaryJudgement)
    assert results[0].verdict is False, f"Expected False, got: {results[0]}"
    print(f"verdict={results[0].verdict}, reasoning={results[0].reasoning}")
