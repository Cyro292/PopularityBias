"""Unit tests for LLMBase._parse_structured and _structured_prompt.

These run fully locally — no Modal, no GPU required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel, Field
from src.llm.base import LLMBase


class BinaryJudgement(BaseModel):
    verdict: bool = Field(description="True if correct, False otherwise.")
    reasoning: str = Field(description="One sentence explanation.")


# ── _parse_structured ─────────────────────────────────────────────────────────

def test_parse_strict_json():
    raw = '{"verdict": true, "reasoning": "The answer is correct."}'
    result = LLMBase._parse_structured(raw, BinaryJudgement)
    assert result.verdict is True
    assert "correct" in result.reasoning


def test_parse_json_embedded_in_text():
    raw = 'Sure! Here is my answer: {"verdict": false, "reasoning": "Wrong answer."} Hope that helps.'
    result = LLMBase._parse_structured(raw, BinaryJudgement)
    assert result.verdict is False


def test_parse_regex_fallback_true():
    raw = 'I think the verdict is "verdict": true because it seems right'
    result = LLMBase._parse_structured(raw, BinaryJudgement)
    assert result.verdict is True


def test_parse_regex_fallback_false():
    raw = 'some broken json... "verdict": false ... "reasoning": "Not quite right"'
    result = LLMBase._parse_structured(raw, BinaryJudgement)
    assert result.verdict is False


def test_parse_raises_on_garbage():
    raw = "I have no idea what you are asking me to do here."
    with pytest.raises(ValueError):
        LLMBase._parse_structured(raw, BinaryJudgement)


# ── _structured_prompt ────────────────────────────────────────────────────────

def test_structured_prompt_contains_schema():
    prompt = "Is Paris the capital of France?"
    wrapped = LLMBase._structured_prompt(prompt, BinaryJudgement)
    assert "Is Paris the capital of France?" in wrapped
    assert "verdict" in wrapped
    assert "reasoning" in wrapped
    assert "JSON" in wrapped


def test_structured_prompt_roundtrip():
    """A well-behaved LLM response to a structured prompt should parse cleanly."""
    prompt = "Is Berlin the capital of Germany?"
    wrapped = LLMBase._structured_prompt(prompt, BinaryJudgement)
    # Simulate a model that responds correctly
    simulated_response = '{"verdict": true, "reasoning": "Berlin is indeed the capital."}'
    result = LLMBase._parse_structured(simulated_response, BinaryJudgement)
    assert result.verdict is True
