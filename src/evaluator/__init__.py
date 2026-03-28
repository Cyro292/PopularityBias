"""Evaluator subpackage — LLM-based answer evaluation utilities."""

from __future__ import annotations

from src.evaluator.base import EvaluationObjects, EvaluationResult, EvaluatorBase
from src.evaluator.binary_evaluator import BinaryEvaluator

__all__ = [
    "EvaluationObjects",
    "EvaluationResult",
    "EvaluatorBase",
    "BinaryEvaluator",
]
