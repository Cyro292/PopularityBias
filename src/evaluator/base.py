"""Base dataclasses and abstract evaluator for the evaluation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain.schema import Document

from src.llm.base import LLMBase


@dataclass
class EvaluationObjects:
    id: str
    question: str
    proposed_answer: str
    page_content: str
    answers: list[str]
    retrieved_docs: list[Document]
    metadata: dict | None = None


@dataclass
class EvaluationResult:
    id: str
    question: str
    answers: list[str]
    proposed_answer: str
    evaluation_score: float | str | bool
    reasoning: str
    metadata: dict | None = None

class EvaluatorBase:
    def __init__(self) -> None:
        pass

    def evaluate(self, evaluation_objects: list[EvaluationObjects], *, checkpoint_path: str | Path | None = None) -> list[EvaluationResult]:
        """Evaluate the given evaluation objects and return a list of results."""
        raise NotImplementedError("evaluate method is not implemented")
