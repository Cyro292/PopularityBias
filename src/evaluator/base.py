"""Base dataclasses and abstract evaluator for the evaluation pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from langchain.schema import Document

from src.llm.base import LLMBase


@dataclass
class EvaluationObjects:
    id: str
    question: str
    proposed_answer: str
    answer: str
    retrieved_docs: list[Document]
    page_content: str


@dataclass
class EvaluationResult:
    id: str
    question: str
    answer: str
    proposed_answer: str
    evaluation_score: float | str | bool
    reasoning: str


class EvaluatorBase:
    def __init__(self, evaluation_service: LLMBase) -> None:
        self.evaluation_service = evaluation_service

    def evaluate(self, evaluation_objects: list[EvaluationObjects]) -> list[EvaluationResult]:
        """Evaluate the given evaluation objects and return a list of results."""
        raise NotImplementedError("evaluate method is not implemented")
