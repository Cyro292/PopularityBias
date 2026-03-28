"""Binary evaluator: judges whether a proposed answer is relevant to the question."""

from __future__ import annotations

import logging
from typing import Literal, Type

from pydantic import BaseModel, Field

from src.evaluator.base import EvaluationObjects, EvaluationResult, EvaluatorBase
from src.llm.base import LLMBase

logger = logging.getLogger(__name__)

# ── Structured output schema ──────────────────────────────────────────────────

class BinaryJudgement(BaseModel):
    """Structured output returned by the LLM judge."""

    verdict: bool = Field(
        description=(
            "True if the proposed answer is relevant and responsive to the question, "
            "False otherwise."
        )
    )
    reasoning: str = Field(
        description="One or two sentences explaining the verdict."
    )


# ── Prompt ────────────────────────────────────────────────────────────────────

_PROMPT_TEMPLATE = """\
You are a strict relevance judge. Your task is to decide whether a proposed answer \
adequately addresses the given question.

Rules:
- Answer TRUE if the proposed answer is relevant and directly responsive to the question, \
even if it is incomplete or partially correct.
- Answer FALSE if the proposed answer is off-topic, nonsensical, or fails to address \
the question at all.
- Do NOT require the proposed answer to be factually correct against the reference answer; \
judge relevance only.

Question: {question}
Proposed answer: {proposed_answer}

Respond with a JSON object containing:
  "verdict": <true|false>
  "reasoning": "<one or two sentences>"
"""


# ── Evaluator ─────────────────────────────────────────────────────────────────

class BinaryEvaluator(EvaluatorBase):
    """LLM-based binary relevance evaluator.

    For each :class:`~src.evaluator.base.EvaluationObjects` item, asks an LLM
    judge whether the *proposed_answer* is relevant to the *question*.  The
    result is a :class:`~src.evaluator.base.EvaluationResult` whose
    ``evaluation_score`` is a :class:`bool` (``True`` = relevant,
    ``False`` = not relevant).

    Args:
        evaluation_service: Any :class:`~src.llm.base.LLMBase` implementation
            used to call the LLM judge.
        prompt_template: Optional custom prompt template.  Must contain
            ``{question}`` and ``{proposed_answer}`` placeholders.

    Example::

        from src.llm.openAi_service import OpenAIService
        from src.evaluator.binary_evaluator import BinaryEvaluator
        from src.evaluator.base import EvaluationObjects

        service = OpenAIService(model_name="gpt-4o-mini")
        evaluator = BinaryEvaluator(evaluation_service=service)

        objects = [
            EvaluationObjects(
                id="q1",
                question="What is the capital of France?",
                proposed_answer="Paris is the capital of France.",
                answer="Paris",
                retrieved_docs=[],
            )
        ]
        results = evaluator.evaluate(objects)
        print(results[0].evaluation_score)  # True
    """

    def __init__(
        self,
        evaluation_service: LLMBase,
        prompt_template: str = _PROMPT_TEMPLATE,
    ) -> None:
        super().__init__(evaluation_service=evaluation_service)
        self.prompt_template = prompt_template

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_prompt(self, obj: EvaluationObjects) -> str:
        """Render the prompt template for a single evaluation object."""
        return self.prompt_template.format(
            question=obj.question,
            proposed_answer=obj.proposed_answer,
        )

    # ── Public interface ──────────────────────────────────────────────────────

    def evaluate(self, evaluation_objects: list[EvaluationObjects]) -> list[EvaluationResult]:
        """Evaluate a list of question–answer pairs for relevance.

        All prompts are dispatched as a single batch via
        :meth:`~src.llm.base.LLMBase.batch_generate_structured`, so requests
        are sent concurrently (subject to any rate limiter on the LLM service).
        Failures on individual items are logged and those items are omitted
        from the returned list.

        Args:
            evaluation_objects: List of items to evaluate.

        Returns:
            List of :class:`~src.evaluator.base.EvaluationResult` instances,
            one per successfully evaluated input.  Items that raised an
            exception are omitted and logged as warnings.
        """
        if not evaluation_objects:
            return []

        prompts = [self._build_prompt(obj) for obj in evaluation_objects]

        try:
            judgements = self.evaluation_service.batch_generate_structured(
                prompts, BinaryJudgement
            )
        except Exception as e:
            logger.error("batch_generate_structured failed: %s", e)
            raise

        results: list[EvaluationResult] = []
        for obj, judgement in zip(evaluation_objects, judgements):
            try:
                results.append(
                    EvaluationResult(
                        id=obj.id,
                        question=obj.question,
                        answer=obj.answer,
                        proposed_answer=obj.proposed_answer,
                        evaluation_score=judgement.verdict,
                        reasoning=judgement.reasoning,
                    )
                )
                logger.info(
                    "[%s] verdict=%s — %s", obj.id, judgement.verdict, judgement.reasoning
                )
            except Exception as e:
                logger.warning("[%s] Result assembly failed, skipping: %s", obj.id, e)

        return results
