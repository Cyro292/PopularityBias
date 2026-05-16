"""Binary evaluator: judges whether a proposed answer is relevant to the question."""

from __future__ import annotations

import logging
from pathlib import Path

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
You are a strict factual judge. Your task is to decide whether a proposed answer \
correctly answers the given question, using the gold standard document as the source of truth.

Rules:
- Answer TRUE if the proposed answer is factually correct and directly responsive to the \
question according to the gold standard document, even if it is incomplete or phrased differently.
- Answer FALSE if the proposed answer is factually wrong, off-topic, nonsensical, or \
contradicts the gold standard document.
- Base your judgement solely on the gold standard document — do not rely on outside knowledge.

Gold standard document:
{page_content}

Question: {question}
Proposed answer: {proposed_answer}

Respond with a JSON object containing:
  "verdict": <true|false>
  "reasoning": "<one or two sentences>"
"""


# ── Evaluator ─────────────────────────────────────────────────────────────────

class BinaryEvaluator(EvaluatorBase):
    """LLM-based binary factual evaluator.

    For each :class:`~src.evaluator.base.EvaluationObjects` item, asks an LLM
    judge whether the *proposed_answer* is factually correct given the gold
    standard document (``page_content``).  The result is a
    :class:`~src.evaluator.base.EvaluationResult` whose ``evaluation_score``
    is a :class:`bool` (``True`` = correct, ``False`` = incorrect).

    ``page_content`` on each :class:`~src.evaluator.base.EvaluationObjects`
    must be non-empty; it is provided automatically by the pipeline when a
    :class:`~src.corpus_handler.base.CorpusHandler` is supplied to the
    question handler.

    Args:
        evaluation_service: Any :class:`~src.llm.base.LLMBase` implementation
            used to call the LLM judge.
        prompt_template: Optional custom prompt template.  Must contain
            ``{page_content}``, ``{question}``, and ``{proposed_answer}``
            placeholders.

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
        super().__init__()
        self.evaluation_service = evaluation_service
        self.prompt_template = prompt_template

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_prompt(self, obj: EvaluationObjects) -> str:
        """Render the prompt template for a single evaluation object."""
        return self.prompt_template.format(
            page_content=obj.page_content,
            question=obj.question,
            proposed_answer=obj.proposed_answer,
        )

    # ── Public interface ──────────────────────────────────────────────────────

    def evaluate(
        self,
        evaluation_objects: list[EvaluationObjects],
        *,
        checkpoint_path: str | Path | None = None,
    ) -> list[EvaluationResult]:
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

        missing = sum(1 for obj in evaluation_objects if not obj.page_content)
        if missing:
            logger.warning(
                "%d / %d evaluation objects have empty page_content — "
                "ensure a CorpusHandler was provided to the question handler.",
                missing, len(evaluation_objects),
            )

        prompts = [self._build_prompt(obj) for obj in evaluation_objects]

        try:
            raw_texts = self.evaluation_service.batch_generate(
                prompts, checkpoint_path=checkpoint_path
            )
        except Exception as e:
            logger.error("batch_generate failed: %s", e)
            raise

        judgements: list[BinaryJudgement] = []
        for raw in raw_texts:
            try:
                judgements.append(self.evaluation_service._parse_structured(raw, BinaryJudgement))
            except Exception as e:
                logger.warning("Failed to parse judgement, defaulting to False: %s", e)
                judgements.append(BinaryJudgement(verdict=False, reasoning="parse error"))

        results: list[EvaluationResult] = []
        for obj, judgement in zip(evaluation_objects, judgements):
            try:
                results.append(
                    EvaluationResult(
                        id=obj.id,
                        question=obj.question,
                        answers=obj.answers,
                        proposed_answer=obj.proposed_answer,
                        evaluation_score=judgement.verdict,
                        reasoning=judgement.reasoning,
                        metadata=obj.metadata,
                    )
                )
                logger.info(
                    "[%s] verdict=%s — %s", obj.id, judgement.verdict, judgement.reasoning
                )
            except Exception as e:
                logger.warning("[%s] Result assembly failed, skipping: %s", obj.id, e)

        return results
