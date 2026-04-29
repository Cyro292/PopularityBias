"""Substring match evaluator: marks a prediction correct if any gold answer is a substring of the prediction."""

from __future__ import annotations

import logging
from pathlib import Path

from tqdm import tqdm

from src.evaluator.base import EvaluationObjects, EvaluationResult, EvaluatorBase
from src.llm.base import LLMBase

logger = logging.getLogger(__name__)


class SubstringEvaluator(EvaluatorBase):
    """Exact substring match evaluator (no LLM required).

    A prediction is marked correct (``evaluation_score=True``) if any gold
    answer string from ``EvaluationObjects.answers`` appears as a case-insensitive
    substring of ``proposed_answer``.

    This evaluator does **not** call the LLM service; ``evaluation_service`` is
    accepted only to satisfy the :class:`~src.evaluator.base.EvaluatorBase`
    interface and may be ``None``.

    Args:
        evaluation_service: Unused — pass ``None`` or any :class:`~src.llm.base.LLMBase`
            instance.
        case_sensitive: If ``True``, comparisons are case-sensitive.
            Defaults to ``False``.

    Example::

        from src.evaluator.substring_evaluator import SubstringEvaluator
        from src.evaluator.base import EvaluationObjects

        evaluator = SubstringEvaluator(evaluation_service=None)

        objects = [
            EvaluationObjects(
                id="q1",
                question="What is the capital of France?",
                proposed_answer="The capital is Paris, located in northern France.",
                page_content="",
                answers=["Paris"],
                retrieved_docs=[],
            )
        ]
        results = evaluator.evaluate(objects)
        print(results[0].evaluation_score)  # True
    """

    def __init__(
        self,
        *,
        case_sensitive: bool = False,
    ) -> None:
        # EvaluatorBase expects evaluation_service; pass None-safe value
        super().__init__()  # type: ignore[arg-type]
        self.case_sensitive = case_sensitive

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _is_correct(self, proposed_answer: str, gold_answers: list[str]) -> bool:
        """Return True if any gold answer is a substring of proposed_answer.

        Args:
            proposed_answer: The model-generated answer string.
            gold_answers: List of ground-truth answer strings.

        Returns:
            ``True`` if at least one gold answer appears verbatim (subject to
            case folding) inside ``proposed_answer``.
        """
        # Handle non-string proposed_answer
        if not isinstance(proposed_answer, str):
            logger.warning("Proposed answer is not a string (type=%s), converting", type(proposed_answer).__name__)
            proposed_answer = str(proposed_answer)
        
        haystack = proposed_answer if self.case_sensitive else proposed_answer.lower()
        
        for gold in gold_answers:
            # Handle non-string gold answers
            if not isinstance(gold, str):
                logger.warning("Gold answer is not a string (type=%s), converting", type(gold).__name__)
                gold = str(gold)
            
            needle = gold if self.case_sensitive else gold.lower()
            if needle and needle in haystack:
                return True
        return False

    # ── Public interface ──────────────────────────────────────────────────────

    def evaluate(
        self,
        evaluation_objects: list[EvaluationObjects],
        *,
        checkpoint_path: str | Path | None = None,  # noqa: ARG002 — kept for interface parity
    ) -> list[EvaluationResult]:
        """Evaluate a list of question–answer pairs via substring matching.

        No LLM calls are made.  Each item is scored independently and
        synchronously.  All items are returned; no items are silently dropped.

        Args:
            evaluation_objects: List of items to evaluate.
            checkpoint_path: Ignored — present for interface compatibility
                with other evaluators.

        Returns:
            List of :class:`~src.evaluator.base.EvaluationResult` instances,
            one per input item.  ``evaluation_score`` is ``True`` when any gold
            answer is found as a substring of the proposed answer, ``False``
            otherwise.  ``reasoning`` contains a short human-readable
            explanation.
        """
        results: list[EvaluationResult] = []
        
        logger.info("Starting substring evaluation for %d items", len(evaluation_objects))

        for obj in tqdm(evaluation_objects, desc="Evaluating (substring match)", unit="item"):
            try:
                correct = self._is_correct(obj.proposed_answer, obj.answers)
                reasoning = (
                    f"Gold answer matched as substring of proposed answer."
                    if correct
                    else f"No gold answer found as substring of proposed answer."
                )
                results.append(
                    EvaluationResult(
                        id=obj.id,
                        question=obj.question,
                        answers=obj.answers,
                        proposed_answer=obj.proposed_answer,
                        evaluation_score=correct,
                        reasoning=reasoning,
                        metadata=obj.metadata,
                    )
                )
            except Exception as e:
                logger.warning("[%s] Result assembly failed, skipping: %s", obj.id, e)

        logger.info("✓ Completed %d evaluations (%d correct, %d incorrect)", 
                    len(results),
                    sum(1 for r in results if r.evaluation_score),
                    sum(1 for r in results if not r.evaluation_score))

        return results
