"""Base class for question input sources."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


# ── Question item ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class QuestionItem:
    """A single question together with its ground-truth metadata.

    This is the canonical unit exchanged between a :class:`QuestionInput`
    source and downstream pipeline components (RAG retrieval, evaluation).

    Attributes:
        question_id: Unique identifier for this question (e.g.
            ``"nq_123"`` or ``"syn_3_42"``).
        question_text: The question string passed to the retrieval engine.
        answer_texts: List of acceptable answer strings from the dataset.
            May be empty for synthetic questions.
        wikipedia_id: Wikipedia article ID of the ground-truth document.
            Used to compute Recall@K and MRR.
        wikipedia_title: Human-readable article title (informational only).
        decile: Popularity decile (0–9) of the ground-truth document.
            ``-1`` when not available.
        dataset: Source dataset name (e.g. ``"natural_questions"``).
        page_content: Full text of the ground-truth Wikipedia article as
            stored in the corpus.  Empty string when no
            :class:`~src.corpus_handler.base.CorpusHandler` is available.    """

    question_id: str
    question_text: str
    answer_texts: list[str]
    wikipedia_id: str
    wikipedia_title: str = ""
    decile: int = -1
    dataset: str = ""
    page_content: str = ""


# ── Base ──────────────────────────────────────────────────────────────────────

class QuestionInput:
    """Abstract base for all question input sources.

    Subclasses must implement :meth:`get_items`.  The helper methods
    :meth:`get_questions` and :meth:`__iter__` / :meth:`__len__` are
    implemented here in terms of :meth:`get_items` so concrete classes only
    need one override.

    Design contract
    ---------------
    * :meth:`get_items` may filter, shuffle, or subsample the underlying
      dataset.  The returned list is a stable snapshot — calling the method
      twice on the same instance must return equivalent results.
    * :meth:`get_questions` is a convenience wrapper that extracts only the
      ``question_text`` strings; useful for passing directly to
      ``RagService.batch_retrieve()``.
    """

    def get_items(self) -> list[QuestionItem]:
        """Return all question items from this source.

        Returns:
            List of :class:`QuestionItem` instances.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("get_items is not implemented")

    # ── Convenience helpers ───────────────────────────────────────────────────

    def get_questions(self) -> list[str]:
        """Return only the question text strings.

        Shorthand for ``[item.question_text for item in self.get_items()]``.

        Returns:
            Ordered list of question strings.
        """
        raise NotImplementedError("get_questions is not implemented")
