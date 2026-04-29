"""OpenAI LLM service implementation using LangChain."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Type, TypeVar

from langchain_core.language_models import BaseLanguageModel
from langchain_core.language_models.base import LanguageModelInput
from pydantic import BaseModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from tqdm import tqdm

from src.llm.base import LLMBase

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)


class OpenAIService(LLMBase):  # type: ignore[misc]  # LLMBase is not generic
    """LangChain-backed OpenAI LLM service.

    Supports freeform and structured (Pydantic) generation for both single
    prompts and batches.  Batches are dispatched via LangChain's
    ``Runnable.batch()``, which sends requests concurrently.

    Pass a ``langchain_core.rate_limiters.InMemoryRateLimiter`` (or any object
    satisfying the ``BaseRateLimiter`` protocol) as ``rate_limiter`` — it is
    forwarded directly to ``ChatOpenAI`` so LangChain applies throttling
    automatically, including during batch calls.

    Args:
        model_name: OpenAI model identifier, e.g. ``"gpt-4o"`` or
            ``"gpt-4o-mini"``.
        temperature: Sampling temperature (default ``0.0`` for deterministic
            outputs).
        api_key: OpenAI API key. Falls back to the ``OPENAI_API_KEY``
            environment variable if ``None``.
        rate_limiter: Optional LangChain ``BaseRateLimiter`` passed directly
            to ``ChatOpenAI``.  Example::

                from langchain_core.rate_limiters import InMemoryRateLimiter
                limiter = InMemoryRateLimiter(requests_per_second=10)
                service = OpenAIService("gpt-4o-mini", rate_limiter=limiter)

        **kwargs: Additional keyword arguments forwarded to ``ChatOpenAI``
            (e.g. ``max_tokens``, ``timeout``).
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        api_key: str | None = None,
        rate_limiter: Any | None = None,
        requests_per_second: int = 30,
        **kwargs: Any,
    ) -> None:
        
        rate_limiter = InMemoryRateLimiter(requests_per_second=requests_per_second) if rate_limiter is None else rate_limiter

        super().__init__(
            model_name=model_name,
            temperature=temperature,
            api_key=api_key,
            rate_limiter=rate_limiter,
        )
        self._kwargs = kwargs
        self._llm: BaseLanguageModel = self._build_llm()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_llm(self) -> BaseLanguageModel:
        """Instantiate the underlying ``ChatOpenAI`` model."""
        from langchain_openai import ChatOpenAI

        resolved_key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "OpenAI API key must be provided via the `api_key` argument or "
                "the OPENAI_API_KEY environment variable."
            )

        init_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "temperature": self.temperature,
            "api_key": resolved_key,
            "rate_limiter": self.rate_limiter,
            **self._kwargs,
        }
        if self.rate_limiter is not None:
            init_kwargs["rate_limiter"] = self.rate_limiter

        return ChatOpenAI(**init_kwargs)

    # ── Single-item interface ─────────────────────────────────────────────────

    def generate(self, prompt: str) -> str:
        """Generate a freeform text response for the given prompt.

        Args:
            prompt: The input prompt string.

        Returns:
            The model's text response.

        Raises:
            Exception: Propagates any LangChain / OpenAI API errors.
        """
        try:
            response = self._llm.invoke(prompt)
            return response.content  # type: ignore[union-attr]
        except Exception as e:
            logger.error("OpenAI generate failed: %s", e)
            raise

    def generate_structured(self, prompt: str, schema: Type[T]) -> T:
        """Generate a structured response validated against a Pydantic model.

        Uses LangChain's ``with_structured_output`` to bind the JSON schema
        derived from *schema* to the model call.

        Args:
            prompt: The input prompt string.
            schema: A :class:`pydantic.BaseModel` subclass defining the
                expected output structure.

        Returns:
            An instance of *schema* populated with the model's response.

        Raises:
            ValueError: If the model returns an unexpected type.
            Exception: Propagates any LangChain / OpenAI API errors.
        """
        try:
            result = self._llm.with_structured_output(schema).invoke(prompt)  # type: ignore[union-attr]
        except Exception as e:
            logger.error("OpenAI generate_structured failed: %s", e)
            raise

        if not isinstance(result, schema):
            raise ValueError(
                f"Expected {schema.__name__}, got {type(result).__name__}"
            )
        return result

    # ── Batch interface ───────────────────────────────────────────────────────

    def batch_generate(
        self,
        prompts: list[str],
        batch_size: int = 50,
        *,
        checkpoint_path: str | Path | None = None,
        question_ids: list[str] | None = None,
    ) -> list[str]:
        if not prompts:
            return []

        ckpt_path = Path(checkpoint_path) if checkpoint_path else None
        completed = self._load_checkpoint(ckpt_path, question_ids) if ckpt_path else []
        pending = prompts[len(completed):]
        results: list[str] = list(completed)

        if not pending:
            logger.info("✓ All %d prompts already completed (checkpoint fully reused)", len(results))
            return results
        
        if len(completed) != len(prompts):
            logger.warning(
                "⚠ Checkpoint length mismatch (checkpoint=%d, prompts=%d), generating %d new responses",
                len(completed), len(prompts), len(pending)
            )

        try:
            for i in tqdm(range(0, len(pending), batch_size), desc="Generating responses", unit="batch"):
                batch: list[LanguageModelInput] = pending[i : i + batch_size]
                responses = self._llm.batch(batch)  # type: ignore[union-attr]
                results.extend(r.content for r in responses)
        except Exception as e:
            logger.error("OpenAI batch_generate failed: %s", e)
            if ckpt_path:
                self._save_checkpoint(ckpt_path, results, question_ids)
                logger.error("Checkpoint saved with %d completed results", len(results))
            raise

        if ckpt_path:
            self._save_checkpoint(ckpt_path, results, question_ids)

        return results

    def batch_generate_structured(
        self,
        prompts: list[str],
        schema: Type[T],
        batch_size: int = 50,
        *,
        checkpoint_path: str | Path | None = None,
    ) -> list[T]:
        if not prompts:
            return []

        ckpt_path = Path(checkpoint_path) if checkpoint_path else None
        completed = self._load_checkpoint(ckpt_path) if ckpt_path else []
        pending = prompts[len(completed):]
        results: list[T] = list(completed)  # type: ignore[assignment]

        if not pending:
            return results

        structured_llm = self._llm.with_structured_output(schema)  # type: ignore[union-attr]

        try:
            for i in tqdm(range(0, len(pending), batch_size), desc="Generating structured responses", unit="batch"):
                batch: list[LanguageModelInput] = pending[i : i + batch_size]
                batch_results = structured_llm.batch(batch)
                results.extend(batch_results)  # type: ignore[arg-type]
        except Exception as e:
            from openai import BadRequestError
            if isinstance(e, BadRequestError):
                # find the offending prompt by retrying one-by-one
                batch_start = (len(results) - len(completed))
                for j, prompt in enumerate(pending[batch_start : batch_start + batch_size]):
                    try:
                        structured_llm.invoke(prompt)
                    except BadRequestError:
                        logger.error(
                            "Bad prompt at index %d caused the 400 error:\n  %.500s",
                            len(completed) + batch_start + j, prompt,
                        )
            logger.error("OpenAI batch_generate_structured failed: %s", e)
            if ckpt_path:
                self._save_checkpoint(ckpt_path, results)  # type: ignore[arg-type]
                logger.error("Checkpoint saved with %d completed results", len(results))
            raise

        return results  # type: ignore[return-value]
