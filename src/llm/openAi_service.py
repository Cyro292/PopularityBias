"""OpenAI LLM service implementation using LangChain."""

from __future__ import annotations

import logging
import os
from typing import Any, Type, TypeVar

from langchain_core.language_models import BaseLanguageModel
from pydantic import BaseModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from tqdm import tqdm

from src.llm.base import LLMBase, T

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

    def batch_generate(self, prompts: list[str], batch_size: int = 50) -> list[str]:
        """Generate freeform text responses for multiple prompts concurrently.

        Delegates to LangChain's ``Runnable.batch()``, which dispatches all
        prompts concurrently (subject to any ``rate_limiter`` set on the
        model).

        Args:
            prompts: List of input prompt strings.

        Returns:
            List of text responses in the same order as *prompts*.

        Raises:
            Exception: Propagates any LangChain / OpenAI API errors.
        """
        if not prompts:
            return []
        try:
            all_responses = []

            for i in tqdm(range(0, len(prompts), batch_size), desc="Generating responses", unit="batch"):
                batch_prompts = prompts[i : i + batch_size]
                responses = self._llm.batch(batch_prompts)  # type: ignore[union-attr]
                all_responses.extend(responses)

            contents = [r.content for r in all_responses]

            return contents
        except Exception as e:
            logger.error("OpenAI batch_generate failed: %s", e)
            raise

    def batch_generate_structured(
        self,
        prompts: list[str],
        schema: Type[T],
        batch_size: int = 50,
    ) -> list[T]:
        """Generate structured responses for multiple prompts concurrently.

        Binds the model with ``with_structured_output(schema)`` then calls
        ``Runnable.batch()`` so all prompts are dispatched concurrently.

        Args:
            prompts: List of input prompt strings.
            schema: A :class:`pydantic.BaseModel` subclass defining the
                expected output structure.

        Returns:
            List of *schema* instances in the same order as *prompts*.

        Raises:
            ValueError: If any result is not an instance of *schema*.
            Exception: Propagates any LangChain / OpenAI API errors.
        """
        if not prompts:
            return []
        try:
            results = []
            for i in tqdm(range(0, len(prompts), batch_size), desc="Generating structured responses", unit="batch"):
                batch_prompts = prompts[i : i + batch_size]
                batch_results = self._llm.with_structured_output(schema).batch(batch_prompts)  # type: ignore[union-attr]
                results.extend(batch_results)
        except Exception as e:
            logger.error("OpenAI batch_generate_structured failed: %s", e)
            raise

        for i, result in enumerate(results):
            if not isinstance(result, schema):
                raise ValueError(
                    f"Result at index {i}: expected {schema.__name__}, "
                    f"got {type(result).__name__}"
                )
        return results  # type: ignore[return-value]
