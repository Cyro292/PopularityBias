"""Base class for all LLM service implementations."""

from __future__ import annotations

from typing import Any, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMBase:
    """Abstract base for LLM services.

    All concrete implementations (e.g. ``OpenAIService``) must subclass this
    and implement :meth:`generate`, :meth:`generate_structured`,
    :meth:`batch_generate`, and :meth:`batch_generate_structured`.

    Args:
        model_name: Model identifier string (e.g. ``"gpt-4o-mini"``).
        temperature: Sampling temperature.  ``0.0`` gives deterministic output.
        api_key: Provider API key.  Falls back to the relevant environment
            variable when ``None``.
        rate_limiter: Optional LangChain-compatible rate limiter passed
            directly to the underlying model constructor.  Use
            ``langchain_core.rate_limiters.InMemoryRateLimiter`` or any object
            that satisfies the ``BaseRateLimiter`` protocol.
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        api_key: str | None = None,
        rate_limiter: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = api_key
        self.rate_limiter = rate_limiter

    # ── Single-item interface ─────────────────────────────────────────────────

    def generate(self, prompt: str) -> str:
        """Generate a freeform text response.

        Args:
            prompt: Input prompt string.

        Returns:
            The model's text response.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("generate is not implemented")

    def generate_structured(self, prompt: str, schema: Type[T]) -> T:
        """Generate a response validated against a Pydantic model.

        Args:
            prompt: Input prompt string.
            schema: A :class:`pydantic.BaseModel` subclass whose fields define
                the expected output structure.

        Returns:
            An instance of *schema* populated with the model's response.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("generate_structured is not implemented")

    # ── Batch interface ───────────────────────────────────────────────────────

    def batch_generate(self, prompts: list[str]) -> list[str]:
        """Generate freeform text responses for multiple prompts.

        Implementations should exploit the underlying model's native batch
        endpoint (e.g. LangChain's ``Runnable.batch()``) rather than
        looping over :meth:`generate`.

        Args:
            prompts: List of input prompt strings.

        Returns:
            List of text responses, same length and order as *prompts*.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("batch_generate is not implemented")

    def batch_generate_structured(
        self,
        prompts: list[str],
        schema: Type[T],
    ) -> list[T]:
        """Generate structured responses for multiple prompts.

        Implementations should use the underlying model's native batch
        endpoint bound with ``with_structured_output``.

        Args:
            prompts: List of input prompt strings.
            schema: A :class:`pydantic.BaseModel` subclass whose fields define
                the expected output structure.

        Returns:
            List of *schema* instances, same length and order as *prompts*.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("batch_generate_structured is not implemented")
