"""Shared utilities for RAG services.

Provides:
- ``IndexingConfig``         Central configuration dataclass for the indexing pipeline.
- ``build_embeddings``       Factory that builds an embeddings instance with optional
                             rate limiting.
- ``RateLimitedEmbeddings``  Thin wrapper adding rate limiting to any embeddings object.
- ``get_embedding_class``    Lazy-loader for embedding provider classes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

from langchain_core.rate_limiters import BaseRateLimiter, InMemoryRateLimiter

logger = logging.getLogger(__name__)


# ── Embedding provider registry ───────────────────────────────────────────────

_EMBEDDING_CLASSES: dict[str, type] = {}


def get_embedding_class(provider: str):
    """Lazy-load embedding classes to avoid import overhead.

    Args:
        provider: Name of the embedding provider
            (``"openai"``, ``"google"``, ``"huggingface"``, ``"modal"``).

    Returns:
        The embedding class for the specified provider.

    Raises:
        ValueError: If the provider is not supported or is empty.
    """
    if not provider:
        raise ValueError("Provider required")

    provider = provider.lower()
    if provider not in _EMBEDDING_CLASSES:
        if provider == "openai":
            from langchain_openai import OpenAIEmbeddings
            _EMBEDDING_CLASSES[provider] = OpenAIEmbeddings
        elif provider == "google":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            _EMBEDDING_CLASSES[provider] = GoogleGenerativeAIEmbeddings
        elif provider == "huggingface":
            from langchain_huggingface import HuggingFaceEmbeddings
            _EMBEDDING_CLASSES[provider] = HuggingFaceEmbeddings
        elif provider == "modal":
            from src.embeddings.modal_embedding import ModalEmbeddings
            _EMBEDDING_CLASSES[provider] = ModalEmbeddings
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")
    return _EMBEDDING_CLASSES[provider]


class RateLimitedEmbeddings:
    """Wrapper that adds rate limiting to any embeddings instance."""

    def __init__(self, embeddings: Any, rate_limiter: BaseRateLimiter) -> None:
        self._embeddings = embeddings
        self._rate_limiter = rate_limiter

    def _acquire(self) -> None:
        """Block until a rate limit token is available."""
        self._rate_limiter.acquire(blocking=True)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self._acquire()
        return self._embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        self._acquire()
        return self._embeddings.embed_query(text)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._embeddings, item)


def build_embeddings(
    provider: str | None = None,
    model: str | None = None,
    request_batch_size: int | None = None,
    gpu_batch_size: int | None = None,
    normalise_embeddings: bool | None = None,
    trust_remote_code: bool = False,
    rate_limiter: BaseRateLimiter | None = None,
    requests_per_second: float | None = None,
    check_interval: float = 0.1,
    bucket_size: float = 1.0,
    **kwargs: Any,
):
    """Build an embeddings instance with optional rate limiting.

    Args:
        provider: Embedding provider name
            (``"openai"``, ``"google"``, ``"huggingface"``, ``"modal"``).
        model: Model name / identifier.
        request_batch_size: Texts per Modal request batch (Modal only).
        gpu_batch_size: Forward-pass batch size on GPU (Modal only).
        normalise_embeddings: Whether to L2-normalise embeddings (Modal only).
        trust_remote_code: Allow custom HuggingFace model code (HuggingFace only).
        rate_limiter: Pre-configured rate limiter (takes precedence over
            ``requests_per_second``).
        requests_per_second: Auto-create an ``InMemoryRateLimiter`` at this rate.
        check_interval: Rate-limit check interval in seconds.
        bucket_size: Token bucket size.
        **kwargs: Extra provider-specific options (ignored silently).

    Returns:
        Embeddings instance, optionally wrapped with ``RateLimitedEmbeddings``.

    Raises:
        ValueError: If ``provider`` or ``model`` is not set, or if required
            Modal parameters are missing.
    """
    if not provider:
        raise ValueError("provider is required")
    if not model:
        raise ValueError("model is required")

    embedding_cls = get_embedding_class(provider)

    if provider.lower() == "huggingface":
        base = embedding_cls(
            model_name=model,
            model_kwargs={"trust_remote_code": trust_remote_code},
        )
    elif provider.lower() == "modal":
        if request_batch_size is None or gpu_batch_size is None or normalise_embeddings is None:
            raise ValueError(
                "gpu_batch_size, request_batch_size, and normalise_embeddings "
                "are required for the modal provider"
            )
        base = embedding_cls(
            model_name=model,
            request_batch_size=request_batch_size,
            gpu_batch_size=gpu_batch_size,
            normalise_embeddings=normalise_embeddings,
        )
    else:
        base = embedding_cls(model=model)

    if rate_limiter:
        return RateLimitedEmbeddings(base, rate_limiter)

    if requests_per_second and requests_per_second > 0:
        limiter = InMemoryRateLimiter(
            requests_per_second=requests_per_second,
            check_every_n_seconds=check_interval,
            max_bucket_size=bucket_size,
        )
        return RateLimitedEmbeddings(base, limiter)

    return base


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class IndexingConfig:
    """Central configuration for the entire indexing pipeline.

    All tunable hyperparameters live here so the notebook controls everything.

    Groups:
        Text processing:  chunk_size, chunk_overlap
        ES bulk insert:   batch_size
        GPU embedding:    gpu_batch_size
        Embedding:        embedding_provider, embedding_model, distance_function
        Rate limiting:    rate_limiter, requests_per_second, ...
    """

    # ── Text processing ──────────────────────────────────────────────────
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # ── GPU embedding (runtime — no redeploy needed) ────────────────────
    gpu_batch_size: int | None = None       # forward-pass batch on GPU
    request_batch_size: int | None = None
    normalise_embeddings: bool | None = None

    # ── Embedding provider ──────────────────────────────────────────────
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-large"
    trust_remote_code: bool = False
    distance_function: str = "COSINE"

    # ── Prompt templates ─────────────────────────────────────────────────
    passage_prompt_file: str | None = None   # path to passage/embedding prompt file
    query_prompt_file: str | None = None     # path to query prompt file

    # ── Rate limiting ───────────────────────────────────────────────────
    rate_limiter: BaseRateLimiter | None = None
    requests_per_second: float | None = None
    rate_limit_check_interval: float = 0.1
    rate_limit_bucket_size: float = 1.0

    # ── UI ───────────────────────────────────────────────────────────────
    use_progress: bool = True
