"""Base class for all LLM service implementations."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)


class LLMBase:
    """Abstract base for LLM services."""

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

    # ── Checkpoint helpers ────────────────────────────────────────────────────

    def _load_checkpoint(self, ckpt_path: Path) -> list[str]:
        """Load completed prompt results from a .jsonl checkpoint file.

        Returns a list of result strings in the order they were saved.
        """
        results: list[str] = []
        if not ckpt_path.exists():
            return results
        with ckpt_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    results.append(json.loads(line.strip()))
                except Exception:
                    pass
        logger.info("Checkpoint: loaded %d completed results", len(results))
        return results

    def _save_checkpoint(self, ckpt_path: Path, results: list[str]) -> None:
        """Write all completed results to a .jsonl checkpoint file."""
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        with ckpt_path.open("w", encoding="utf-8") as fh:
            for r in results:
                fh.write(json.dumps(r) + "\n")

    # ── Single-item interface ─────────────────────────────────────────────────

    def generate(self, prompt: str) -> str:
        raise NotImplementedError("generate is not implemented")

    def generate_structured(self, prompt: str, schema: Type[T]) -> T:
        raise NotImplementedError("generate_structured is not implemented")

    # ── Batch interface ───────────────────────────────────────────────────────

    def batch_generate(
        self,
        prompts: list[str],
        *,
        checkpoint_path: str | Path | None = None,
    ) -> list[str]:
        raise NotImplementedError("batch_generate is not implemented")

    def batch_generate_structured(
        self,
        prompts: list[str],
        schema: Type[T],
        *,
        checkpoint_path: str | Path | None = None,
    ) -> list[T]:
        raise NotImplementedError("batch_generate_structured is not implemented")
