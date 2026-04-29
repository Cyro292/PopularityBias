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

    def _load_checkpoint(self, ckpt_path: Path, question_ids: list[str] | None = None) -> list[str]:
        """Load completed prompt results from a checkpoint file (.csv or .jsonl).

        Returns a list of result strings in the order they were saved.
        
        Args:
            ckpt_path: Path to checkpoint file.
            question_ids: Optional list of question IDs to filter and order results.
        """
        import pandas as pd
        
        results: list[str] = []
        if not ckpt_path.exists():
            logger.debug("Checkpoint: no checkpoint file at %s", ckpt_path)
            return results
        
        # Support both CSV and JSONL formats
        if ckpt_path.suffix == ".csv":
            try:
                df = pd.read_csv(ckpt_path)
                if "answer" in df.columns and "question_id" in df.columns:
                    # If question_ids provided, filter and order by them
                    if question_ids is not None:
                        # Build mapping from question_id to answer
                        id_to_answer = dict(zip(df["question_id"], df["answer"].fillna("")))
                        # Return answers in the order of question_ids
                        results = [id_to_answer.get(qid, "") for qid in question_ids]
                    else:
                        # Legacy: sort by question_idx if it exists, otherwise maintain CSV order
                        if "question_idx" in df.columns:
                            df = df.sort_values("question_idx")
                        results = df["answer"].fillna("").tolist()
                    logger.info("✓ Checkpoint: loaded %d completed results from %s", len(results), ckpt_path.name)
                    return results
                else:
                    logger.warning("CSV checkpoint missing required columns (question_id, answer)")
                    return []
            except Exception as e:
                logger.error("Failed to load CSV checkpoint %s: %s", ckpt_path, e)
                return []
        
        # JSONL format (legacy support)
        with ckpt_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    data = json.loads(line.strip())
                    # Handle both old format (plain string) and new format (dict with 'answer' key)
                    if isinstance(data, dict) and "answer" in data:
                        results.append(data["answer"])
                    elif isinstance(data, str):
                        results.append(data)
                    else:
                        # Fallback: convert to string
                        results.append(str(data))
                except Exception:
                    pass
        logger.info("✓ Checkpoint: loaded %d completed results from %s", len(results), ckpt_path.name)
        return results

    def _save_checkpoint(self, ckpt_path: Path, results: list[str], question_ids: list[str] | None = None) -> None:
        """Write all completed results to a checkpoint file (.csv or .jsonl).
        
        Args:
            ckpt_path: Path to checkpoint file.
            results: List of answer strings.
            question_ids: Optional list of question IDs corresponding to results.
        """
        import pandas as pd
        
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Support both CSV and JSONL formats based on file extension
        if ckpt_path.suffix == ".csv":
            data = {"answer": results}
            if question_ids is not None and len(question_ids) == len(results):
                data["question_id"] = question_ids
            else:
                # Fallback to index-based
                data["question_idx"] = range(len(results))
            df = pd.DataFrame(data)
            df.to_csv(ckpt_path, index=False)
        else:
            # JSONL format (legacy)
            with ckpt_path.open("w", encoding="utf-8") as fh:
                for idx, r in enumerate(results):
                    entry = {"question_idx": idx, "answer": r}
                    if question_ids is not None and idx < len(question_ids):
                        entry["question_id"] = question_ids[idx]
                    fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

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
        question_ids: list[str] | None = None,
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
