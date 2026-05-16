"""Base class for all LLM service implementations."""

from __future__ import annotations

import json
import logging
import re
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

        Returns answers ordered to match *question_ids*.  Any ID not present
        in the checkpoint gets an empty string placeholder so that the caller
        can detect gaps via :meth:`_load_checkpoint_map` instead.

        Args:
            ckpt_path: Path to checkpoint file.
            question_ids: Ordered list of all question IDs for this run.
        """
        import pandas as pd

        results: list[str] = []
        if not ckpt_path.exists():
            logger.debug("Checkpoint: no checkpoint file at %s", ckpt_path)
            return results

        if ckpt_path.suffix == ".csv":
            try:
                df = pd.read_csv(ckpt_path)
                if "answer" in df.columns and "question_id" in df.columns:
                    if question_ids is not None:
                        id_to_answer = dict(zip(df["question_id"].astype(str), df["answer"].fillna("")))
                        results = [id_to_answer.get(str(qid), "") for qid in question_ids]
                    else:
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
                    if isinstance(data, dict) and "answer" in data:
                        results.append(data["answer"])
                    elif isinstance(data, str):
                        results.append(data)
                    else:
                        results.append(str(data))
                except Exception:
                    pass
        logger.info("✓ Checkpoint: loaded %d completed results from %s", len(results), ckpt_path.name)
        return results

    def _load_checkpoint_map(self, ckpt_path: Path) -> dict[str, str]:
        """Load a checkpoint as a ``{question_id: answer}`` mapping.

        Only works for CSV checkpoints that have a ``question_id`` column.
        Returns an empty dict for JSONL checkpoints or missing files.

        Args:
            ckpt_path: Path to the checkpoint file.

        Returns:
            Mapping of question_id (str) → answer (str) for every row that
            has a non-empty ``question_id``.
        """
        import pandas as pd

        if not ckpt_path.exists() or ckpt_path.suffix != ".csv":
            return {}
        try:
            df = pd.read_csv(ckpt_path)
            if "answer" not in df.columns or "question_id" not in df.columns:
                return {}
            return dict(zip(df["question_id"].astype(str), df["answer"].fillna("")))
        except Exception as e:
            logger.error("Failed to load checkpoint map from %s: %s", ckpt_path, e)
            return {}

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

    # ── Shared structured-output helpers ─────────────────────────────────────

    @staticmethod
    def _structured_prompt(prompt: str, schema: Type[T]) -> str:
        """Append a JSON-format instruction to *prompt* so plain-text models
        can produce structured output without constrained decoding.

        Args:
            prompt: The base prompt text.
            schema: Pydantic model whose JSON schema describes the expected output.

        Returns:
            The prompt with a JSON instruction appended.
        """
        schema_str = json.dumps(schema.model_json_schema(), indent=2)
        return (
            f"{prompt}\n\n"
            f"Respond with a JSON object matching this schema:\n{schema_str}\n"
            f"Output only valid JSON, no extra text."
        )

    @staticmethod
    def _parse_structured(raw: str, schema: Type[T]) -> T:
        """Parse *raw* LLM output into a *schema* instance.

        Tries strict JSON parsing first, then falls back to extracting the
        first JSON object found in the text, then to field-level regex for
        boolean ``verdict`` fields.

        Args:
            raw: Raw string returned by the LLM.
            schema: Pydantic model to validate against.

        Returns:
            A validated *schema* instance.

        Raises:
            ValueError: If all parsing strategies fail.
        """
        # 1. Strict parse
        try:
            return schema.model_validate_json(raw)
        except Exception:
            pass

        # 2. Extract first {...} block
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return schema.model_validate_json(match.group(0))
            except Exception:
                pass

        # 3. Regex fallback for verdict/reasoning fields (binary evaluator)
        verdict_match = re.search(r'"verdict"\s*:\s*(true|false)', raw, re.IGNORECASE)
        if verdict_match:
            verdict = verdict_match.group(1).lower() == "true"
            reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', raw)
            reasoning = reasoning_match.group(1) if reasoning_match else raw[:200]
            logger.warning("JSON parse fell back to regex (verdict=%s): %.80s…", verdict, raw)
            try:
                return schema.model_validate({"verdict": verdict, "reasoning": reasoning})
            except Exception:
                pass

        raise ValueError(f"Could not parse structured output: {raw[:300]}")
