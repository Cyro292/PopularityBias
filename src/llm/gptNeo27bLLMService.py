from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel
from tqdm import tqdm
import modal

if not modal.is_local():
    LLMBase = object
else:
    from src.llm.base import LLMBase

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
APP_NAME = "popularity_bias_gpt_neo_27b_service_as"
MODEL_NAME = "EleutherAI/gpt-neo-2.7B"
    
def download_model() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
    AutoTokenizer.from_pretrained(MODEL_NAME)

app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("transformers[torch]", "accelerate", "huggingface_hub", "pydantic", "tqdm", "outlines", "prompt_toolkit")
    .run_function(download_model)
)

@app.function(gpu="H100", image=image, timeout=600, max_containers=4)
def generate(prompts: list[str], model_name: str = MODEL_NAME, max_new_tokens: int = 256, gpu_batch_size: int = 32) -> list[str]:
    import logging
    import warnings
    from transformers import AutoTokenizer, pipeline

    # Suppress noisy HuggingFace warnings that clutter Modal logs
    logging.getLogger("transformers").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message=".*max_length.*")
    warnings.filterwarnings("ignore", message=".*generation_config.*")
    warnings.filterwarnings("ignore", message=".*generation flags.*")

    max_input_tokens = 2048 - max_new_tokens  # GPT-Neo max_position_embeddings=2048

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    generator = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=tokenizer,
        device_map="cuda",
        max_length=max_input_tokens + max_new_tokens,
    )

    # Batch-truncate all prompts upfront
    truncated = [
        tokenizer.decode(
            tokenizer.encode(p, truncation=True, max_length=max_input_tokens),
            skip_special_tokens=True,
        )
        for p in prompts
    ]

    print(f"[neo] generating {len(truncated)} prompts | gpu_batch_size={gpu_batch_size}")

    # Single batched call — H100 processes gpu_batch_size prompts per forward pass
    results_raw = generator(
        truncated,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        batch_size=gpu_batch_size,
        truncation=True,
    )
    return [r[0]["generated_text"][len(p):].strip() for r, p in zip(results_raw, truncated)]


class GPTNeo27bLLMService(LLMBase):  # type: ignore[misc]
    def __init__(self, temperature: float = 0.0, request_batch_size: int = 128, gpu_batch_size: int = 32) -> None:
        super().__init__(model_name=MODEL_NAME, temperature=temperature)
        self._request_batch_size = request_batch_size
        self._gpu_batch_size = gpu_batch_size
        self.generate_fn = modal.Function.from_name(APP_NAME, "generate")

    def generate(self, prompt: str) -> str:
        return self.generate_fn.remote([prompt], model_name=self.model_name, max_new_tokens=256)[0]

    def batch_generate(
        self,
        prompts: list[str],
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

        batched = [pending[i:i + self._request_batch_size] for i in range(0, len(pending), self._request_batch_size)]
        n_batches = len(batched)
        logger.info(
            "→ [neo] dispatching %d prompts across %d Modal batches (request_batch=%d, gpu_batch=%d)",
            len(pending), n_batches, self._request_batch_size, self._gpu_batch_size,
        )

        try:
            for i, batch_result in enumerate(tqdm(
                self.generate_fn.map(batched, kwargs={"model_name": self.model_name, "max_new_tokens": 256, "gpu_batch_size": self._gpu_batch_size}, return_exceptions=True, wrap_returned_exceptions=False),
                total=n_batches,
                desc="[neo] Modal batches",
                unit="batch",
            )):
                if isinstance(batch_result, Exception):
                    raise batch_result
                results.extend(batch_result)
                logger.info("[neo] batch %d/%d done — %d/%d prompts complete", i + 1, n_batches, len(results), len(prompts))
        except Exception as e:
            logger.error("GPTNeo batch_generate failed: %s", e)
            if ckpt_path:
                ids_slice = question_ids[:len(results)] if question_ids is not None else None
                self._save_checkpoint(ckpt_path, results, ids_slice)
                logger.error("Checkpoint saved with %d completed results", len(results))
            raise

        if ckpt_path:
            self._save_checkpoint(ckpt_path, results, question_ids)
            logger.info("✓ Checkpoint saved with %d completed results", len(results))

        return results

    def generate_structured(self, prompt: str, schema: T) -> Any:
        raw = self.batch_generate([prompt])[0]
        return self._parse_structured(raw, schema)

    def batch_generate_structured(
        self,
        prompts: list[str],
        schema: T,
        *,
        checkpoint_path: str | Path | None = None,
    ) -> list[Any]:
        if not prompts:
            return []

        wrapped = [self._structured_prompt(p, schema) for p in prompts]
        raw_results = self.batch_generate(wrapped, checkpoint_path=checkpoint_path)

        results: list[Any] = []
        for raw in raw_results:
            try:
                results.append(self._parse_structured(raw, schema))
            except Exception as e:
                logger.warning("[neo] structured parse failed, substituting None: %s", e)
                results.append(None)
        return results
