from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, TypeVar, Type

from pydantic import BaseModel
from tqdm import tqdm
import modal

if not modal.is_local():
    LLMBase = object
else:
    from src.llm.base import LLMBase

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
APP_NAME = "popularity_bias_ministral_service_as"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

def download_model() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
    AutoTokenizer.from_pretrained(MODEL_NAME)

app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("transformers[torch]", "accelerate", "huggingface_hub", "pydantic", "tqdm")
    .run_function(download_model)
)

@app.function(gpu="H100", image=image, timeout=600, max_containers=4)
def generate(prompts: list[str], model_name: str = MODEL_NAME, max_new_tokens: int = 256, gpu_batch_size: int = 4) -> list[str]:
    import warnings
    from transformers import AutoTokenizer, pipeline

    logging.getLogger("transformers").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message=".*max_length.*")
    warnings.filterwarnings("ignore", message=".*generation_config.*")
    warnings.filterwarnings("ignore", message=".*generation flags.*")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    chatbot = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=tokenizer,
        device_map="cuda",
        max_new_tokens=max_new_tokens,
    )

    messages = [[{"role": "user", "content": p}] for p in prompts]
    print(f"[mistral] generating {len(messages)} prompts | gpu_batch_size={gpu_batch_size}")

    results_raw = chatbot(messages, batch_size=gpu_batch_size)
    return [r[0]["generated_text"][-1]["content"] for r in results_raw]


class MistralLLMService(LLMBase):  # type: ignore[misc]
    def __init__(self, temperature: float = 0.0, request_batch_size: int = 64, gpu_batch_size: int = 4) -> None:
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
            "→ [mistral] dispatching %d prompts across %d Modal batches (request_batch=%d, gpu_batch=%d)",
            len(pending), n_batches, self._request_batch_size, self._gpu_batch_size,
        )

        try:
            for i, batch_result in enumerate(tqdm(
                self.generate_fn.map(batched, kwargs={"model_name": self.model_name, "max_new_tokens": 256, "gpu_batch_size": self._gpu_batch_size}, return_exceptions=True, wrap_returned_exceptions=False),
                total=n_batches,
                desc="[mistral] Modal batches",
                unit="batch",
            )):
                if isinstance(batch_result, Exception):
                    raise batch_result
                results.extend(batch_result)
                logger.info("[mistral] batch %d/%d done — %d/%d prompts complete", i + 1, n_batches, len(results), len(prompts))
        except Exception as e:
            logger.error("Mistral batch_generate failed: %s", e)
            if ckpt_path:
                self._save_checkpoint(ckpt_path, results, question_ids)
                logger.error("Checkpoint saved with %d completed results", len(results))
            raise

        if ckpt_path:
            self._save_checkpoint(ckpt_path, results, question_ids)
            logger.info("✓ Checkpoint saved with %d completed results", len(results))

        return results

    def generate_structured(self, prompt: str, schema: Type[T]) -> T:
        wrapped = self._structured_prompt(prompt, schema)
        raw = self.generate_fn.remote([wrapped], model_name=self.model_name, max_new_tokens=512)[0]
        return self._parse_structured(raw, schema)

    def batch_generate_structured(
        self,
        prompts: list[str],
        schema: Type[T],
        *,
        checkpoint_path: str | Path | None = None,
    ) -> list[T]:
        if not prompts:
            return []

        wrapped = [self._structured_prompt(p, schema) for p in prompts]
        raw_results = self.batch_generate(wrapped, checkpoint_path=checkpoint_path)

        results: list[T] = []
        for raw in raw_results:
            try:
                results.append(self._parse_structured(raw, schema))
            except Exception as e:
                logger.warning("[mistral] structured parse failed, substituting None: %s", e)
                results.append(None)
        return results


