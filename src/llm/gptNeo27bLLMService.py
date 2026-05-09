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

@app.function(gpu="A10G", image=image, timeout=600, max_containers=2)
def generate(prompts: list[str], model_name: str = MODEL_NAME, max_new_tokens: int = 256) -> list[str]:
    from transformers import AutoTokenizer, pipeline
    from tqdm import tqdm

    max_input_tokens = 2048 - max_new_tokens  # GPT-Neo max_position_embeddings=2048

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    generator = pipeline("text-generation", model=model_name, tokenizer=tokenizer, device_map="cuda")
    results = []
    for prompt in tqdm(prompts, desc="Generating text"):
        # Truncate by token count — character limits are unreliable
        token_ids = tokenizer.encode(prompt, truncation=True, max_length=max_input_tokens)
        truncated = tokenizer.decode(token_ids, skip_special_tokens=True)
        result = generator(
            truncated,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        generated: str = result[0]["generated_text"]  # type: ignore[index]
        results.append(generated[len(truncated):].strip())
    return results

@app.function(gpu="A10G", image=image, timeout=600, max_containers=2)
def generate_structured(
    prompts: list[str],
    schema_class: type[BaseModel],
    *,
    model_name: str = MODEL_NAME,
    max_new_tokens: int = 256,
) -> list:
    import outlines
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm import tqdm

    model = outlines.from_transformers(
        AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda"),
        AutoTokenizer.from_pretrained(model_name),
    )
    results = []
    for prompt in tqdm(prompts, desc="Generating structured responses"):
        results.append(model(prompt, schema_class, max_new_tokens=max_new_tokens))
    return results


class GPTNeo27bLLMService(LLMBase):  # type: ignore[misc]
    def __init__(self, temperature: float = 0.0, request_batch_size: int = 64) -> None:
        super().__init__(model_name=MODEL_NAME, temperature=temperature)
        self._request_batch_size = request_batch_size
        self.generate_fn = modal.Function.from_name(APP_NAME, "generate")
        self.generate_structured_fn = modal.Function.from_name(APP_NAME, "generate_structured")

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

        try:
            for batch_result in tqdm(
                self.generate_fn.map(batched, kwargs={"model_name": self.model_name, "max_new_tokens": 256}, return_exceptions=True, wrap_returned_exceptions=False),
                total=len(batched), desc="Generating text (parallel)", unit="batch",
            ):
                if isinstance(batch_result, Exception):
                    raise batch_result
                results.extend(batch_result)
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
        return self.generate_structured_fn.remote([prompt], schema_class=schema, model_name=self.model_name, max_new_tokens=256)[0]

    def batch_generate_structured(
        self,
        prompts: list[str],
        schema: T,
        *,
        checkpoint_path: str | Path | None = None,
    ) -> list[Any]:
        if not prompts:
            return []

        ckpt_path = Path(checkpoint_path) if checkpoint_path else None
        completed = self._load_checkpoint(ckpt_path) if ckpt_path else []
        pending = prompts[len(completed):]
        results: list[Any] = list(completed)

        if not pending:
            return results

        batched = [pending[i:i + self._request_batch_size] for i in range(0, len(pending), self._request_batch_size)]

        try:
            for batch_result in tqdm(
                self.generate_structured_fn.map(batched, kwargs={"schema_class": schema, "model_name": self.model_name, "max_new_tokens": 256}, return_exceptions=True),
                total=len(batched), desc="Generating structured (parallel)", unit="batch",
            ):
                if isinstance(batch_result, Exception):
                    raise batch_result
                results.extend(batch_result)
        except Exception as e:
            logger.error("GPTNeo batch_generate_structured failed: %s", e)
            if ckpt_path:
                self._save_checkpoint(ckpt_path, results)
                logger.error("Checkpoint saved with %d completed results", len(results))
            raise

        return results
