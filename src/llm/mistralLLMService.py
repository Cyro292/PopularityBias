from __future__ import annotations

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
    .pip_install("transformers[torch]", "accelerate", "huggingface_hub", "pydantic", "tqdm", "outlines")
    .run_function(download_model)
)

@app.function(gpu="A10G", image=image, timeout=600, max_containers=2)
def generate(prompts: list[str], model_name: str = MODEL_NAME, max_new_tokens: int = 256) -> list[str]:
    from transformers import pipeline
    from tqdm import tqdm

    chatbot = pipeline("text-generation", model=model_name, device_map="cuda", max_new_tokens=max_new_tokens)
    results = []
    for prompt in tqdm(prompts, desc="Generating text"):
        result = chatbot([{"role": "user", "content": prompt}])
        results.append(result[0]["generated_text"][-1]["content"])  # type: ignore[index]
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


class MistralLLMService(LLMBase):  # type: ignore[misc]
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
    ) -> list[str]:
        if not prompts:
            return []

        ckpt_path = Path(checkpoint_path) if checkpoint_path else None
        completed = self._load_checkpoint(ckpt_path) if ckpt_path else []
        pending = prompts[len(completed):]
        results: list[str] = list(completed)

        if not pending:
            return results

        batched = [pending[i:i + self._request_batch_size] for i in range(0, len(pending), self._request_batch_size)]

        try:
            for batch_result in tqdm(
                self.generate_fn.map(batched, kwargs={"model_name": self.model_name, "max_new_tokens": 256}, return_exceptions=True),
                total=len(batched), desc="Generating text (parallel)", unit="batch",
            ):
                if isinstance(batch_result, Exception):
                    raise batch_result
                results.extend(batch_result)
        except Exception as e:
            logger.error("Mistral batch_generate failed: %s", e)
            if ckpt_path:
                self._save_checkpoint(ckpt_path, results)
                logger.error("Checkpoint saved with %d completed results", len(results))
            raise

        return results

    def generate_structured(self, prompt: str, schema: Type[T]) -> T:
        return self.generate_structured_fn.remote([prompt], schema_class=schema, model_name=self.model_name, max_new_tokens=256)[0]

    def batch_generate_structured(
        self,
        prompts: list[str],
        schema: Type[T],
        *,
        checkpoint_path: str | Path | None = None,
    ) -> list[T]:
        if not prompts:
            return []

        ckpt_path = Path(checkpoint_path) if checkpoint_path else None
        completed = self._load_checkpoint(ckpt_path) if ckpt_path else []
        pending = prompts[len(completed):]
        results: list[T] = list(completed)

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
            logger.error("Mistral batch_generate_structured failed: %s", e)
            if ckpt_path:
                self._save_checkpoint(ckpt_path, results)
                logger.error("Checkpoint saved with %d completed results", len(results))
            raise

        return results
