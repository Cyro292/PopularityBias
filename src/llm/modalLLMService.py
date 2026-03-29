"""Modal GPU LLM service.

Deploy:  modal deploy src/llm/ModalLLM.py
Use:     from src.llm.ModalLLM import ModalLLMService
         svc = ModalLLMService()
         replies = svc.batch_generate(["What is RAG?"])
"""

from __future__ import annotations

import logging
from typing import Any
from typing import TypeVar

import modal
from tqdm import tqdm

logger = logging.getLogger(__name__)

T = TypeVar("T")

# LLMBase is only needed for the client-side class, not on the Modal worker.
# Lazy import avoids ModuleNotFoundError when Modal runs this file remotely.
if not modal.is_local():
    LLMBase = object  # type: ignore[misc,assignment]
else:
    from src.llm.base import LLMBase

APP_NAME = "popularity_bias_general_service_as"

app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("transformers[torch]", "accelerate", "huggingface_hub", "pydantic", "tqdm")
)


@app.function(gpu="T4", image=image, timeout=600)
def generate(prompts: list[str], model_name: str, max_new_tokens: int) -> list[str]:
    from transformers import pipeline
    from tqdm import tqdm

    chatbot = pipeline(
            "text-generation",
            model=model_name,
            device_map="cuda",
            max_new_tokens=max_new_tokens,
    )
    results = []
    for prompt in tqdm(prompts, desc="Generating text"):
        result = chatbot([{"role": "user", "content": prompt}])
        results.append(result[0]["generated_text"][-1]["content"])  # type: ignore[index]
    return results

@app.function(gpu="T4", image=image, timeout=600)
def generate_structured(
    prompts: list[str],
    model_name: str,
    max_new_tokens: int,
    schema_class,   # pass the Pydantic model class here
) -> list:
    """
    Generate structured outputs for a list of prompts using a HF model on GPU.

    Args:
        prompts: list of text prompts
        model_name: HF model name
        max_new_tokens: max tokens to generate
        schema_class: a Pydantic model class for structured output

    Returns:
        List of validated Pydantic objects
    """
    from transformers import pipeline
    from pydantic import ValidationError
    import json
    from tqdm import tqdm

    # Initialize pipeline on GPU
    chatbot = pipeline(
        task="text-generation",
        model=model_name,
        device_map="auto",
        torch_dtype="auto",
        max_new_tokens=max_new_tokens
    )

    results = []

    for prompt in tqdm(prompts, desc="Generating structured output"):
        # Add schema instructions to force JSON output
        full_prompt = (
            prompt
            + "\n\nReturn output strictly as valid JSON matching this schema:\n"
            + json.dumps(schema_class.model_json_schema())
        )

        output_text = chatbot(full_prompt, return_full_text=False)[0]["generated_text"]

        # Extract JSON from output (robust to extra text)
        try:
            json_start = output_text.find("{")
            json_end = output_text.rfind("}") + 1
            data = json.loads(output_text[json_start:json_end])
            validated = schema_class.model_validate(data)
            results.append(validated)
        except (json.JSONDecodeError, ValidationError) as e:
            raise ValueError(f"Failed to parse structured output for prompt '{prompt}': {e}")

    return results

class ModalLLMService(LLMBase):  # type: ignore[misc]
    def __init__(self, model_name: str, temperature: float = 0.0,
                 max_new_tokens: int = 512, request_batch_size: int = 64) -> None:
        super().__init__(model_name=model_name, temperature=temperature)
        self._max_new_tokens = max_new_tokens
        self._request_batch_size = request_batch_size

    def _call(self, prompts: list[str]) -> list[str]:
        fn = modal.Function.from_name(APP_NAME, "generate")
        return fn.remote(prompts, self.model_name, self._max_new_tokens)  # type: ignore[attr-defined]

    def generate(self, prompt: str) -> str:
        return self._call([prompt])[0]

    def batch_generate(self, prompts: list[str], batch_size: int | None = None) -> list[str]:
        if not prompts:
            return []
        bs = batch_size or self._request_batch_size
        results: list[str] = []
        for i in tqdm(range(0, len(prompts), bs), desc=f"ModalLLM [{self.model_name}]", unit="batch"):
            results.extend(self._call(prompts[i : i + bs]))
        return results

    def generate_structured(self, prompt: str, schema: T) -> Any:
        fn = modal.Function.from_name(APP_NAME, "generate_structured")
        return fn.remote([prompt], self.model_name, self._max_new_tokens, schema)[0]  # type: ignore[attr-defined]

    def batch_generate_structured(self, prompts: list[str], schema: T, batch_size: int | None = None) -> Any:
        fn = modal.Function.from_name(APP_NAME, "generate_structured")
        bs = batch_size or self._request_batch_size
        results = []
        for i in tqdm(range(0, len(prompts), bs), desc=f"ModalLLM Structured [{self.model_name}]", unit="batch"):
            results.extend(fn.remote(prompts[i : i + bs], self.model_name, self._max_new_tokens, schema))  # type: ignore[attr-defined]
        return results
