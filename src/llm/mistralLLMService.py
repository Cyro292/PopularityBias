from typing import Any, TypeVar
from pydantic import BaseModel
from tqdm import tqdm
import modal

if not modal.is_local():
    LLMBase = object
else:
    from src.llm.base import LLMBase

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

@app.function(gpu="A100", image=image, timeout=600, max_containers=3)
def generate(prompts: list[str], model_name: str = MODEL_NAME, max_new_tokens: int = 256) -> list[str]:
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

@app.function(gpu="A100", image=image, timeout=600, max_containers=3)
def generate_structured(
    prompts: list[str],
    schema_class: type[BaseModel],
    *,
    model_name: str = MODEL_NAME,
    max_new_tokens: int = 256
) -> list:
    import outlines
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = outlines.from_transformers(
        AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda"),
        AutoTokenizer.from_pretrained(model_name),
    )

    results = []

    for prompt in tqdm(prompts, desc="Generating structured responses"):
        result = model(
            f"{prompt}",
            schema_class,
            max_new_tokens=max_new_tokens,
        )
        results.append(result)

    return results
    

class MistralLLMService(LLMBase):  # type: ignore[misc]
    def __init__(self, temperature: float = 0.0, request_batch_size: int = 64) -> None:
        super().__init__(model_name=MODEL_NAME, temperature=temperature)
        self._request_batch_size = request_batch_size

        self.generate_fn = modal.Function.from_name(APP_NAME, "generate")
        self.generate_structured_fn = modal.Function.from_name(APP_NAME, "generate_structured")

    def generate(self, prompt: str) -> str:
        return list(self.generate_fn.map(
                [[prompt]],  # prompts
                [self.model_name],
                [256],
            )
        )[0]

    def batch_generate(self, prompts: list[str]) -> list[str]:
        # split into batches
        batched_prompts = [
            prompts[i:i + self._request_batch_size]
            for i in range(0, len(prompts), self._request_batch_size)
        ]

        results = []
        for batch_result in tqdm(
            self.generate_fn.map(
                batched_prompts,
                [self.model_name] * len(batched_prompts),
                [256] * len(batched_prompts),
            ),
            total=len(batched_prompts),
            desc="Generating text (parallel)",
            unit="batch",
        ):
            results.extend(batch_result)

        return results

    def generate_structured(self, prompt: str, schema: T) -> Any:
        results = list(
            self.generate_structured_fn.map(
                [[prompt]],
                [schema],
                [self.model_name],
                [256],
            )
        )
        return results[0][0]

    def batch_generate_structured(self, prompts: list[str], schema: T) -> list[Any]:
        batched_prompts = [
            prompts[i:i + self._request_batch_size]
            for i in range(0, len(prompts), self._request_batch_size)
        ]

        results = []

        for batch_result in tqdm(
            self.generate_structured_fn.map(
                batched_prompts,
                [schema] * len(batched_prompts),
                [self.model_name] * len(batched_prompts),
                [256] * len(batched_prompts),
            ),
            total=len(batched_prompts),
            desc="Generating structured (parallel)",
            unit="batch",
        ):
            results.extend(batch_result)

        return results
    
    