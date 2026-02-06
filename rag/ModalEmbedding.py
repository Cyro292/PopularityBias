"""Modal GPU embeddings — LangChain Embeddings interface.

Setup:  modal deploy rag/ModalEmbedding.py
Usage:  ModalEmbeddings(model_name="intfloat/multilingual-e5-small")
"""
from __future__ import annotations

from typing import List
import modal

try:
    from langchain_core.embeddings import Embeddings as _EmbeddingsBase
except ImportError:
    _EmbeddingsBase = object

# ── Configuration ─────────────────────────────────────────────────────────────
APP_NAME = "PopularityBias_Thesis_Amon_Embedding_Service"
MODEL_NAME = "intfloat/multilingual-e5-small"
GPU_CONFIG = "T4"
DEFAULT_BATCH_SIZE = 128
MAX_CONTAINERS = 4         # Scaled up for index jobs
CONTAINER_TIMEOUT = 300     # 5 min idle timeout to reduce cold starts during gaps
FUNCTION_TIMEOUT = 3600     # 1 hour max execution time per call

def download_model():
    from sentence_transformers import SentenceTransformer
    # This downloads the model to the local cache (~/.cache/huggingface)
    SentenceTransformer(MODEL_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "sentence-transformers", "numpy")
    .env({"PYTORCH_ALLOC_CONF": "expandable_segments:True"})
    .run_function(download_model)
)

app = modal.App(APP_NAME)

@app.cls(
    image=image,
    gpu=GPU_CONFIG,
    timeout=FUNCTION_TIMEOUT,
    max_containers=MAX_CONTAINERS,
    scaledown_window=CONTAINER_TIMEOUT, # Keep warm for 5 mins
)
class Model:
    @modal.enter()  # <--- THIS DECORATOR IS MISSING
    def enter(self):
        # This runs ONCE when container starts
        print(f"Loading model {MODEL_NAME}...")
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(MODEL_NAME)
        print("Model loaded!")

    @modal.method()
    def embed(
        self,
        texts: List[str],
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> List[List[float]]:
        # This uses the pre-loaded model immediately
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False 
        )
        return embeddings.tolist()



# ── Client ────────────────────────────────────────────────────────────────────
class ModalEmbeddings(_EmbeddingsBase):
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        gpu_batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        self.model_name = model_name
        self.gpu_batch_size = gpu_batch_size
        ModelService = modal.Cls.from_name(APP_NAME, "Model")
        self.service_instance = ModelService()
        self.embed_function = self.service_instance.embed

    def embed_documents(self, texts: List[str]) -> List[List[float]]:

        import concurrent.futures

        # Split texts into batches
        batches = [
            texts[i : i + self.gpu_batch_size]
            for i in range(0, len(texts), self.gpu_batch_size)
        ]
        
        # Prepare arguments as tuples for starmap
        map_args = [(batch, self.gpu_batch_size) for batch in batches]

        def _run_sync_starmap():
            # This runs inside a thread, isolated from Jupyter's loop
            return list(self.embed_function.starmap(map_args))

        # Offload the blocking Modal call to a separate thread
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run_sync_starmap)
            embeddings_list = future.result()
        
        # Flatten results
        embeddings = [emb for batch_embeddings in embeddings_list for emb in batch_embeddings]
        return embeddings
    
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError("Async embedding not implemented for ModalEmbeddings")

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

