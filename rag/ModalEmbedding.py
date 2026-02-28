"""Modal GPU embeddings — LangChain Embeddings interface.

Setup:  modal deploy rag/ModalEmbedding.py
Usage:  ModalEmbeddings(model_name="intfloat/multilingual-e5-small")
"""
from __future__ import annotations

from typing import List
import modal
import logging

logger = logging.getLogger(__name__)

try:
    from langchain_core.embeddings import Embeddings as _EmbeddingsBase
except ImportError:
    _EmbeddingsBase = object

# ── Configuration ─────────────────────────────────────────────────────────────
APP_NAME = "PopularityBias_Thesis_Amon_Embedding_Service"
# MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_NAME = "intfloat/multilingual-e5-large"
# MODEL_NAME = "intfloat/multilingual-e5-small"
GPU_CONFIG = "A10"
DEFAULT_GPU_BATCH_SIZE = 512  # Max batch size for A10, can be tuned based on actual GPU memory and model requirements
DEFAULT_BATCH_SIZE = 2048  # Max batch size for A10, can be tuned based on actual GPU memory and model requirements
MAX_CONTAINERS = 9         # Scaled up for index jobs
CONTAINER_TIMEOUT = 300     # 5 min idle timeout to reduce cold starts during gaps
FUNCTION_TIMEOUT = 300     # 5 min max execution time per call
MAX_RETRIES = 2           # Retry on failure

def download_model():
    from sentence_transformers import SentenceTransformer
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
    retries=MAX_RETRIES
)
class Model:
    @modal.enter()  
    def enter(self):
        # This runs ONCE when container starts
        print(f"Loading model {MODEL_NAME}...")
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(MODEL_NAME, device="cuda")
        print("Model loaded!")

    @modal.method()
    def embed(
        self,
        texts: List[str],
        gpu_batch_size: int,
        normalise_embeddings: bool
    ) -> List[List[float]]:
        # This uses the pre-loaded model immediately
        embeddings = self.model.encode(
            texts,
            batch_size=gpu_batch_size,
            normalize_embeddings=normalise_embeddings,
            show_progress_bar=False 
        )
        return embeddings.tolist()



# ── Client ────────────────────────────────────────────────────────────────────
class ModalEmbeddings(_EmbeddingsBase):
    def __init__(
        self,
        model_name: str = None,
        gpu_batch_size: int = None,
        request_batch_size : int = None,
        normalise_embeddings: bool = True,
    ):
        if not model_name or not gpu_batch_size or not request_batch_size:
            raise ValueError("Must provide model_name, gpu_batch_size and request_batch_size for ModalEmbeddings")
        

        self.model_name = model_name
        self.gpu_batch_size = gpu_batch_size
        self.request_batch_size = request_batch_size
        self.normalise_embeddings = normalise_embeddings
        ModelService = modal.Cls.from_name(APP_NAME, "Model")
        self.service_instance = ModelService()
        self.embed_function = self.service_instance.embed
        logger.info(f"Initialized ModalEmbeddings with model {model_name} and GPU batch size {gpu_batch_size}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:

        import concurrent
        
        batches = [
            texts[i : i + self.request_batch_size]
            for i in range(0, len(texts), self.request_batch_size)
        ]
        
        # Prepare arguments as tuples for starmap
        map_args = [(batch, self.gpu_batch_size, self.normalise_embeddings) for batch in batches]

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

