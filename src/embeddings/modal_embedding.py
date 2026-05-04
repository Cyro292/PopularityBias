"""Modal GPU embeddings — LangChain Embeddings interface.

Setup:  modal deploy src/embeddings/modal_embedding.py
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
MODEL_NAME = "Lajavaness/bilingual-embedding-small"
# MODEL_NAME = "intfloat/multilingual-e5-small"
# MODEL_NAME = "intfloat/multilingual-e5-large"
GPU_CONFIG = "A10"
DEFAULT_GPU_BATCH_SIZE = 512  # Max batch size for A10, can be tuned based on actual GPU memory and model requirements
DEFAULT_BATCH_SIZE = 2048  # Max batch size for A10, can be tuned based on actual GPU memory and model requirements
MAX_CONTAINERS = 9         # Scaled up for index jobs
CONTAINER_TIMEOUT = 300     # 5 min idle timeout to reduce cold starts during gaps
FUNCTION_TIMEOUT = 300     # 5 min max execution time per call
MAX_RETRIES = 2           # Retry on failure

def download_model():
    from sentence_transformers import SentenceTransformer
    SentenceTransformer(MODEL_NAME, trust_remote_code=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "transformers==4.40.2", "sentence-transformers", "numpy",
        "langchain-elasticsearch", "elasticsearch"
    )
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
        self.model = SentenceTransformer(MODEL_NAME, device="cuda", trust_remote_code=True)
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

    @modal.method()
    def embed_and_index(
        self,
        texts: List[str],
        metadatas: List[dict],
        gpu_batch_size: int,
        normalise_embeddings: bool,
        es_url: str,
        index_name: str,
        strategy: str,
        distance_strategy: str | None,
        es_user: str | None,
        es_password: str | None,
        request_timeout: int,
        es_upload_batch: int,
    ) -> int:
        """Embed on GPU and upload directly to Elasticsearch — vectors never leave Modal."""
        import time as _time
        from elasticsearch import Elasticsearch, helpers

        n = len(texts)
        if n == 0:
            return 0

        # ── Step 1: embed on GPU ──────────────────────────────────────────
        vectors = self.model.encode(
            texts,
            batch_size=gpu_batch_size,
            normalize_embeddings=normalise_embeddings,
            show_progress_bar=False,
        ).tolist()

        # ── Step 2: build Elasticsearch Client ────────────────────────────
        es_params = {
            "hosts": [es_url],
            "request_timeout": request_timeout,
        }
        if es_user and es_password:
            es_params["basic_auth"] = (es_user, es_password)
        
        client = Elasticsearch(**es_params)

        # ── Step 3: upload using bulk helper ──────────────────────────────
        def generate_actions():
            for i in range(n):
                action = {
                    "_index": index_name,
                    "_source": {
                        "text": texts[i],
                        "vector": vectors[i],
                        **metadatas[i]
                    }
                }
                yield action

        success, _ = helpers.bulk(
            client,
            generate_actions(),
            chunk_size=es_upload_batch,
            stats_only=True,
            refresh=False
        )
        
        return success



# ── Client ────────────────────────────────────────────────────────────────────
class ModalEmbeddings(_EmbeddingsBase):
    def __init__(
        self,
        model_name: str,
        gpu_batch_size: int,
        request_batch_size : int,
        normalise_embeddings: bool = True,
    ):
        self.model_name = model_name
        self.gpu_batch_size = gpu_batch_size
        self.request_batch_size = request_batch_size
        self.normalise_embeddings = normalise_embeddings
        ModelService = modal.Cls.from_name(APP_NAME, "Model")
        self.service_instance = ModelService()
        self.embed_function = self.service_instance.embed
        self.embed_and_index_function = self.service_instance.embed_and_index
        logger.info(f"Initialized ModalEmbeddings with model {model_name} and GPU batch size {gpu_batch_size}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:

        import concurrent
        
        batches = [
            texts[i : i + self.request_batch_size]
            for i in range(0, len(texts), self.request_batch_size)
        ]
        
        # Prepare arguments as tuples for starmap
        map_args = [(batch, self.gpu_batch_size, self.normalise_embeddings) for batch in batches]

        # Offload the blocking Modal call to a separate thread
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

    # ── Embed + upload to Elasticsearch in one shot (runs on Modal) ─────

    def embed_and_send_documents(
        self,
        texts: List[str],
        metadatas: List[dict],
        *,
        es_url: str,
        index_name: str,
        strategy: str = "vector",
        distance_strategy: str | None = None,
        es_user: str | None = None,
        es_password: str | None = None,
        request_timeout: int = 600,
        es_upload_batch: int = 2000,
    ) -> int:
        """Embed texts on Modal GPU and upload directly to Elasticsearch.

        Everything runs on the Modal container — vectors never travel back
        to the client. Each batch is dispatched to a separate container
        via ``starmap``, embedding on GPU and pushing to ES in one hop.

        Returns:
            Number of documents successfully indexed.
        """
        import concurrent.futures

        n = len(texts)
        if n == 0:
            return 0

        # Split into request_batch_size chunks (same as embed_documents)
        bs = self.request_batch_size
        batches_texts = [texts[i : i + bs] for i in range(0, n, bs)]
        batches_metas = [metadatas[i : i + bs] for i in range(0, n, bs)]

        # Build starmap args — each tuple is one embed_and_index call
        map_args = [
            (
                batch_t,                    # texts
                batch_m,                    # metadatas
                self.gpu_batch_size,        # gpu_batch_size
                self.normalise_embeddings,  # normalise_embeddings
                es_url,
                index_name,
                strategy,
                distance_strategy,
                es_user,
                es_password,
                request_timeout,
                es_upload_batch,
            )
            for batch_t, batch_m in zip(batches_texts, batches_metas)
        ]

        logger.info(
            f"[embed_and_send] Dispatching {len(map_args)} batch(es) "
            f"({n:,} texts) to Modal → embed + ES upload …"
        )

        embed_and_index_fn = self.embed_and_index_function

        # Offload the blocking Modal starmap to a thread (Jupyter-safe)
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(map_args)) as executor:
            results = list(embed_and_index_fn.starmap(map_args))

        indexed = sum(results)
        logger.info(
            f"[embed_and_send] ✓ {indexed:,} docs indexed to '{index_name}' "
            f"(vectors stayed on Modal)"
        )
        return indexed
