"""Shared utilities for RAG services.

This module provides universal utilities that can be used across different RAG implementations:
- Embedding providers and rate limiting (for vector-based RAG)
- Text chunking and splitting (universal)
- Distance metrics and reranking (for vector-based RAG)
- Batch retrieval helpers (universal)
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_core.rate_limiters import BaseRateLimiter, InMemoryRateLimiter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ============================================================================
# EMBEDDINGS AND RATE LIMITING
# ============================================================================

_EMBEDDING_CLASSES: dict[str, type] = {}


def get_embedding_class(provider: str):
    """Lazy-load embedding classes to avoid import overhead.

    Args:
        provider: Name of the embedding provider ("openai", "google", "huggingface", "modal").

    Returns:
        The embedding class for the specified provider.

    Raises:
        ValueError: If the provider is not supported.
    """

    if not provider:
        raise ValueError("Provider Required")

    provider = provider.lower()
    if provider not in _EMBEDDING_CLASSES:
        if provider == "openai":
            from langchain_openai import OpenAIEmbeddings

            _EMBEDDING_CLASSES[provider] = OpenAIEmbeddings
        elif provider == "google":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            _EMBEDDING_CLASSES[provider] = GoogleGenerativeAIEmbeddings
        elif provider == "huggingface":
            from langchain_huggingface import HuggingFaceEmbeddings

            _EMBEDDING_CLASSES[provider] = HuggingFaceEmbeddings
        elif provider == "modal":
            from rag.ModalEmbedding import ModalEmbeddings

            _EMBEDDING_CLASSES[provider] = ModalEmbeddings
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")
    return _EMBEDDING_CLASSES[provider]



class RateLimitedEmbeddings:
    """Wrapper that adds rate limiting to any embeddings instance."""

    def __init__(self, embeddings: Any, rate_limiter: BaseRateLimiter):
        self._embeddings = embeddings
        self._rate_limiter = rate_limiter

    def _acquire(self) -> None:
        """Block until a rate limit token is available."""
        self._rate_limiter.acquire(blocking=True)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self._acquire()
        return self._embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        self._acquire()
        return self._embeddings.embed_query(text)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._embeddings, item)


def build_embeddings(
    provider: str = None,
    model: str = None,
    request_batch_size = None,
    gpu_batch_size = None,
    normalise_embeddings = None,
    trust_remote_code: bool = False,
    rate_limiter: BaseRateLimiter | None = None,
    requests_per_second: float | None = None,
    check_interval: float = 0.1,
    bucket_size: float = 1.0,
    **kwargs,
):
    """Build embeddings instance with optional rate limiting and E5 prefix handling.

    Args:
        provider: Embedding provider name ("openai", "google", "huggingface", "modal").
        model: Model name/identifier.
        trust_remote_code: Allow custom HuggingFace model code execution.
        rate_limiter: Pre-configured rate limiter (takes precedence).
        requests_per_second: Auto-create rate limiter with this rate.
        check_interval: Rate limit check interval.
        bucket_size: Token bucket size.
        **kwargs: Extra provider-specific options (e.g. modal_gpu_type).

    Returns:
        Embeddings instance, optionally wrapped with rate limiting.
    """
    embedding_cls = get_embedding_class(provider)

    if not model:
        raise ValueError("Model required for creating the embeddings model")

    if provider.lower() == "huggingface":
        base = embedding_cls(
            model_name=model,
            model_kwargs={"trust_remote_code": trust_remote_code},
        )
    elif provider.lower() == "modal":

        if request_batch_size is None or gpu_batch_size is None or normalise_embeddings is None:
            raise ValueError("Must provide both gpu_batch_size and request_batch_size and normalise_embeddings for modal")

        base = embedding_cls(
            model_name=model,
            request_batch_size=request_batch_size,
            gpu_batch_size=gpu_batch_size,
            normalise_embeddings=normalise_embeddings
        )
    else:
        base = embedding_cls(model=model)

    if rate_limiter:
        return RateLimitedEmbeddings(base, rate_limiter)

    if requests_per_second and requests_per_second > 0:
        limiter = InMemoryRateLimiter(
            requests_per_second=requests_per_second,
            check_every_n_seconds=check_interval,
            max_bucket_size=bucket_size,
        )
        return RateLimitedEmbeddings(base, limiter)

    return base


# ============================================================================
# TEXT PROCESSING AND CHUNKING
# ============================================================================


def split_documents(
    documents: Sequence[Document],
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[Document]:
    """Split documents into smaller chunks for better retrieval.

    Args:
        documents: Documents to split.
        chunk_size: Maximum characters per chunk.
        overlap: Number of characters to overlap between chunks.

    Returns:
        List of document chunks with preserved metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    return splitter.split_documents(list(documents))


# ============================================================================
# VECTOR STORE CREATION (Chroma-specific but could be generalized)
# ============================================================================


def create_chroma_index(
    documents: list[Document],
    embeddings: Any,
    collection_name: str,
    persist_dir: str | None = None,
    batch_size: int = 500,
    show_progress: bool = False,
    distance_function: str = "cosine",
) -> Chroma:
    """Create Chroma vector index with batched insertion and progress tracking.

    Args:
        documents: Documents to index (must be already chunked if desired).
        embeddings: Embeddings instance to use for vectorization.
        collection_name: Name for the Chroma collection.
        persist_dir: Optional directory to save the index to disk.
        batch_size: Number of documents to process in each batch.
        show_progress: Whether to display a progress bar.
        distance_function: Distance function to use ("cosine", "l2", "ip").

    Returns:
        Initialized Chroma vector store.

    Raises:
        ValueError: If no documents provided or invalid distance function.
    """
    if not documents:
        raise ValueError("No documents to index")

    valid_functions = ["cosine", "l2", "ip"]
    if distance_function not in valid_functions:
        raise ValueError(f"Invalid distance function: {distance_function}. Must be one of {valid_functions}")

    collection_metadata = {"hnsw:space": distance_function}

    if len(documents) <= batch_size or not show_progress:
        kwargs = {
            "documents": documents,
            "embedding": embeddings,
            "collection_name": collection_name,
            "collection_metadata": collection_metadata,
        }
        if persist_dir:
            kwargs["persist_directory"] = persist_dir
        return Chroma.from_documents(**kwargs)

    # Batched insertion for large datasets
    first_batch = documents[:batch_size]
    kwargs = {
        "documents": first_batch,
        "embedding": embeddings,
        "collection_name": collection_name,
        "collection_metadata": collection_metadata,
    }
    if persist_dir:
        kwargs["persist_directory"] = persist_dir

    vectorstore = Chroma.from_documents(**kwargs)

    remaining = documents[batch_size:]
    with tqdm(total=len(remaining), desc="Indexing documents", unit="docs") as pbar:
        for i in range(0, len(remaining), batch_size):
            batch = remaining[i : i + batch_size]
            vectorstore.add_documents(batch)
            pbar.update(len(batch))

    return vectorstore


def prepare_persist_dir(output_dir: Path | None) -> Path | None:
    """Prepare and validate the persistence directory for Chroma.

    Args:
        output_dir: Parent directory for the index. If None, returns None (in-memory index).

    Returns:
        Path to the chroma subdirectory, or None for in-memory operation.

    Raises:
        PermissionError: If the directory is not writable or cannot be accessed.
    """
    if not output_dir:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    persist_dir = output_dir / "chroma"

    if persist_dir.exists():
        logger.info(f"Deleting existing index at {persist_dir}")
        try:
            shutil.rmtree(persist_dir)
            time.sleep(0.1)
        except PermissionError as e:
            logger.error(f"Permission denied deleting {persist_dir}: {e}")
            raise

    persist_dir.mkdir(parents=True, exist_ok=True)
    if not os.access(persist_dir, os.W_OK):
        raise PermissionError(f"Directory {persist_dir} is not writable")

    # Test write capability
    test_file = persist_dir / ".write_test"
    try:
        test_file.touch()
        test_file.unlink()
    except Exception as e:
        raise PermissionError(f"Cannot write to {persist_dir}: {e}")

    return persist_dir


# ============================================================================
# DISTANCE METRICS AND RETRIEVAL
# ============================================================================


def compute_distance(query_vec: np.ndarray, doc_vec: np.ndarray, metric: str) -> float:
    """Compute distance between vectors.

    Args:
        query_vec: Query embedding vector.
        doc_vec: Document embedding vector.
        metric: Distance metric ("cosine", "l2", "ip").

    Returns:
        Distance score. Lower is better except for "ip" (higher is better).

    Raises:
        ValueError: If metric is not supported.
    """
    if metric == "cosine":
        similarity = np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
        return 1.0 - similarity
    if metric == "l2":
        return float(np.linalg.norm(query_vec - doc_vec))
    if metric == "ip":
        return -float(np.dot(query_vec, doc_vec))
    raise ValueError(f"Invalid metric: {metric}. Expected: cosine, l2, ip")


def rerank_with_metric(
    index: Any,
    embeddings: Any,
    text: str,
    top_k: int,
    metric: str,
    fetch_k: int,
) -> list[tuple[Document, float]]:
    """Fetch candidates with index's metric, then re-rank with requested metric.

    Args:
        index: Vector store to search.
        embeddings: Embeddings instance for computing query vector.
        text: Query text.
        top_k: Number of final results to return.
        metric: Desired distance metric for re-ranking.
        fetch_k: Number of candidates to fetch before re-ranking.

    Returns:
        List of (Document, score) tuples, ranked by the requested metric.
    """
    query_vec = np.array(embeddings.embed_query(text))
    candidates = index.similarity_search_by_vector(query_vec.tolist(), k=fetch_k)

    # Get embeddings for candidates
    cand_embeddings = []
    try:
        ids = [d.metadata.get("_id") for d in candidates if d.metadata.get("_id")]
        result = index._collection.get(ids=ids, include=["embeddings"])
        if result and result.get("embeddings"):
            cand_embeddings = result["embeddings"]
    except Exception:
        pass

    # Fallback: re-embed candidates
    if not cand_embeddings:
        cand_embeddings = [embeddings.embed_query(doc.page_content) for doc in candidates]

    scored = [(doc, compute_distance(query_vec, np.array(emb), metric)) for doc, emb in zip(candidates, cand_embeddings)]

    scored.sort(key=lambda x: x[1])
    if metric == "ip":
        return [(doc, -score) for doc, score in scored[:top_k]]
    return scored[:top_k]


def batch_retrieve(
    index: Any,
    embeddings: Any,
    questions: list[str],
    top_k: int = 5,
    batch_size: int = 32,
) -> list[list[tuple[Document, float]]]:
    """Batch retrieve documents for multiple queries using efficient embedding.

    Args:
        index: Vector store to search.
        embeddings: Embeddings instance for computing query vectors.
        questions: List of query texts.
        top_k: Number of results per query.
        batch_size: Number of queries to embed at once.

    Returns:
        List of results, one per query. Each result is a list of (Document, score) tuples.

    Raises:
        ValueError: If index is None.
    """
    if not index:
        raise ValueError("Index required")

    all_embeddings = []
    clean_questions = [str(q) if q is not None else "" for q in questions]

    iterator = range(0, len(clean_questions), batch_size)
    if len(clean_questions) > 100:
        iterator = tqdm(iterator, desc="Embedding queries", unit="batch")

    for i in iterator:
        batch = clean_questions[i : i + batch_size]
        if not batch:
            continue
        try:
            batch_embeddings = embeddings.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)
        except Exception as e:
            logger.error(f"Error embedding batch: {e}")
            raise e

    all_results = []
    has_vector_with_score = hasattr(index, "similarity_search_by_vector_with_score")

    for vec in all_embeddings:
        if has_vector_with_score:
            results = index.similarity_search_by_vector_with_score(vec, k=top_k)
        else:
            docs = index.similarity_search_by_vector(vec, k=top_k)
            results = [(d, 0.0) for d in docs]
        all_results.append(results)

    return all_results


def retrieve_topk_by_metric(
    index: Any,
    embeddings: Any,
    questions: list[str],
    expected_ids: list[str],
    top_k: int = 5,
    metrics: list[str] | None = None,
    batch_size: int = 64,
) -> list[dict]:
    """Retrieve top-k results for multiple distance metrics in one pass.

    Args:
        index: Vector store to search.
        embeddings: Embeddings instance.
        questions: Query texts.
        expected_ids: Expected document IDs for each query.
        top_k: Number of results per query.
        metrics: List of metrics to evaluate (e.g., ["cosine", "l2", "ip"]).
        batch_size: Number of queries to process at once.

    Returns:
        List of result dictionaries.

    Raises:
        ValueError: If index is None or questions/expected_ids length mismatch.
    """
    if not index:
        raise ValueError("Index required")

    if len(questions) != len(expected_ids):
        raise ValueError(f"Length mismatch: {len(questions)} questions vs {len(expected_ids)} expected IDs")

    results_data = []

    # Normalize metrics list
    metric_map = {"euclidean": "l2", "inner_product": "ip"}
    target_metrics = metrics if metrics else ["native"]
    target_metrics = [metric_map.get(m, m) for m in target_metrics]
    # Remove duplicates while preserving order
    seen = set()
    target_metrics = [m for m in target_metrics if not (m in seen or seen.add(m))]

    # Determine internal index metric
    index_metric = "cosine"
    if hasattr(index, "_collection"):
        metadata = getattr(index._collection, "metadata", None) or {}
        index_metric = metadata.get("hnsw:space", "cosine")

    total = len(questions)
    iterator = range(0, total, batch_size)
    if total > 100:
        desc = f"Evaluating ({', '.join(target_metrics)})"
        iterator = tqdm(iterator, desc=desc, unit="batch")

    for i in iterator:
        batch_qs = questions[i : i + batch_size]
        batch_ids = expected_ids[i : i + batch_size]

        clean_batch = [str(q) if q is not None else "" for q in batch_qs]
        try:
            batch_embeddings = embeddings.embed_documents(clean_batch)
        except Exception as e:
            logger.error(f"Error embedding batch {i}-{i+batch_size}: {e}")
            raise e

        has_vector_with_score = hasattr(index, "similarity_search_by_vector_with_score")

        for j, query_vec in enumerate(batch_embeddings):
            expected_id = str(batch_ids[j]).strip()
            query_vec_np = np.asarray(query_vec)

            # Determine whether we need re-ranking for non-native metrics
            needs_rerank = any(m not in ["native", index_metric] for m in target_metrics)
            candidates = None
            cand_embs_np = None
            fetch_k = top_k * 3

            if needs_rerank:
                candidates = index.similarity_search_by_vector(query_vec, k=fetch_k)
                # Get candidate embeddings
                cand_embeddings = []
                try:
                    ids = [d.metadata.get("_id") for d in candidates if d.metadata.get("_id")]
                    result = index._collection.get(ids=ids, include=["embeddings"])
                    if result and result.get("embeddings"):
                        cand_embeddings = result["embeddings"]
                except Exception:
                    pass

                if not cand_embeddings:
                    cand_embeddings = [embeddings.embed_query(doc.page_content) for doc in candidates]

                cand_embs_np = np.asarray(cand_embeddings)
                if cand_embs_np.size == 0:
                    cand_embs_np = None

            # Evaluate for EACH requested metric
            for current_metric in target_metrics:
                docs_and_scores = []

                if current_metric == "native" or current_metric == index_metric:
                    # Use fast native search
                    if has_vector_with_score:
                        docs_and_scores = index.similarity_search_by_vector_with_score(query_vec, k=top_k)
                    else:
                        docs = index.similarity_search_by_vector(query_vec, k=top_k)
                        docs_and_scores = [(d, 0.0) for d in docs]
                else:
                    # Re-rank using pre-fetched candidates/embeddings (vectorized)
                    if candidates is None or cand_embs_np is None:
                        docs_and_scores = []
                    else:
                        if current_metric == "cosine":
                            denom = np.linalg.norm(cand_embs_np, axis=1) * np.linalg.norm(query_vec_np)
                            denom[denom == 0] = 1e-12
                            scores = 1.0 - (cand_embs_np @ query_vec_np) / denom
                        elif current_metric == "l2":
                            scores = np.linalg.norm(cand_embs_np - query_vec_np, axis=1)
                        elif current_metric == "ip":
                            scores = -(cand_embs_np @ query_vec_np)
                        else:
                            scores = np.array([compute_distance(query_vec_np, emb, current_metric) for emb in cand_embs_np])

                        top_idx = np.argsort(scores)[:top_k]
                        if current_metric == "ip":
                            docs_and_scores = [(candidates[i], -scores[i]) for i in top_idx]
                        else:
                            docs_and_scores = [(candidates[i], scores[i]) for i in top_idx]

                topk_ids = []
                topk_scores = []
                topk_popularities = []

                for doc, score in docs_and_scores:
                    doc_id = doc.metadata.get("wikipedia_id", None)
                    if doc_id is None:
                        doc_id = doc.metadata.get("id", "")
                    topk_ids.append(str(doc_id).strip())
                    topk_scores.append(score)
                    topk_popularities.append(doc.metadata.get("popularity_avg", None))

                results_data.append(
                    {
                        "question": batch_qs[j],
                        "wikipedia_id": expected_id,
                        "metric": current_metric,
                        "topk_ids": topk_ids,
                        "topk_scores": topk_scores,
                        "topk_popularities": topk_popularities,
                    }
                )

    return results_data


# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass
class IndexingConfig:
    """Central configuration for the entire indexing pipeline.

    All tunable hyperparameters live here so the notebook controls everything.

    Groups:
        Text processing:  chunk_size, chunk_overlap
        ES bulk insert:   batch_size
        GPU embedding:    gpu_batch_size
        Embedding:        embedding_provider, embedding_model, distance_function
        Rate limiting:    rate_limiter, requests_per_second, ...
    """

    # ── Text processing ──────────────────────────────────────────────────
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # ── GPU embedding (runtime — no redeploy needed) ────────────────────
    gpu_batch_size: int = None       # forward-pass batch on GPU
    request_batch_size: int = None
    normalise_embeddings: bool = None

    # ── Embedding provider ──────────────────────────────────────────────
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-large"
    trust_remote_code: bool = False
    distance_function: str = "COSINE"

    # ── Prompt templates ─────────────────────────────────────────────────
    passage_prompt_file: str | None = None   # path to passage/embedding prompt file
    query_prompt_file: str | None = None     # path to query prompt file

    # ── Rate limiting ───────────────────────────────────────────────────
    rate_limiter: BaseRateLimiter | None = None
    requests_per_second: float | None = None
    rate_limit_check_interval: float = 0.1
    rate_limit_bucket_size: float = 1.0

    # ── UI ───────────────────────────────────────────────────────────────
    use_progress: bool = True
