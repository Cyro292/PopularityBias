"""RouterRagService — popularity-aware retrieval router.

Routes each query to either a dense or sparse backend using a TorchScript
classifier that takes the query text and the target article's popularity score
as input.

The classifier was trained with four labels:

    0 → bm25_plus   (sparse)
    1 → ivfpq_high  (dense)
    2 → ivfpq_low   (dense)
    3 → zero_shot   (no retrieval)

Labels 1 and 2 are both treated as "dense" and routed to ``dense_service``.
Label 0 is routed to ``sparse_service``.  Label 3 returns empty results.

This design makes the router backend-agnostic: you can pair it with FAISS
(``ivfpq_high``) or Elasticsearch (``es_approx``) as the dense backend and
compare fairly against ES hybrid.

Example — FAISS variant::

    router_faiss = RouterRagService(
        dense_service  = faiss_high,
        sparse_service = bm25_plus,
    )
    docs = router_faiss.retrieve_documents("Who wrote Hamlet?", popularity=25000.0)

Example — ES variant::

    router_es = RouterRagService(
        dense_service  = es_approx,
        sparse_service = bm25_plus,
    )
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import torch
from langchain.schema import Document
from sentence_transformers import SentenceTransformer

from .base import IndexResult, RagService

logger = logging.getLogger(__name__)

# ── Constants (from training) ─────────────────────────────────────────────────
_POP_MEAN:  float = 12313.55
_POP_STD:   float = 60130.65
_HF_REPO:   str   = "Cyro1/popularity_based_retrieval_predictor"
_HF_FILE:   str   = "best_backend_predictor_jit.pt"
_EMBEDDER:  str   = "all-MiniLM-L6-v2"

# Classifier label → routing decision.
# Labels 1 (ivfpq_high) and 2 (ivfpq_low) both map to "dense".
# Label 0 (bm25_plus) maps to "sparse".
# Label 3 (zero_shot) maps to "zero_shot" → empty results.
_LABEL_MAP: dict[int, str] = {
    0: "sparse",
    1: "dense",
    2: "dense",
    3: "zero_shot",
}


class RouterRagService(RagService):
    """Popularity-aware retrieval router.

    Wraps a dense and a sparse ``RagService`` backend and selects between them
    per-query using a lightweight TorchScript classifier.  The classifier was
    trained with FAISS labels but the routing decision is reduced to a binary
    dense/sparse choice, making this backend-agnostic (works with FAISS or ES).

    Args:
        dense_service:  Dense retrieval backend — e.g. ``FaissRagService``
                        (``ivfpq_high``) or ``ElasticsearchRagService``
                        (``approximation``).  Required.
        sparse_service: Sparse retrieval backend — e.g. ``BM25RagService``
                        (``bm25_plus``).  Required.
        device:         Torch device for the classifier (default ``"cpu"``).
    """

    def __init__(
        self,
        *,
        dense_service:  RagService,
        sparse_service: RagService,
        device: str = "cpu",
    ) -> None:
        self._backends: dict[str, RagService] = {
            "dense":  dense_service,
            "sparse": sparse_service,
        }
        self._device = device

        logger.info("RouterRagService — loading classifier from HuggingFace Hub…")
        self._model    = self._load_model()
        self._embedder = SentenceTransformer(_EMBEDDER, device=device)
        logger.info("RouterRagService — ready (dense=%s, sparse=%s)",
                    type(dense_service).__name__, type(sparse_service).__name__)

    # ── Router internals ──────────────────────────────────────────────────────

    def _load_model(self) -> torch.jit.ScriptModule:
        """Download and load the TorchScript classifier.

        Returns:
            Loaded TorchScript model in eval mode.
        """
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(_HF_REPO, _HF_FILE)
        model = torch.jit.load(path, map_location=self._device)
        model.eval()
        return model

    def _predict_backend(self, query: str, popularity: float) -> str:
        """Run the classifier for a single query and return the backend key.

        Args:
            query:      Query string.
            popularity: Article popularity score (raw pageviews/month).

        Returns:
            One of ``"bm25_plus"``, ``"ivfpq_high"``, ``"ivfpq_low"``,
            ``"zero_shot"``.
        """
        with torch.no_grad():
            pop_scaled = (popularity - _POP_MEAN) / _POP_STD
            pop_tensor = torch.tensor([[pop_scaled]], dtype=torch.float32)
            text_emb   = self._embedder.encode(
                [query], convert_to_tensor=True, device=self._device
            )
            logits = self._model(torch.cat([pop_tensor, text_emb], dim=1))
            label  = int(torch.argmax(logits, dim=1).item())
        return self._label_to_backend(label)

    def _predict_backends_batch(
        self,
        queries: list[str],
        popularities: list[float],
    ) -> list[str]:
        """Run the classifier for a batch of queries in one forward pass.

        Embeds all queries at once, builds a single feature matrix, and
        runs the TorchScript model once.

        Args:
            queries:      List of query strings.
            popularities: Corresponding popularity scores (same length).

        Returns:
            List of backend keys, one per query.
        """
        with torch.no_grad():
            pop_scaled = torch.tensor(
                [[(p - _POP_MEAN) / _POP_STD] for p in popularities],
                dtype=torch.float32,
            )  # (N, 1)
            text_embs = self._embedder.encode(
                queries,
                convert_to_tensor=True,
                device=self._device,
                show_progress_bar=False,
            )  # (N, D)
            features = torch.cat([pop_scaled, text_embs], dim=1)  # (N, D+1)
            logits   = self._model(features)                       # (N, num_classes)
            labels   = torch.argmax(logits, dim=1).tolist()
        return [self._label_to_backend(lbl) for lbl in labels]

    def _label_to_backend(self, label: int) -> str:
        """Map a classifier label index to ``"dense"``, ``"sparse"``, or ``"zero_shot"``.

        Args:
            label: Integer class index from the classifier.

        Returns:
            ``"dense"``, ``"sparse"``, or ``"zero_shot"``.
        """
        return _LABEL_MAP.get(label, "sparse")

    def _resolve(self, query: str, popularity: float) -> tuple[str, RagService | None]:
        """Return (backend_key, service) for a single query.

        Returns ``(key, None)`` when the predicted backend is ``"zero_shot"``
        or otherwise not registered — callers must return empty results in
        that case.

        Args:
            query:      Query string.
            popularity: Popularity score (required).

        Returns:
            Tuple of backend key and service, or ``(key, None)`` for zero_shot.
        """
        key     = self._predict_backend(query, popularity)
        service = self._backends.get(key)
        logger.debug("Router → %s  (popularity=%.1f, query=%r)", key, popularity, query[:60])
        return key, service

    # ── RagService interface ──────────────────────────────────────────────────

    def retrieve_documents(
        self,
        query: str,
        *,
        top_k: int = 5,
        popularity: float,
        **kwargs: Any,
    ) -> list[Document]:
        """Route query to the predicted backend and retrieve documents.

        Args:
            query:      Query string.
            top_k:      Number of documents to return.
            popularity: Article popularity (pageviews/month). Required.
            **kwargs:   Passed through to the selected backend.

        Returns:
            List of ``Document`` objects ranked by relevance.
        """
        _, service = self._resolve(query, popularity)
        if service is None:
            return []
        return service.retrieve_documents(query, top_k=top_k, **kwargs)

    def retrieve_documents_with_scores(
        self,
        query: str,
        *,
        top_k: int = 5,
        popularity: float,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Route query and retrieve documents with relevance scores.

        Args:
            query:      Query string.
            top_k:      Number of results to return.
            popularity: Article popularity score. Required.
            **kwargs:   Passed through to the selected backend.

        Returns:
            List of ``(Document, score)`` tuples ranked best-first.
        """
        _, service = self._resolve(query, popularity)
        if service is None:
            return []
        return service.retrieve_documents_with_scores(query, top_k=top_k, **kwargs)

    def retrieve_documents_with_backend(
        self,
        query: str,
        *,
        top_k: int = 5,
        popularity: float,
        **kwargs: Any,
    ) -> tuple[str, list[Document]]:
        """Retrieve documents and also return which backend was chosen.

        Args:
            query:      Query string.
            top_k:      Number of documents to return.
            popularity: Article popularity score. Required.
            **kwargs:   Passed through to the selected backend.

        Returns:
            Tuple of ``(backend_key, documents)``.
        """
        key, service = self._resolve(query, popularity)
        if service is None:
            return key, []
        docs = service.retrieve_documents(query, top_k=top_k, **kwargs)
        return key, docs

    def batch_retrieve(
        self,
        queries: list[str],
        *,
        top_k: int = 5,
        popularities: list[float],
        progress_bar: bool = True,
        **kwargs: Any,
    ) -> list[list[Document]]:
        """Route and retrieve for a list of queries efficiently.

        Embeds all queries in one pass, predicts all backends at once, groups
        queries by backend, calls each backend once with its group, then
        reassembles results in original order.

        Args:
            queries:      List of query strings.
            top_k:        Number of results per query.
            popularities: Per-query popularity scores. Required. Must match
                          length of ``queries``.
            progress_bar: Show tqdm progress bar over backends.
            **kwargs:     Passed through to each backend.

        Returns:
            List of document lists — one per query, in original order.

        Raises:
            ValueError: If ``popularities`` length does not match ``queries``.
        """
        scored = self.batch_retrieve_with_scores(
            queries,
            top_k=top_k,
            popularities=popularities,
            progress_bar=progress_bar,
            **kwargs,
        )
        return [[doc for doc, _ in group] for group in scored]

    def batch_retrieve_with_scores(
        self,
        queries: list[str],
        *,
        top_k: int = 5,
        popularities: list[float],
        progress_bar: bool = True,
        **kwargs: Any,
    ) -> list[list[tuple[Document, float]]]:
        """Route and retrieve scored results efficiently.

        Embeds all queries in one forward pass, predicts all backends at once,
        groups queries by predicted backend, calls each backend's
        ``batch_retrieve_with_scores`` once, then reassembles in original order.

        Args:
            queries:      List of query strings.
            top_k:        Number of results per query.
            popularities: Per-query popularity scores. Required. Must match
                          length of ``queries``.
            progress_bar: Show tqdm progress bar over backends.
            **kwargs:     Passed through to each backend.

        Returns:
            List of scored result lists — one per query, in original order.

        Raises:
            ValueError: If ``popularities`` length does not match ``queries``.
        """
        from collections import defaultdict
        from tqdm import tqdm as _tqdm

        if len(popularities) != len(queries):
            raise ValueError(
                f"popularities length ({len(popularities)}) must match "
                f"queries length ({len(queries)})"
            )

        n = len(queries)

        # ── Step 1: batch-predict all backends in one forward pass ────────────
        logger.info("Router — predicting backends for %d queries…", n)
        backend_keys = self._predict_backends_batch(queries, popularities)

        # Log routing distribution
        from collections import Counter
        dist = Counter(backend_keys)
        logger.info("Router — routing distribution: %s", dict(dist))

        # ── Step 2: group original indices by backend ─────────────────────────
        groups: dict[str, list[int]] = defaultdict(list)
        for i, key in enumerate(backend_keys):
            groups[key].append(i)

        # ── Step 3: batch-retrieve per backend ────────────────────────────────
        results: list[list[tuple[Document, float]]] = [[] for _ in range(n)]

        for backend_key, indices in _tqdm(
            groups.items(),
            desc="RouterRag backends",
            disable=not progress_bar,
        ):
            # zero_shot → return empty results (Recall@0 is the correct score)
            if backend_key == "zero_shot" or backend_key not in self._backends:
                logger.info(
                    "Router — %s: %d queries → [] (no retrieval)",
                    backend_key, len(indices),
                )
                continue  # results[i] already initialised to []

            service  = self._backends[backend_key]
            batch_qs = [queries[i] for i in indices]
            logger.info(
                "Router — %s: retrieving %d queries…", backend_key, len(batch_qs)
            )
            batch_scored = service.batch_retrieve_with_scores(
                batch_qs, top_k=top_k, **kwargs
            )
            for i, scored_docs in zip(indices, batch_scored):
                results[i] = scored_docs

        return results

    # ── Index lifecycle — delegate to all backends ────────────────────────────

    def load_index(self, path_or_name: str | Path, **kwargs: Any) -> Any:
        """Not supported — backends must be loaded individually before init.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "RouterRagService does not support load_index(). "
            "Load each backend individually before passing them to RouterRagService."
        )

    def index_from_dataframe(self, df, text_field, **kwargs) -> IndexResult:
        raise NotImplementedError("RouterRagService does not support indexing directly.")

    def index_from_parquet(self, parquet_path, **kwargs) -> IndexResult:
        raise NotImplementedError("RouterRagService does not support indexing directly.")

    # ── Inspection ────────────────────────────────────────────────────────────

    def get_doc_count(self) -> int:
        """Return the doc count of the dense backend.

        Returns:
            Document count from the dense backend.
        """
        return self._backends["dense"].get_doc_count()

    def get_index_stats(self) -> dict[str, Any]:
        """Return stats from all backends.

        Returns:
            Dict keyed by backend name, each value is that backend's stats dict.
        """
        return {key: svc.get_index_stats() for key, svc in self._backends.items()}
