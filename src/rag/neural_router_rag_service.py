"""NeuralRouterRagService — neural router with strict/hybrid modes.

Uses a trained BERT-based classifier (via Modal GPU) to route queries between
retrieval backends.  All model loading and inference is delegated to
``RouterService`` (``src/router/router_service.py``); this class only manages
the routing logic, RRF fusion, and retrieval delegation.

Supports two modes:

1. **Strict Mode** (`strict=True`):
   - Uses argmax to select single backend with highest probability
   - Routes query to that backend only
   - Fast, single retrieval per query

2. **Hybrid Mode** (`strict=False`):
   - Uses softmax probabilities as weights for RRF fusion
   - Retrieves from all backends and fuses with probability-weighted RRF
   - Score = prob_backend1 * (1/(k+rank1)) + prob_backend2 * (1/(k+rank2)) + ...
   - More robust, considers multiple backends

Example — Strict Mode::

    from src.rag import NeuralRouterRagService
    
    router = NeuralRouterRagService(
        backends={"bm25_plus": bm25_svc, "ivfpq_high": faiss_svc},
        backend_order=["bm25_plus", "ivfpq_high"],
        model_path="models/router_mrr20.pt",
        strict=True,
    )
    docs = router.retrieve_documents("Who wrote Hamlet?", popularity=5000.0, top_k=10)

Example — Hybrid Mode::

    router = NeuralRouterRagService(
        backends={"bm25_plus": bm25_svc, "ivfpq_high": faiss_svc},
        backend_order=["bm25_plus", "ivfpq_high"],
        model_path="models/router_mrr20.pt",
        strict=False,
        rrf_k=60,
        rrf_depth=60,
    )
    docs = router.retrieve_documents("Who wrote Hamlet?", popularity=5000.0, top_k=10)

Training the router model::

    python -m src.router.train_router \\
        --label-mode retrieval \\
        --retrieval-metric mrr \\
        --retrieval-k 20 \\
        --backends bm25_plus ivfpq_high \\
        --model-name router_mrr20
"""
from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from langchain.schema import Document
from tqdm import tqdm

from .base import IndexResult, RagService

if TYPE_CHECKING:
    from .router_rag_service import RouterRagService

logger = logging.getLogger(__name__)


class NeuralRouterRagService(RagService):
    """Neural router with strict (argmax) or hybrid (probability-weighted RRF) modes.
    
    Routes queries between multiple retrieval backends using a trained BERT
    classifier.  Model inference is delegated to ``RouterService`` on Modal
    GPU — no local BERT/classifier loading.

    A classic ``RouterRagService`` vote can optionally be blended into the
    neural probabilities via ``assist_weight``.

    Args:
        backends: Dict mapping backend names to loaded RagService instances.
            Example: {"bm25_plus": bm25_svc, "ivfpq_high": faiss_svc}.
            May be empty when ``set_precomputed_results()`` will be called.
        backend_order: List of backend names in the order the model was trained.
            Model output index i corresponds to backend_order[i].
            Example: ["bm25_plus", "ivfpq_high"]
        model_path: Path to trained model .pt file (from train_router.py).
        strict: If True, use argmax (select single backend). If False, use
            probability-weighted RRF (retrieve from all backends and fuse).
        rrf_k: RRF smoothing constant (only for hybrid mode). Default 60.
        rrf_depth: Number of candidates to fetch per backend before fusion
            (only for hybrid mode). Default 60.
        predict_batch_size: Batch size for Modal inference. Default 32.
        device: Torch device (used only for the classic assist router).
            Default "cpu".
        assist_weight: Blend weight for the classic RouterRagService vote.
            The final probability is ``(1 - w) * neural_probs + w * classic_onehot``.
            Set to 0.0 to disable. Only active when ``backend_order`` has
            exactly 2 entries. Default 0.25.
    """
    
    def __init__(
        self,
        *,
        backends: dict[str, RagService] | None = None,
        backend_order: list[str],
        model_path: str | Path,
        strict: bool = True,
        rrf_k: int = 60,
        rrf_depth: int = 60,
        predict_batch_size: int = 32,
        device: str = "cpu",
        assist_weight: float = 0.25,
    ) -> None:
        if not 0.0 <= assist_weight <= 1.0:
            raise ValueError(f"assist_weight must be in [0, 1], got {assist_weight}")

        self.backends = backends or {}
        self.backend_order = backend_order
        self.model_path = Path(model_path)
        self.strict = strict
        self.rrf_k = rrf_k
        self.rrf_depth = rrf_depth
        self.predict_batch_size = predict_batch_size
        self.device = device
        self.assist_weight = assist_weight
        self.precomputed_results: dict[str, list[list[Document]]] | None = None
        self._assist_backend_map = self._infer_assist_backend_map()
        self._assist_router: RouterRagService | None = None
        
        # Validate backend_order matches backends (skip when backends intentionally empty)
        if self.backends:
            for name in backend_order:
                if name not in self.backends:
                    raise ValueError(f"Backend '{name}' in backend_order not found in backends dict")
        
        # Load model dict (weights + config only — no BERT/classifier construction).
        # Actual inference is delegated to RouterService on Modal.
        logger.info(f"NeuralRouterRagService — loading model dict from {model_path}...")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        self._model_dict = torch.load(self.model_path, map_location="cpu")

        # Initialize Modal-based prediction client
        from src.router.router_service import RouterService
        self._router_service = RouterService()

        include_pop = self._model_dict.get("model_config", {}).get("include_popularity", True)
        
        mode_str = "strict (argmax)" if strict else f"hybrid (probability-weighted RRF, k={rrf_k})"
        pop_str = "with popularity" if include_pop else "BERT-only (no popularity)"
        assist_str = (
            f", assist_weight={assist_weight}"
            if assist_weight > 0 and len(backend_order) == 2
            else ", assist=disabled"
        )
        logger.info(
            f"NeuralRouterRagService ready — mode={mode_str}, "
            f"backends={backend_order}, {pop_str}{assist_str}, device={device}"
        )
    
    # ── Pre-computed results ───────────────────────────────────────────────────
    
    def set_precomputed_results(
        self, results: dict[str, list[list[Document]]]
    ) -> None:
        """Supply cached sub-backend results so live retrieval is skipped.

        Args:
            results: Dict keyed by backend name (matching ``backend_order``).
                Each value is a list (one per query) of Document lists.
        """
        self.precomputed_results = results
        logger.info(
            f"NeuralRouterRagService — pre-computed results set for "
            f"backends {list(results.keys())} — live retrieval will be skipped"
        )
    
    # ── Assist router (classic RouterRagService) ───────────────────────────────

    def _infer_assist_backend_map(self) -> dict[str, str]:
        """Map classic router labels onto the neural router backend keys."""
        if len(self.backend_order) < 2:
            return {}

        sparse_candidates = [
            name for name in self.backend_order
            if "bm25" in name.lower() or "sparse" in name.lower()
        ]
        sparse_key = sparse_candidates[0] if sparse_candidates else self.backend_order[0]
        dense_candidates = [name for name in self.backend_order if name != sparse_key]
        dense_key = dense_candidates[0] if dense_candidates else sparse_key
        return {
            "sparse": sparse_key,
            "dense": dense_key,
        }

    def _build_assist_router(self) -> RouterRagService | None:
        """Build the classic router used to assist neural predictions."""
        if self.assist_weight == 0.0:
            return None

        if len(self.backend_order) != 2:
            logger.warning(
                "NeuralRouterRagService — classic router assist disabled: "
                "backend_order must contain exactly 2 backends, got %d",
                len(self.backend_order),
            )
            return None

        from .router_rag_service import RouterRagService

        dense_service = self.backends.get(self._assist_backend_map["dense"])
        sparse_service = self.backends.get(self._assist_backend_map["sparse"])
        return RouterRagService(
            dense_service=dense_service,
            sparse_service=sparse_service,
            device=self.device,
        )

    def _get_assist_router(self) -> RouterRagService | None:
        """Return the lazily-built classic router assist instance."""
        if self._assist_router is None:
            self._assist_router = self._build_assist_router()
        return self._assist_router

    def _apply_router_assist(
        self,
        probabilities: list[dict[str, float]],
        queries: list[str],
        popularities: list[float],
    ) -> list[dict[str, float]]:
        """Blend neural probabilities with the classic router's hard vote."""
        assist_router = self._get_assist_router()
        if assist_router is None or not probabilities:
            return probabilities

        router_votes = assist_router.predict_backends_batch(queries, popularities)
        logger.info(
            "NeuralRouter — classic router assist distribution: %s",
            dict(Counter(router_votes)),
        )

        assisted: list[dict[str, float]] = []
        for prob_dict, router_vote in zip(probabilities, router_votes):
            mapped_vote = self._assist_backend_map.get(router_vote)
            if mapped_vote is None or mapped_vote not in prob_dict:
                assisted.append(prob_dict)
                continue

            blended = {
                backend_name: prob * (1.0 - self.assist_weight)
                for backend_name, prob in prob_dict.items()
            }
            blended[mapped_vote] += self.assist_weight
            total = sum(blended.values())
            if total > 0:
                blended = {
                    backend_name: prob / total
                    for backend_name, prob in blended.items()
                }
            assisted.append(blended)
        return assisted
    
    # ── Prediction (delegated to RouterService on Modal) ───────────────────────

    def _predict_probabilities_batch(
        self,
        queries: list[str],
        popularities: list[float],
    ) -> list[list[float]]:
        """Run the neural classifier via Modal and return softmax probabilities.

        Args:
            queries: List of query texts.
            popularities: Per-query popularity scores.

        Returns:
            List of probability lists (one per query, one float per backend
            class in ``backend_order`` order).
        """
        if not queries:
            return []

        cfg = self._model_dict
        return self._router_service.predict(
            questions=queries,
            popularity=popularities,
            classifier_state=cfg["classifier_state"],
            scaler_mean=cfg["scaler_mean"],
            scaler_scale=cfg["scaler_scale"],
            model_config=cfg["model_config"],
            batch_size=self.predict_batch_size,
        )

    def _predict_single(
        self,
        query: str,
        popularity: float,
    ) -> str | dict[str, float]:
        """Predict backend for a single query.
        
        Args:
            query: Query text
            popularity: Popularity score
        
        Returns:
            If strict: backend name (str)
            If hybrid: dict of {backend_name: probability}
        """
        return self._predict_batch([query], [popularity])[0]
    
    def _predict_batch(
        self,
        queries: list[str],
        popularities: list[float],
    ) -> list[str | dict[str, float]]:
        """Predict backends for a batch of queries via Modal.
        
        Args:
            queries: List of query texts
            popularities: List of popularity scores (same length)
        
        Returns:
            If strict: list of backend names
            If hybrid: list of dicts {backend_name: probability}
        """
        prob_lists = self._predict_probabilities_batch(queries, popularities)
        prob_dicts = [
            {
                self.backend_order[j]: prob_lists[i][j]
                for j in range(len(self.backend_order))
            }
            for i in range(len(queries))
        ]
        prob_dicts = self._apply_router_assist(prob_dicts, queries, popularities)

        if self.strict:
            return [
                max(prob_dict.items(), key=lambda item: item[1])[0]
                for prob_dict in prob_dicts
            ]
        return prob_dicts
    
    # ── RRF Fusion (Hybrid Mode) ─────────────────────────────────────────────
    
    def _fuse_with_probabilities(
        self,
        backend_results: dict[str, list[tuple[Document, float]]],
        probabilities: dict[str, float],
        top_k: int,
    ) -> list[tuple[Document, float]]:
        """Apply probability-weighted RRF fusion.
        
        Each document's score is the sum of its weighted RRF contributions:
            score = sum(prob_backend * 1/(rrf_k + rank_in_backend))
        
        Args:
            backend_results: Dict of {backend_name: [(doc, score), ...]}
            probabilities: Dict of {backend_name: probability}
            top_k: Number of results to return
        
        Returns:
            Top-k [(Document, weighted_rrf_score), ...] sorted descending
        """
        k = self.rrf_k
        scores: dict[str, float] = {}
        docs: dict[str, Document] = {}
        
        for backend_name, results in backend_results.items():
            prob = probabilities.get(backend_name, 0.0)
            if prob == 0:
                continue
            
            for rank, (doc, _) in enumerate(results, start=1):
                key = doc.page_content  # Deduplicate by content
                contribution = prob * (1.0 / (k + rank))
                scores[key] = scores.get(key, 0.0) + contribution
                docs[key] = doc
        
        # Sort by fused score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(docs[key], score) for key, score in ranked[:top_k]]
    
    # ── Pre-computed routing (no live sub-backend retrieval) ───────────────────
    
    def _route_from_precomputed(
        self,
        queries: list[str],
        popularities: list[float],
        top_k: int,
    ) -> list[list[tuple[Document, float]]]:
        """Route and fuse using pre-computed sub-backend results.
        
        Runs neural inference via Modal to get routing decisions, then
        selects/fuses from cached sub-backend Document lists instead of
        calling live services.
        
        Args:
            queries: Query texts.
            popularities: Per-query popularity scores.
            top_k: Number of results per query.
        
        Returns:
            Scored result lists (one per query), same as live retrieval.
        """
        n = len(queries)
        logger.info(f"NeuralRouter (pre-computed) — predicting for {n} queries...")
        predictions = self._predict_batch(queries, popularities)
        
        if self.strict:
            dist = Counter(predictions)
            logger.info(f"NeuralRouter (pre-computed) — routing distribution: {dict(dist)}")

            # Detect backends predicted but missing from pre-computed cache
            predicted_set = set(predictions)
            missing = predicted_set - set(self.precomputed_results.keys())
            if missing:
                logger.warning(
                    "NeuralRouter (pre-computed) — predicted backends not in "
                    "pre-computed cache: %s (those queries get empty results)",
                    missing,
                )

            results: list[list[tuple[Document, float]]] = [[] for _ in range(n)]
            for i, backend_name in enumerate(predictions):
                cached = self.precomputed_results.get(backend_name, [])
                if i < len(cached):
                    results[i] = [(doc, 0.0) for doc in cached[i][:top_k]]
            return results
        
        else:
            # Hybrid: probability-weighted RRF from cached results
            prob_dicts = predictions
            results = []
            for i in tqdm(range(n), desc="NeuralRouter fusion (pre-computed)"):
                query_backend_results: dict[str, list[tuple[Document, float]]] = {}
                probs = prob_dicts[i]
                for backend_name in self.backend_order:
                    cached = self.precomputed_results.get(backend_name, [])
                    if i < len(cached):
                        query_backend_results[backend_name] = [
                            (doc, 0.0) for doc in cached[i]
                        ]
                fused = self._fuse_with_probabilities(
                    query_backend_results, probs, top_k
                )
                results.append(fused)
            return results
    
    # ── RagService Interface ──────────────────────────────────────────────────
    
    def retrieve_documents(
        self,
        query: str,
        *,
        top_k: int = 5,
        popularity: float,
        **kwargs: Any,
    ) -> list[Document]:
        """Route query and retrieve documents.
        
        Args:
            query: Query text
            top_k: Number of documents to return
            popularity: Popularity score (required)
            **kwargs: Passed through to backend services
        
        Returns:
            List of Document objects
        """
        scored = self.retrieve_documents_with_scores(
            query, top_k=top_k, popularity=popularity, **kwargs
        )
        return [doc for doc, _ in scored]
    
    def retrieve_documents_with_scores(
        self,
        query: str,
        *,
        top_k: int = 5,
        popularity: float,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Route query and retrieve documents with scores.
        
        Args:
            query: Query text
            top_k: Number of results to return
            popularity: Popularity score (required)
            **kwargs: Passed through to backend services
        
        Returns:
            List of (Document, score) tuples
        """
        prediction = self._predict_single(query, popularity)
        
        if self.strict:
            # Strict mode: route to single backend
            backend_name = prediction
            if backend_name not in self.backends:
                logger.warning(f"Backend '{backend_name}' not found, returning empty")
                return []
            
            service = self.backends[backend_name]
            return service.retrieve_documents_with_scores(query, top_k=top_k, **kwargs)
        
        else:
            # Hybrid mode: retrieve from all backends and fuse
            probabilities = prediction
            depth = max(self.rrf_depth, top_k)
            
            backend_results = {}
            for backend_name, service in self.backends.items():
                results = service.retrieve_documents_with_scores(
                    query, top_k=depth, **kwargs
                )
                backend_results[backend_name] = results
            
            return self._fuse_with_probabilities(backend_results, probabilities, top_k)
    
    def batch_retrieve(
        self,
        queries: list[str],
        *,
        top_k: int = 5,
        popularities: list[float],
        progress_bar: bool = True,
        **kwargs: Any,
    ) -> list[list[Document]]:
        """Batch retrieve documents.
        
        Args:
            queries: List of query texts
            top_k: Number of results per query
            popularities: List of popularity scores (required, same length as queries)
            progress_bar: Show progress bar
            **kwargs: Passed through to backend services
        
        Returns:
            List of document lists (one per query)
        """
        scored = self.batch_retrieve_with_scores(
            queries, top_k=top_k, popularities=popularities,
            progress_bar=progress_bar, **kwargs
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
        """Batch retrieve documents with scores.
        
        Args:
            queries: List of query texts
            top_k: Number of results per query
            popularities: List of popularity scores (required)
            progress_bar: Show progress bar
            **kwargs: Passed through to backend services
        
        Returns:
            List of scored result lists (one per query)
        """
        if len(queries) != len(popularities):
            raise ValueError(
                f"queries ({len(queries)}) and popularities ({len(popularities)}) "
                f"must have same length"
            )
        
        # ── Pre-computed path: route/fuse from cached sub-backend results ──────
        if self.precomputed_results is not None:
            return self._route_from_precomputed(
                queries, popularities, top_k
            )
        
        n = len(queries)
        
        # Step 1: Batch predict all queries
        logger.info(f"NeuralRouter — predicting for {n} queries...")
        predictions = self._predict_batch(queries, popularities)
        
        if self.strict:
            # Strict mode: group by backend and batch retrieve
            from collections import defaultdict
            
            backend_names = predictions
            
            # Log distribution
            dist = Counter(backend_names)
            logger.info(f"NeuralRouter — routing distribution: {dict(dist)}")
            
            # Group indices by backend
            groups: dict[str, list[int]] = defaultdict(list)
            for i, backend_name in enumerate(backend_names):
                groups[backend_name].append(i)
            
            # Initialize results
            results: list[list[tuple[Document, float]]] = [[] for _ in range(n)]
            
            # Retrieve per backend
            for backend_name, indices in tqdm(
                groups.items(),
                desc="NeuralRouter backends",
                disable=not progress_bar,
            ):
                if backend_name not in self.backends:
                    logger.warning(
                        f"Backend '{backend_name}' not found, "
                        f"skipping {len(indices)} queries"
                    )
                    continue
                
                service = self.backends[backend_name]
                batch_queries = [queries[i] for i in indices]
                
                logger.info(
                    f"NeuralRouter — {backend_name}: retrieving {len(batch_queries)} queries"
                )
                
                batch_results = service.batch_retrieve_with_scores(
                    batch_queries, top_k=top_k, **kwargs
                )
                
                for i, scored_docs in zip(indices, batch_results):
                    results[i] = scored_docs
            
            return results
        
        else:
            # Hybrid mode: retrieve from all backends for all queries, then fuse
            prob_dicts = predictions
            depth = max(self.rrf_depth, top_k)
            
            # Retrieve from each backend for all queries
            all_backend_results: dict[str, list[list[tuple[Document, float]]]] = {}
            
            for backend_name, service in tqdm(
                self.backends.items(),
                desc="NeuralRouter backends (hybrid)",
                disable=not progress_bar,
            ):
                logger.info(
                    f"NeuralRouter — {backend_name}: retrieving {n} queries @ depth={depth}"
                )
                backend_results = service.batch_retrieve_with_scores(
                    queries, top_k=depth, **kwargs
                )
                all_backend_results[backend_name] = backend_results
            
            # Fuse per-query
            results = []
            for i in tqdm(
                range(n),
                desc="NeuralRouter fusion",
                disable=not progress_bar,
            ):
                # Build backend_results dict for this query
                query_backend_results = {
                    backend_name: all_backend_results[backend_name][i]
                    for backend_name in self.backends.keys()
                }
                probabilities = prob_dicts[i]
                
                fused = self._fuse_with_probabilities(
                    query_backend_results, probabilities, top_k
                )
                results.append(fused)
            
            return results
    
    # ── RagService ABC Stubs ──────────────────────────────────────────────────
    
    def load_index(self, path_or_name: str | Path, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "NeuralRouterRagService does not support load_index(). "
            "Load each backend individually before passing to NeuralRouterRagService."
        )
    
    def index_from_dataframe(self, df, text_field, **kwargs) -> IndexResult:
        raise NotImplementedError(
            "NeuralRouterRagService does not support indexing. "
            "Index each backend separately."
        )
    
    def index_from_parquet(self, parquet_path, **kwargs) -> IndexResult:
        raise NotImplementedError(
            "NeuralRouterRagService does not support indexing. "
            "Index each backend separately."
        )
    
    def get_doc_count(self) -> int:
        if self.precomputed_results is not None or not self.backends:
            return 1  # non-zero so runner's empty-index check passes
        first_backend = list(self.backends.values())[0]
        return first_backend.get_doc_count()
    
    def get_index_stats(self) -> dict[str, Any]:
        return {
            name: service.get_index_stats()
            for name, service in self.backends.items()
        }
