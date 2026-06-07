"""NeuralRouterRagService — BERT-based neural router with strict/hybrid modes.

Uses a trained BERT-based classifier to route queries between retrieval backends.
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
from pathlib import Path
from typing import Any

import torch
import numpy as np
from langchain.schema import Document
from tqdm import tqdm

from .base import IndexResult, RagService

logger = logging.getLogger(__name__)


class NeuralRouterRagService(RagService):
    """BERT-based neural router with strict (argmax) or hybrid (probability-weighted RRF) modes.
    
    Routes queries between multiple retrieval backends using a trained BERT classifier.
    The classifier takes question text + normalized popularity as input and outputs
    probabilities for each backend.
    
    Args:
        backends: Dict mapping backend names to loaded RagService instances.
            Example: {"bm25_plus": bm25_svc, "ivfpq_high": faiss_svc}
        backend_order: List of backend names in the order the model was trained.
            Model output index i corresponds to backend_order[i].
            Example: ["bm25_plus", "ivfpq_high"]
        model_path: Path to trained model .pt file (from train_router.py).
        strict: If True, use argmax (select single backend). If False, use
            probability-weighted RRF (retrieve from all backends and fuse).
        rrf_k: RRF smoothing constant (only for hybrid mode). Default 60.
        rrf_depth: Number of candidates to fetch per backend before fusion
            (only for hybrid mode). Default 60.
        predict_batch_size: Batch size for BERT inference. Default 32.
        device: Torch device for inference. Default "cpu".
    """
    
    def __init__(
        self,
        *,
        backends: dict[str, RagService],
        backend_order: list[str],
        model_path: str | Path,
        strict: bool = True,
        rrf_k: int = 60,
        rrf_depth: int = 60,
        predict_batch_size: int = 32,
        device: str = "cpu",
    ) -> None:
        self.backends = backends
        self.backend_order = backend_order
        self.model_path = Path(model_path)
        self.strict = strict
        self.rrf_k = rrf_k
        self.rrf_depth = rrf_depth
        self.predict_batch_size = predict_batch_size
        self.device = device
        
        # Validate backend_order matches backends
        for name in backend_order:
            if name not in backends:
                raise ValueError(f"Backend '{name}' in backend_order not found in backends dict")
        
        # Load model
        logger.info(f"NeuralRouterRagService — loading model from {model_path}...")
        self._load_model()
        
        self.include_popularity = self.model_config.get('include_popularity', True)
        
        mode_str = "strict (argmax)" if strict else f"hybrid (probability-weighted RRF, k={rrf_k})"
        pop_str = "with popularity" if self.include_popularity else "BERT-only (no popularity)"
        logger.info(
            f"NeuralRouterRagService ready — mode={mode_str}, "
            f"backends={backend_order}, {pop_str}, device={device}"
        )
    
    # ── Model Loading ─────────────────────────────────────────────────────────
    
    def _load_model(self) -> None:
        """Load trained router model and extract configuration."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Load model dict
        model_dict = torch.load(self.model_path, map_location=self.device)
        
        # Extract components
        self.classifier_state = model_dict['classifier_state']
        self.scaler_mean = model_dict['scaler_mean']
        self.scaler_scale = model_dict['scaler_scale']
        self.model_config = model_dict['model_config']
        
        # Initialize BERT tokenizer and model
        from transformers import AutoTokenizer, AutoModel
        
        logger.info("Loading BERT tokenizer and model...")
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.bert = AutoModel.from_pretrained('bert-base-uncased').to(self.device)
        self.bert.eval()
        
        # Freeze BERT (inference only)
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # Initialize classifier network
        self.classifier = self._build_classifier()
        self.classifier.to(self.device)
        self.classifier.eval()
        
        logger.info(
            f"Model loaded: {self.model_config['num_classes']} classes, "
            f"{self.model_config['input_dim']} input dims"
        )
    
    def _build_classifier(self) -> torch.nn.Module:
        """Build and load classifier network from saved state."""
        import torch.nn as nn
        
        cfg = self.model_config
        network = nn.Sequential(
            nn.Linear(cfg['input_dim'], cfg['hidden_dim1']),
            nn.ReLU(),
            nn.Dropout(cfg['dropout']),
            nn.Linear(cfg['hidden_dim1'], cfg['hidden_dim2']),
            nn.ReLU(),
            nn.Dropout(cfg['dropout']),
            nn.Linear(cfg['hidden_dim2'], cfg['num_classes'])
        )
        
        # Load weights
        network.load_state_dict(self.classifier_state)
        return network
    
    # ── Prediction ────────────────────────────────────────────────────────────
    
    def _normalize_popularity(self, popularity: float | list[float]) -> torch.Tensor:
        """Normalize popularity score(s) using saved scaler parameters.
        
        Args:
            popularity: Single value or list of values
        
        Returns:
            Normalized tensor of shape (1, 1) or (N, 1)
        """
        if isinstance(popularity, (int, float)):
            popularity = [popularity]
        
        normalized = [
            (p - self.scaler_mean[0]) / self.scaler_scale[0]
            for p in popularity
        ]
        return torch.tensor([[n] for n in normalized], dtype=torch.float32, device=self.device)
    
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
        with torch.no_grad():
            # Tokenize
            tokens = self.tokenizer(
                [query],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            
            # Get BERT embedding
            outputs = self.bert(**tokens)
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # (1, 768)
            
            # Combine features
            if self.include_popularity:
                pop_normalized = self._normalize_popularity(popularity)  # (1, 1)
                combined = torch.cat([cls_embedding, pop_normalized], dim=1)  # (1, 769)
            else:
                combined = cls_embedding
            
            # Forward pass
            logits = self.classifier(combined)  # (1, num_classes)
            
            if self.strict:
                # Argmax mode
                pred_idx = int(torch.argmax(logits, dim=1).item())
                backend_name = self.backend_order[pred_idx]
                logger.debug(f"Router → {backend_name} (popularity={popularity:.1f})")
                return backend_name
            else:
                # Softmax probabilities
                probs = torch.softmax(logits, dim=1)[0]  # (num_classes,)
                prob_dict = {
                    self.backend_order[i]: float(probs[i].item())
                    for i in range(len(self.backend_order))
                }
                logger.debug(f"Router probs: {prob_dict} (popularity={popularity:.1f})")
                return prob_dict
    
    def _predict_batch(
        self,
        queries: list[str],
        popularities: list[float],
    ) -> list[str | dict[str, float]]:
        """Predict backends for a batch of queries.
        
        Args:
            queries: List of query texts
            popularities: List of popularity scores (same length)
        
        Returns:
            If strict: list of backend names
            If hybrid: list of dicts {backend_name: probability}
        """
        if len(queries) != len(popularities):
            raise ValueError(
                f"queries ({len(queries)}) and popularities ({len(popularities)}) "
                f"must have same length"
            )
        
        results = []
        
        # Process in batches for memory efficiency
        for i in range(0, len(queries), self.predict_batch_size):
            batch_queries = queries[i:i + self.predict_batch_size]
            batch_pops = popularities[i:i + self.predict_batch_size]
            
            with torch.no_grad():
                # Tokenize batch
                tokens = self.tokenizer(
                    batch_queries,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors='pt'
                )
                tokens = {k: v.to(self.device) for k, v in tokens.items()}
                
                # Get BERT embeddings
                outputs = self.bert(**tokens)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]  # (B, 768)
                
                # Combine features
                if self.include_popularity:
                    pop_normalized = self._normalize_popularity(batch_pops)  # (B, 1)
                    combined = torch.cat([cls_embeddings, pop_normalized], dim=1)  # (B, 769)
                else:
                    combined = cls_embeddings
                
                # Forward pass
                logits = self.classifier(combined)  # (B, num_classes)
                
                if self.strict:
                    # Argmax mode
                    pred_indices = torch.argmax(logits, dim=1).tolist()
                    batch_results = [self.backend_order[idx] for idx in pred_indices]
                else:
                    # Softmax probabilities
                    probs = torch.softmax(logits, dim=1)  # (B, num_classes)
                    batch_results = [
                        {
                            self.backend_order[j]: float(probs[i, j].item())
                            for j in range(len(self.backend_order))
                        }
                        for i in range(len(batch_queries))
                    ]
                
                results.extend(batch_results)
        
        return results
    
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
        
        n = len(queries)
        
        # Step 1: Batch predict all queries
        logger.info(f"NeuralRouter — predicting for {n} queries...")
        predictions = self._predict_batch(queries, popularities)
        
        if self.strict:
            # Strict mode: group by backend and batch retrieve
            from collections import defaultdict, Counter
            
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
        # Return doc count from first backend
        first_backend = list(self.backends.values())[0]
        return first_backend.get_doc_count()
    
    def get_index_stats(self) -> dict[str, Any]:
        return {
            name: service.get_index_stats()
            for name, service in self.backends.items()
        }
