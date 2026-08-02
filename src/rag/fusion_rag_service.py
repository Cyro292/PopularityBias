"""FusionRagService — learned backend transformations with fixed RRF.

This service does not learn RRF weights. For each query it predicts backend-
specific transformed position scores, reranks each backend independently using
those scores, and then applies standard unweighted RRF on the new ranks.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from langchain.schema import Document
from tqdm import tqdm

from .base import IndexResult, RagService

logger = logging.getLogger(__name__)


class FusionRagService(RagService):
    """Query-conditioned backend transformation service for fixed-RRF fusion."""

    def __init__(
        self,
        *,
        backends: dict[str, RagService] | None = None,
        backend_order: list[str],
        model_path: str | Path,
        rrf_k: int = 60,
        rrf_depth: int = 60,
        predict_batch_size: int = 32,
    ) -> None:
        self.backends = backends or {}
        self.backend_order = backend_order
        self.model_path = Path(model_path)
        self.rrf_k = rrf_k
        self.rrf_depth = rrf_depth
        self.predict_batch_size = predict_batch_size
        self.precomputed_results: dict[str, list[list[Document]]] | None = None

        if self.backends:
            for name in backend_order:
                if name not in self.backends:
                    raise ValueError(f"Backend '{name}' in backend_order not found in backends dict")

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        self._model_dict = torch.load(self.model_path, map_location="cpu")

        from src.router.fusion_modal_service import FusionModalService

        self._service = FusionModalService()

    def set_precomputed_results(self, results: dict[str, list[list[Document]]]) -> None:
        self.precomputed_results = results

    def _predict_transforms(
        self,
        queries: list[str],
        popularities: list[float],
    ) -> list[list[list[float]]]:
        return self._service.predict(
            questions=queries,
            popularity=popularities,
            predictor_state=self._model_dict["predictor_state"],
            scaler_mean=self._model_dict["scaler_mean"],
            scaler_scale=self._model_dict["scaler_scale"],
            model_config=self._model_dict["model_config"],
            batch_size=self.predict_batch_size,
        )

    def _predict_single(self, query: str, popularity: float) -> list[list[float]]:
        return self._predict_transforms([query], [popularity])[0]

    @staticmethod
    def _doc_key(doc: Document) -> str:
        wikipedia_id = doc.metadata.get("wikipedia_id") if hasattr(doc, "metadata") else None
        return str(wikipedia_id) if wikipedia_id is not None else doc.page_content

    def _rerank_backend_results(
        self,
        results: list[tuple[Document, float]],
        transformed_position_scores: list[float],
    ) -> list[tuple[Document, float]]:
        rescored: list[tuple[float, int, Document]] = []
        for original_pos, (doc, _) in enumerate(results):
            if original_pos < len(transformed_position_scores):
                transformed_score = transformed_position_scores[original_pos]
            else:
                transformed_score = -1e6 - original_pos
            rescored.append((transformed_score, -original_pos, doc))

        rescored.sort(reverse=True)
        return [(doc, score) for score, _, doc in rescored]

    def _fuse_rrf(
        self,
        backend_results: dict[str, list[tuple[Document, float]]],
        top_k: int,
    ) -> list[tuple[Document, float]]:
        scores: dict[str, float] = {}
        docs: dict[str, Document] = {}
        for backend_name in self.backend_order:
            results = backend_results.get(backend_name, [])
            for rank, (doc, _) in enumerate(results, start=1):
                key = self._doc_key(doc)
                scores[key] = scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank)
                docs[key] = doc
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(docs[key], score) for key, score in ranked[:top_k]]

    def _route_from_precomputed(
        self,
        queries: list[str],
        popularities: list[float],
        top_k: int,
    ) -> list[list[tuple[Document, float]]]:
        transforms = self._predict_transforms(queries, popularities)
        results: list[list[tuple[Document, float]]] = []
        for i in tqdm(range(len(queries)), desc="FusionRagService fusion (pre-computed)"):
            backend_results: dict[str, list[tuple[Document, float]]] = {}
            for backend_idx, backend_name in enumerate(self.backend_order):
                cached = self.precomputed_results.get(backend_name, [])
                if i < len(cached):
                    backend_results[backend_name] = self._rerank_backend_results(
                        [(doc, 0.0) for doc in cached[i]],
                        transforms[i][backend_idx],
                    )
            results.append(self._fuse_rrf(backend_results, top_k))
        return results

    def retrieve_documents(
        self,
        query: str,
        *,
        top_k: int = 5,
        popularity: float = 0.0,
        **kwargs: Any,
    ) -> list[Document]:
        return [doc for doc, _ in self.retrieve_documents_with_scores(query, top_k=top_k, popularity=popularity, **kwargs)]

    def retrieve_documents_with_scores(
        self,
        query: str,
        *,
        top_k: int = 5,
        popularity: float = 0.0,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        transforms = self._predict_single(query, popularity)
        depth = max(self.rrf_depth, top_k)
        backend_results: dict[str, list[tuple[Document, float]]] = {}
        for backend_idx, backend_name in enumerate(self.backend_order):
            service = self.backends[backend_name]
            raw = service.retrieve_documents_with_scores(query, top_k=depth, **kwargs)
            backend_results[backend_name] = self._rerank_backend_results(raw, transforms[backend_idx])
        return self._fuse_rrf(backend_results, top_k)

    def batch_retrieve(
        self,
        queries: list[str],
        *,
        top_k: int = 5,
        popularities: list[float] | None = None,
        progress_bar: bool = True,
        **kwargs: Any,
    ) -> list[list[Document]]:
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
        popularities: list[float] | None = None,
        progress_bar: bool = True,
        **kwargs: Any,
    ) -> list[list[tuple[Document, float]]]:
        popularities = popularities or [0.0] * len(queries)
        if self.precomputed_results is not None:
            return self._route_from_precomputed(queries, popularities, top_k)

        transforms = self._predict_transforms(queries, popularities)
        depth = max(self.rrf_depth, top_k)

        raw_backend_results: dict[str, list[list[tuple[Document, float]]]] = {}
        for backend_name, service in self.backends.items():
            raw_backend_results[backend_name] = service.batch_retrieve_with_scores(
                queries,
                top_k=depth,
                progress_bar=progress_bar,
                **kwargs,
            )

        results: list[list[tuple[Document, float]]] = []
        for i in range(len(queries)):
            backend_results: dict[str, list[tuple[Document, float]]] = {}
            for backend_idx, backend_name in enumerate(self.backend_order):
                backend_results[backend_name] = self._rerank_backend_results(
                    raw_backend_results[backend_name][i],
                    transforms[i][backend_idx],
                )
            results.append(self._fuse_rrf(backend_results, top_k))
        return results

    def index_from_dataframe(self, *args: Any, **kwargs: Any) -> IndexResult:
        raise NotImplementedError("FusionRagService does not support indexing")

    def index_from_parquet(self, *args: Any, **kwargs: Any) -> IndexResult:
        raise NotImplementedError("FusionRagService does not support indexing")

    def load_index(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("FusionRagService does not support loading indexes")

    def get_doc_count(self) -> int:
        return -1
