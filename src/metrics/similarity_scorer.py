"""Retrieve BM25 and dense scores for target Wikipedia articles.

The services may use local indices or Elasticsearch. One service is supplied
per retrieval strategy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from langchain.schema import Document

from src.rag.base import RagService

logger = logging.getLogger(__name__)


@dataclass
class SimilarityResult:
    """Similarity scores between a question and a target article.

    Attributes:
        bm25_score: BM25 retrieval score (0.0 if article is not in the top-k).
        cosine_score: Dense retrieval score (0.0 if article is not in the top-k).
    """

    bm25_score: float
    cosine_score: float


class SimilarityScorer:
    """Compute BM25 and dense similarity through two RAG services.

    BM25 scores come from the sparse retrieval service; cosine scores come
    from the dense vector service. The services may use separate local indices.
    For each (question, target_wiki_id) pair the score of the target article
    is extracted from the top-k results. If the target does not appear,
    the score is 0.0.

    Args:
        bm25_service: RAG service configured for BM25 retrieval.
        dense_service: RAG service configured for dense retrieval.
        index_name: Dense-index path or name to query.
        bm25_index_name: Optional BM25 index path or name. Defaults to
            ``index_name`` when both retrieval services share an index.
        top_k: How many results to fetch per query (larger = more likely to
            find the target article, but slower).
        id_metadata_key: Metadata field holding the Wikipedia page ID.

    Example::

        from src.rag.elasticsearch_rag_service import ElasticsearchRagService
        from src.metrics.similarity_scorer import SimilarityScorer

        bm25_svc  = ElasticsearchRagService(strategy="bm25",  es_url=..., ...)
        dense_svc = ElasticsearchRagService(strategy="approximation",
                                            config=cfg, es_url=..., ...)
        scorer = SimilarityScorer(bm25_svc, dense_svc, index_name="wiki_full_l")
        result = scorer.score("Who is Reza Pahlavi?", target_wiki_id=613947)
        print(result.bm25_score, result.cosine_score)
    """

    def __init__(
        self,
        bm25_service: RagService,
        dense_service: RagService,
        index_name: str | Path,
        *,
        bm25_index_name: str | Path | None = None,
        top_k: int = 10,
        id_metadata_key: str = "wikipedia_id",
    ) -> None:
        self.bm25_service = bm25_service
        self.dense_service = dense_service
        self.index_name = index_name
        self.top_k = top_k
        self.id_metadata_key = id_metadata_key

        if hasattr(bm25_service, "load_index"):
            bm25_service.load_index(bm25_index_name or index_name)
        if hasattr(dense_service, "load_index"):
            dense_service.load_index(index_name)
        logger.info(
            "SimilarityScorer ready with BM25 index '%s' and dense index '%s'",
            bm25_index_name or index_name,
            index_name,
        )

    # ── Internal helpers ───────────────────────────────────────────────────

    def _extract_score(
        self,
        results: list[tuple[Document, float]],
        target_id: int,
    ) -> float:
        """Pull the score for the target article out of a result list."""
        for doc, score in results:
            raw = doc.metadata.get(self.id_metadata_key)
            try:
                if int(raw) == target_id:
                    return float(score)
            except (TypeError, ValueError):
                continue
        return 0.0

    # ── Public API ─────────────────────────────────────────────────────────

    def score(self, question: str, target_wiki_id: int) -> SimilarityResult:
        """Score a single question against a target Wikipedia article.

        Args:
            question: Query string.
            target_wiki_id: Wikipedia page ID of the target article.

        Returns:
            :class:`SimilarityResult` with bm25_score and cosine_score.
        """
        results = self.score_batch([(question, target_wiki_id)])
        return results[0]

    def score_batch(
        self,
        pairs: list[tuple[str, int]],
    ) -> list[SimilarityResult]:
        """Score multiple (question, target_wiki_id) pairs.

        Both BM25 and dense queries are sent as batches to minimise retrieval
        overhead.

        Args:
            pairs: List of (question, target_wiki_id) tuples.

        Returns:
            List of :class:`SimilarityResult` in the same order as ``pairs``.
        """
        if not pairs:
            return []

        questions = [q for q, _ in pairs]
        ids = [wid for _, wid in pairs]

        logger.info(f"Scoring {len(pairs)} pair(s) — BM25 batch...")
        bm25_raw = self.bm25_service.batch_retrieve_with_scores(
            questions, top_k=self.top_k, progress_bar=True
        )

        logger.info(f"Scoring {len(pairs)} pair(s) — dense batch...")
        dense_raw = self.dense_service.batch_retrieve_with_scores(
            questions, top_k=self.top_k, progress_bar=True
        )

        results: list[SimilarityResult] = []
        for bm25_hits, dense_hits, target_id in zip(bm25_raw, dense_raw, ids):
            results.append(SimilarityResult(
                bm25_score=self._extract_score(bm25_hits, target_id),
                cosine_score=self._extract_score(dense_hits, target_id),
            ))
            # free hit lists immediately
            del bm25_hits, dense_hits

        return results
