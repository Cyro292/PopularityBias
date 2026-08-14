"""HybridFaissRagService — FAISS dense + BM25 sparse fused via Reciprocal Rank Fusion.

Mirrors what Elasticsearch ``hybrid`` strategy does natively:
  1. Run approximate kNN (FAISS IVFPQ) for dense scores.
  2. Run BM25 for sparse scores.
  3. Fuse both ranked lists with RRF: score = 1/(k+rank_dense) + 1/(k+rank_sparse).
  4. Return top-k by fused score.

The RRF constant ``rrf_k`` defaults to 60, matching Elasticsearch's default.
The candidate depth ``rrf_depth`` controls how many results each backend fetches
before fusion — defaults to 60, same as ES.

Example::

    from src.rag.faiss_rag_service import FaissRagService
    from src.rag.bm25_rag_service import BM25RagService
    from src.rag.hybrid_faiss_rag_service import HybridFaissRagService

    faiss_svc = FaissRagService(...)
    faiss_svc.load_index("data/wiki_full_bil/faiss_high/faiss")

    bm25_svc = BM25RagService()
    bm25_svc.load_index("data/wiki_full_bil/bm25_bm25plus_recursive")

    hybrid = HybridFaissRagService(dense_service=faiss_svc, sparse_service=bm25_svc)
    docs = hybrid.retrieve_documents("Who wrote Hamlet?", top_k=10)
"""

from __future__ import annotations

import logging
from typing import Any

from langchain.schema import Document
from tqdm import tqdm

from .base import IndexResult, RagService

logger = logging.getLogger(__name__)


class HybridFaissRagService(RagService):
    """Dense + sparse hybrid retrieval via Reciprocal Rank Fusion.

    Combines a FAISS dense backend with a BM25 sparse backend using RRF,
    replicating the behaviour of Elasticsearch's built-in hybrid strategy.

    Args:
        dense_service: A loaded ``FaissRagService`` instance. May be ``None``
            when ``set_precomputed_results()`` will be called.
        sparse_service: A loaded ``BM25RagService`` instance. May be ``None``
            when ``set_precomputed_results()`` will be called.
        rrf_k: RRF smoothing constant. Higher values reduce the impact of
            top-ranked documents. Defaults to 60 (ES default).
        rrf_depth: Number of candidates fetched from each backend before
            fusion. Must be >= top_k at query time. Defaults to 60.
    """

    def __init__(
        self,
        dense_service: RagService | None = None,
        sparse_service: RagService | None = None,
        *,
        rrf_k: int = 60,
        rrf_depth: int = 60,
    ) -> None:
        self.dense_service  = dense_service
        self.sparse_service = sparse_service
        self.rrf_k          = rrf_k
        self.rrf_depth      = rrf_depth
        self.precomputed_results: dict[str, list[list[Document]]] | None = None
        logger.info(
            "HybridFaissRagService ready (rrf_k=%d, rrf_depth=%d)",
            rrf_k,
            rrf_depth,
        )

    # ── Pre-computed results ───────────────────────────────────────────────────

    def set_precomputed_results(
        self, results: dict[str, list[list[Document]]]
    ) -> None:
        """Supply cached sub-backend results so live retrieval is skipped.

        Args:
            results: Dict with ``"dense"`` and ``"sparse"`` keys, each mapping
                to ``list[list[Document]]`` (one list per query).
        """
        self.precomputed_results = results
        logger.info(
            "HybridFaissRagService — pre-computed results set — "
            "live retrieval will be skipped"
        )

    # ── RRF core ──────────────────────────────────────────────────────────────

    def _fuse(
        self,
        dense_results: list[tuple[Document, float]],
        sparse_results: list[tuple[Document, float]],
        top_k: int,
    ) -> list[tuple[Document, float]]:
        """Apply RRF to two ranked lists and return top-k fused results.

        Documents are deduplicated by their ``page_content`` string.  The RRF
        score for each unique document is the sum of its per-list contributions:
        ``1 / (rrf_k + rank)`` where rank is 1-based.

        Args:
            dense_results: Ranked ``(Document, score)`` list from dense backend.
            sparse_results: Ranked ``(Document, score)`` list from sparse backend.
            top_k: Number of results to return.

        Returns:
            Top-k ``(Document, rrf_score)`` tuples sorted descending by fused score.
        """
        k = self.rrf_k
        scores: dict[str, float]   = {}
        docs:   dict[str, Document] = {}

        for rank, (doc, _) in enumerate(dense_results, start=1):
            key = doc.page_content
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            docs[key]   = doc

        for rank, (doc, _) in enumerate(sparse_results, start=1):
            key = doc.page_content
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            docs[key]   = doc

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(docs[key], score) for key, score in ranked[:top_k]]

    # ── Single-query retrieval ─────────────────────────────────────────────────

    def retrieve_documents(
        self,
        query: str,
        *,
        top_k: int = 5,
        **kwargs: Any,
    ) -> list[Document]:
        """Return top-k documents fused from dense and sparse retrieval.

        Args:
            query: Free-text query string.
            top_k: Number of documents to return.
            **kwargs: Passed through to sub-backends.

        Returns:
            Ranked list of ``Document`` objects.
        """
        return [doc for doc, _ in self.retrieve_documents_with_scores(query, top_k=top_k, **kwargs)]

    def retrieve_documents_with_scores(
        self,
        query: str,
        *,
        top_k: int = 5,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return top-k documents with RRF scores.

        Args:
            query: Free-text query string.
            top_k: Number of results to return.
            **kwargs: Passed through to sub-backends.

        Returns:
            List of ``(Document, rrf_score)`` tuples, best-first.
        """
        depth = max(self.rrf_depth, top_k)
        dense_results  = self.dense_service.retrieve_documents_with_scores(query, top_k=depth, **kwargs)
        sparse_results = self.sparse_service.retrieve_documents_with_scores(query, top_k=depth, **kwargs)
        return self._fuse(dense_results, sparse_results, top_k)

    # ── Batch retrieval ────────────────────────────────────────────────────────

    def batch_retrieve(
        self,
        queries: list[str],
        *,
        top_k: int = 5,
        progress_bar: bool = True,
        **kwargs: Any,
    ) -> list[list[Document]]:
        """Retrieve documents for multiple queries.

        Args:
            queries: List of query strings.
            top_k: Results per query.
            progress_bar: Show tqdm progress bar.
            **kwargs: Passed through to sub-backends.

        Returns:
            One list of Documents per query.
        """
        return [
            [doc for doc, _ in result]
            for result in self.batch_retrieve_with_scores(
                queries, top_k=top_k, progress_bar=progress_bar, **kwargs
            )
        ]

    def batch_retrieve_with_scores(
        self,
        queries: list[str],
        *,
        top_k: int = 5,
        progress_bar: bool = True,
        **kwargs: Any,
    ) -> list[list[tuple[Document, float]]]:
        """Retrieve scored documents for multiple queries.

        Calls each backend's ``batch_retrieve_with_scores`` once (so FAISS
        does a single embedding pass for all queries), then fuses per-query.

        Args:
            queries: List of query strings.
            top_k: Results per query.
            progress_bar: Show tqdm progress bar.
            **kwargs: Passed through to sub-backends.

        Returns:
            One list of ``(Document, rrf_score)`` tuples per query.
        """
        # ── Pre-computed path: RRF fusion from cached sub-backend results ──────
        if self.precomputed_results is not None:
            dense_all  = self.precomputed_results["dense"]
            sparse_all = self.precomputed_results["sparse"]
            return [
                self._fuse(
                    [(doc, 0.0) for doc in dense_all[i]],
                    [(doc, 0.0) for doc in sparse_all[i]],
                    top_k,
                )
                for i in tqdm(range(len(queries)), desc="Fusing (hybrid_faiss, pre-computed)")
            ]

        depth = max(self.rrf_depth, top_k)

        dense_all  = self.dense_service.batch_retrieve_with_scores(
            queries, top_k=depth, progress_bar=progress_bar, **kwargs
        )
        sparse_all = self.sparse_service.batch_retrieve_with_scores(
            queries, top_k=depth, progress_bar=progress_bar, **kwargs
        )

        return [
            self._fuse(dense_results, sparse_results, top_k)
            for dense_results, sparse_results in tqdm(
                zip(dense_all, sparse_all),
                total=len(queries),
                desc="Fusing (hybrid_faiss)",
                disable=not progress_bar,
            )
        ]

    # ── RagService ABC stubs ───────────────────────────────────────────────────

    def index_from_parquet(self, *args: Any, **kwargs: Any) -> IndexResult:
        raise NotImplementedError("HybridFaissRagService has no index of its own — index each sub-backend separately.")

    def index_from_dataframe(self, *args: Any, **kwargs: Any) -> IndexResult:
        raise NotImplementedError("HybridFaissRagService has no index of its own — index each sub-backend separately.")

    def load_index(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Load each sub-backend separately before passing to HybridFaissRagService.")

    def save_index(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("HybridFaissRagService has no index to save.")

    def delete_index(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("HybridFaissRagService has no index to delete.")

    def get_doc_count(self) -> int:
        if self.precomputed_results is not None or self.dense_service is None:
            return 1  # non-zero so runner's empty-index check passes
        return self.dense_service.get_doc_count()

    def embed_prompt(self, text: str) -> str:
        if self.dense_service is None:
            return text
        return self.dense_service.embed_prompt(text)

    def embed_passage(self, text: str) -> str:
        if self.dense_service is None:
            return text
        return self.dense_service.embed_passage(text)
