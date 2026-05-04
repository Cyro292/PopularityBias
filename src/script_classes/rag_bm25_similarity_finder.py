"""BM25 similarity finder — retrieve nearest Wikipedia document IDs for query texts."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from langchain.schema import Document

from src.rag.base import RagService  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass
class BM25Match:
    """A single BM25 retrieval match.

    Attributes:
        wikipedia_id: Wikipedia page ID from document metadata.
        title: Wikipedia page title from document metadata.
        score: BM25 similarity score.
        metadata: Full metadata dict from the matched document.
    """

    wikipedia_id: int | None
    title: str | None
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class RagBM25SimilarityFinder:
    """Find the nearest Wikipedia documents for a list of query texts using BM25.

    Wraps any :class:`~src.rag.base.RagService` and returns lightweight
    :class:`BM25Match` results. Documents are never fully loaded into memory
    all at once — results are parsed and discarded per-query so RAM usage
    stays proportional to ``top_k``, not corpus size.

    Args:
        rag_service: Initialised RagService (must support BM25 retrieval).
        index_name: Name of the index to query.
        id_metadata_key: Metadata key holding the Wikipedia page ID.
        title_metadata_key: Metadata key holding the page title.

    Example::

        from src.rag.elasticsearch_rag_service import ElasticsearchRagService
        from src.similarity.rag_bm25_similarity_finder import RagBM25SimilarityFinder

        service = ElasticsearchRagService(strategy="bm25", es_url=..., ...)
        finder  = RagBM25SimilarityFinder(service, index_name="wiki_full_l")
        results = finder.find(["Reza Pahlavi"], top_k=5)
        for match in results["Reza Pahlavi"]:
            print(match.wikipedia_id, match.title, match.score)
    """

    def __init__(
        self,
        rag_service: RagService,
        index_name: str,
        *,
        id_metadata_key: str = "wikipedia_id",
        title_metadata_key: str = "wikipedia_title",
    ) -> None:
        self.rag_service = rag_service
        self.index_name = index_name
        self.id_metadata_key = id_metadata_key
        self.title_metadata_key = title_metadata_key

        if hasattr(rag_service, "load_index"):
            rag_service.load_index(index_name)
            logger.info(f"Loaded index '{index_name}'")

    def _parse_match(self, doc: Document, score: float) -> BM25Match:
        """Convert a LangChain Document + score to a BM25Match and drop the doc."""
        raw_id = doc.metadata.get(self.id_metadata_key)
        try:
            wikipedia_id = int(raw_id) if raw_id is not None else None
        except (ValueError, TypeError):
            wikipedia_id = None

        match = BM25Match(
            wikipedia_id=wikipedia_id,
            title=doc.metadata.get(self.title_metadata_key),
            score=score,
            metadata=doc.metadata,
        )
        # Explicitly drop the document content to free RAM immediately
        del doc
        return match

    def find(
        self,
        texts: list[str],
        *,
        top_k: int = 5,
        **retrieve_kwargs: Any,
    ) -> dict[str, list[BM25Match]]:
        """Find the top-k BM25 matches for each query text.

        Uses ``batch_retrieve_with_scores`` when available (ES service) for
        efficiency, otherwise falls back to one call per text. Either way,
        document page content is parsed and immediately discarded — only the
        small ``BM25Match`` structs are kept in memory.

        Args:
            texts: Query strings to search for.
            top_k: Number of nearest documents to return per query.
            **retrieve_kwargs: Extra kwargs forwarded to the underlying
                retrieval method (e.g. ``strategy``, ``num_candidates``).

        Returns:
            Dict mapping each input text to its ranked list of
            :class:`BM25Match` results.

        Raises:
            ValueError: If ``texts`` is empty.
        """
        if not texts:
            raise ValueError("texts must not be empty")

        logger.info(f"BM25 search: {len(texts)} query/queries, top_k={top_k}")

        # ── Batch path (ES service) ────────────────────────────────────────
        if hasattr(self.rag_service, "batch_retrieve_with_scores"):
            raw: list[list[tuple[Document, float]]] = (
                self.rag_service.batch_retrieve_with_scores(
                    texts, top_k=top_k, **retrieve_kwargs
                )
            )
            results: dict[str, list[BM25Match]] = {}
            for text, pairs in zip(texts, raw):
                results[text] = [self._parse_match(doc, score) for doc, score in pairs]
                del pairs  # free doc objects for this query immediately
            del raw
            return results

        # ── Fallback: one call per text ───────────────────────────────────
        results = {}
        for text in texts:
            docs: list[Document] = self.rag_service.retrieve_documents(
                text, top_k=top_k, **retrieve_kwargs
            )
            results[text] = [self._parse_match(doc, 0.0) for doc in docs]
            del docs
        return results
