"""Retrieval runner — Stage 1 of the RAG evaluation pipeline.

Backends are declared as :class:`RetrievalBackend` entries — one per index you
want to evaluate.  The runner loads each backend, retrieves documents for every
question, and writes ``retrieved_docs_<key>.csv`` checkpoints consumed by the
next stage (:mod:`src.process.pipeline.generating_runner`).

Declaring backends
------------------
Edit the ``backends`` list in :class:`RetrievalConfig`::

    RetrievalBackend(
        key         = "es_approx",
        label       = "Dense Retrieval (ES approximation)",
        type        = "elasticsearch",
        es_strategy = "approximation",
    ),
    RetrievalBackend(
        key         = "bm25",
        label       = "Sparse Retrieval (BM25)",
        type        = "elasticsearch",
        es_strategy = "bm25",
    ),
    RetrievalBackend(
        key        = "faiss_high",
        label      = "Dense Retrieval (FAISS high-pop ivfpq)",
        type       = "faiss",
        index_path = DATA_DIR / "wiki_full_bil" / "faiss_high",
    ),
    RetrievalBackend(
        key   = "zero_shot",
        label = "Zero-shot (no retrieval)",
        type  = "zero_shot",
    ),

Checkpoint behaviour
--------------------
- ``retrieved_docs_<key>.csv`` is reused if it exists.
- Pass ``--restart`` to overwrite all checkpoints.
- Pass ``--restart-keys key1 key2`` to overwrite only specific backends.

Usage
-----
::

    python -m src.process.pipeline.retrieval_runner
    python -m src.process.pipeline.retrieval_runner --restart
    python -m src.process.pipeline.retrieval_runner --restart-keys faiss_high faiss_low
    python -m src.process.pipeline.retrieval_runner --help
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd
from langchain.schema import Document

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import dotenv
dotenv.load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("elastic_transport.transport").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

from config import DATA_DIR
from src.corpus_handler.parquet_corpus_handler import ParquetCorpusHandler
from src.question_input.huggingface_cyro_input import HuggingFaceCyroInput
from src.rag.elasticsearch_rag_service import ElasticsearchRagService
from src.rag.faiss_rag_service import FaissRagService
from src.rag.bm25_rag_service import BM25RagService
from src.rag.router_rag_service import RouterRagService
from src.rag.hybrid_faiss_rag_service import HybridFaissRagService
from src.rag.utils import IndexingConfig
from src.process.pipeline.latency_utils import time_batch, save_latency


# ═══════════════════════════════════════════════════════════════════════════════
# Backend declaration
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RetrievalBackend:
    """Declaration of one retrieval backend to evaluate.

    Attributes:
        key: Unique identifier used for checkpoint filenames and result keys.
            Becomes ``retrieved_docs_<key>.csv``.
        label: Human-readable name shown in log output.
        type: Backend kind — one of ``"elasticsearch"``, ``"faiss"``,
            ``"bm25"``, or ``"zero_shot"``.
        index_path: Path to the on-disk index directory.  Required for
            ``"faiss"`` and ``"bm25"``; ignored for ``"elasticsearch"`` and
            ``"zero_shot"``.
        es_index: Elasticsearch index name to query.  Defaults to the
            collection-level ``collection_name`` when ``None``.
        es_strategy: Elasticsearch retrieval strategy.  Common values:
            ``"approximation"``, ``"vector"``, ``"bm25"``, ``"hybrid"``.
        faiss_strategy: FAISS index type (``"ivfpq"``, ``"flat"``, ``"hnsw"``).
            Must match the strategy used when the index was built.
        faiss_distance: Distance metric (``"cosine"``, ``"l2"``,
            ``"inner_product"``).
        service_kwargs: Extra keyword arguments forwarded to the service
            constructor.  Use this to override backend-specific parameters
            such as ``ivfpq_nprobe`` for FAISS::

                RetrievalBackend(
                    key            = "ivfpq_high",
                    type           = "faiss",
                    index_path     = DATA_DIR / "wiki_full_bil" / "faiss_high",
                    service_kwargs = {"ivfpq_nprobe": 256},
                )
    """

    key:            str
    label:          str
    type:           Literal["elasticsearch", "faiss", "bm25", "zero_shot", "router", "hybrid_faiss"]
    index_path:     Path | None = None
    es_index:       str | None  = None
    es_strategy:    str         = "approximation"
    faiss_strategy: str         = "ivfpq"
    faiss_distance: str         = "cosine"
    service_kwargs: dict        = field(default_factory=dict)
    router_sub_keys: tuple[str, ...] = ()
    """Keys of other backends (in the same config) to wrap.

    Must be a tuple of 3–4 backend keys in order:
    ``(bm25_plus_key, ivfpq_high_key, ivfpq_low_key[, zero_shot_key])``.
    The referenced backends must appear earlier in ``RetrievalConfig.backends``
    so their services are already loaded.
    """

    def __post_init__(self) -> None:
        if self.index_path is not None:
            self.index_path = Path(self.index_path)


# ═══════════════════════════════════════════════════════════════════════════════
# Settings
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RetrievalConfig:
    """Configuration for the retrieval stage.

    Attributes:
        collection_name: Data folder name under DATA_DIR; also the default
            Elasticsearch index name for backends that don't set ``es_index``.
        output_dir: Subdirectory under the collection folder for all outputs.
        backends: Ordered list of :class:`RetrievalBackend` entries to run.
        dataset_names: HuggingFace dataset names to draw questions from.
        questions_per_decile: Questions sampled per popularity decile (-1 = all).
        top_k: Documents retrieved per question.
        num_candidates: Elasticsearch pre-filter candidate pool size.
        balance_decile_mode: Decile-balancing scheme.
        chunk_size: Text chunk size for embedding-based backends.
        chunk_overlap: Overlap between consecutive chunks.
        embedding_model: Sentence-transformer model identifier.
        embedding_provider: Embedding backend (``"huggingface"`` or ``"modal"``).
        embeddings_request_batch_size: Embedding API request batch size.
        gpu_batch_size: Embedding GPU batch size.
        restart: If True, overwrite all existing retrieval checkpoints.
        restart_keys: If non-empty, only overwrite checkpoints for these keys.
        only_keys: If non-empty, skip all backends whose key is not in this set.
    """

    collection_name:               str  = "wiki_full_bil"
    output_dir:                    str  = "all_qa_8k"
    backends:                      list[RetrievalBackend] = field(default_factory=lambda: [
        RetrievalBackend(
            key         = "zero_shot",
            label       = "Zero-shot (no retrieval)",
            type        = "zero_shot",
        ),
        RetrievalBackend(
            key         = "es_approx",
            label       = "Dense Retrieval (ES hnsw bbq)",
            type        = "elasticsearch",
            es_strategy = "approximation",
        ),
        RetrievalBackend(
            key         = "es_hybrid",
            label       = "Hybrid Retrieval (ES dense + BM25)",
            type        = "elasticsearch",
            es_strategy = "hybrid",
        ),
        RetrievalBackend(
            key         = "bm25",
            label       = "Sparse Retrieval (BM25 Elasticsearch)",
            type        = "elasticsearch",
            es_strategy = "bm25",
        ),
        RetrievalBackend(
            key        = "bm25_naive",
            label      = "Sparse Retrieval (BM25 lucene)",
            type       = "bm25",
            index_path = DATA_DIR / "wiki_full_bil" / "bm25_lucene",
        ),
        RetrievalBackend(
            key        = "bm25_plus",
            label      = "Sparse Retrieval (BM25 plus)",
            type       = "bm25",
            index_path = DATA_DIR / "wiki_full_bil" / "bm25_bm25plus",
        ),
        RetrievalBackend(
            key        = "bm25_plus_nolen",
            label      = "Sparse Retrieval (BM25 plus no length normalisation)",
            type       = "bm25",
            index_path = DATA_DIR / "wiki_full_bil" / "bm25_bm25plus_nolen",
        ),
        RetrievalBackend(
            key            = "ivfpq_low",
            label          = "Dense Retrieval (FAISS low-pop ivfpq)",
            type           = "faiss",
            index_path     = DATA_DIR / "wiki_full_bil" / "faiss_low",
            service_kwargs = {
                "ivfpq_nprobe": 64,
                "docstore_path": DATA_DIR / "wiki_full_bil" / "faiss_high" / "faiss" / "docstore.sqlite",
            },
        ),
        RetrievalBackend(
            key            = "ivfpq_high",
            label          = "Dense Retrieval (FAISS high-pop ivfpq)",
            type           = "faiss",
            index_path     = DATA_DIR / "wiki_full_bil" / "faiss_high",
            service_kwargs = {"ivfpq_nprobe": 256},
        ),
        RetrievalBackend(
            key            = "ivfpq_extremely_high",
            label          = "Dense Retrieval (FAISS extremely high-pop ivfpq)",
            type           = "faiss",
            index_path     = DATA_DIR / "wiki_full_bil" / "faiss_high",
            service_kwargs = {"ivfpq_nprobe": 1024},
        ),
        RetrievalBackend(
            key             = "router",
            label           = "Popularity-aware Router FAISS (BM25+ / IVFPQ-high)",
            type            = "router",
            router_sub_keys = ("bm25_plus", "ivfpq_high"),
        ),
        RetrievalBackend(
            key             = "router_es",
            label           = "Popularity-aware Router ES (BM25+ / ES approx)",
            type            = "router",
            router_sub_keys = ("bm25_plus", "es_approx"),
        ),
        RetrievalBackend(
            key             = "faiss_hybrid",
            label           = "Hybrid Retrieval (FAISS dense + BM25 RRF)",
            type            = "hybrid_faiss",
            router_sub_keys = ("bm25_plus", "ivfpq_high"),
        ),
    ])
    dataset_names:                 tuple[str, ...] = (
        "natural_questions", "trivia_qa", "pop_qa", "fever", "hotpot_qa", "trex"
    )
    questions_per_decile:          int  = 800
    top_k:                         int  = 10
    num_candidates:                int  = 1000
    balance_decile_mode:           Literal["chunk_weighted", "unweighted"] = "chunk_weighted"
    chunk_size:                    int  = 1000
    chunk_overlap:                 int  = 100
    embedding_model:               str  = "Lajavaness/bilingual-embedding-small"
    embedding_provider:            str  = "huggingface"
    embeddings_request_batch_size: int  = 254
    gpu_batch_size:                int  = 254
    restart:                       bool = False
    restart_keys:                  tuple[str, ...] = ()
    only_keys:                     tuple[str, ...] = ()


# ═══════════════════════════════════════════════════════════════════════════════
# Checkpoint helpers (module-level so downstream runners can import them)
# ═══════════════════════════════════════════════════════════════════════════════

def save_retrieved_docs_csv(
    docs: list[list[Document]],
    question_ids: list[str],
    path: Path,
) -> None:
    """Persist retrieved documents to a CSV checkpoint.

    Each row is one retrieved document with columns ``question_id``,
    ``doc_rank``, ``page_content``, and one ``metadata_<key>`` column per
    metadata entry.

    Args:
        docs: Outer list indexed by question; inner list contains Documents.
        question_ids: Question IDs corresponding to each entry in ``docs``.
        path: Destination file path (parent directories created automatically).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for question_id, doc_list in zip(question_ids, docs):
        for doc_rank, doc in enumerate(doc_list):
            row: dict = {
                "question_id":  question_id,
                "doc_rank":     doc_rank,
                "page_content": doc.page_content,
            }
            if hasattr(doc, "metadata") and doc.metadata:
                for k, v in doc.metadata.items():
                    row[f"metadata_{k}"] = v
            rows.append(row)

    df = (
        pd.DataFrame(rows)
        if rows
        else pd.DataFrame(columns=["question_id", "doc_rank", "page_content"])
    )
    df.to_csv(path, index=False)
    logger.info("Saved %d document rows across %d questions to %s", len(rows), len(docs), path)


def _append_retrieved_docs_csv(
    docs: list[list[Document]],
    question_ids: list[str],
    path: Path,
) -> None:
    """Append new retrieval results to an existing CSV checkpoint.

    Rows for *question_ids* that are already present in *path* are not
    duplicated — only genuinely new rows are appended.

    Args:
        docs: Retrieved document lists for the new questions.
        question_ids: Question IDs corresponding to each entry in ``docs``.
        path: Existing CSV written by :func:`save_retrieved_docs_csv`.
    """
    rows = []
    for question_id, doc_list in zip(question_ids, docs):
        for doc_rank, doc in enumerate(doc_list):
            row: dict = {
                "question_id":  question_id,
                "doc_rank":     doc_rank,
                "page_content": doc.page_content,
            }
            if hasattr(doc, "metadata") and doc.metadata:
                for k, v in doc.metadata.items():
                    row[f"metadata_{k}"] = v
            rows.append(row)

    if not rows:
        return

    new_df = pd.DataFrame(rows)
    existing_df = pd.read_csv(path) if path.exists() else pd.DataFrame()
    combined = pd.concat([existing_df, new_df], ignore_index=True)
    combined.to_csv(path, index=False)
    logger.info("Appended %d doc rows for %d questions to %s", len(rows), len(docs), path)


def load_retrieved_docs_csv(
    path: Path,
    question_ids: list[str],
) -> list[list[Document]] | None:
    """Load retrieved documents from a CSV checkpoint.

    Args:
        path: Path written by :func:`save_retrieved_docs_csv`.
        question_ids: Expected question IDs (used for ordering).

    Returns:
        List of document lists (one per question), or ``None`` if the file
        does not exist or cannot be parsed.
    """
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if df.empty or "question_id" not in df.columns:
            return [[] for _ in question_ids]

        id_to_docs: dict[str, list[Document]] = {}
        for qid in df["question_id"].unique():
            q_rows = df[df["question_id"] == qid].sort_values("doc_rank")
            doc_list: list[Document] = []
            for _, row in q_rows.iterrows():
                content = row["page_content"]
                if pd.isna(content):
                    content = ""
                metadata = {
                    col[len("metadata_"):]: row[col]
                    for col in row.index
                    if col.startswith("metadata_") and pd.notna(row[col])
                }
                doc_list.append(Document(page_content=content, metadata=metadata))
            id_to_docs[qid] = doc_list

        results: list[list[Document]] = []
        for qid in question_ids:
            if qid in id_to_docs:
                results.append(id_to_docs[qid])
            else:
                logger.warning("Question ID %s not found in checkpoint — using empty list", qid)
                results.append([])
        return results
    except Exception as e:
        logger.error("Failed to load retrieval checkpoint %s: %s", path, e)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════════

class RetrievalRunner:
    """Stage 1 runner — retrieves documents for all backends.

    Args:
        cfg: Active :class:`RetrievalConfig`.
    """

    def __init__(self, cfg: RetrievalConfig) -> None:
        self.cfg = cfg
        self._collection_folder = DATA_DIR / cfg.collection_name
        self._output_folder     = self._collection_folder / cfg.output_dir

    # ── Service factory ───────────────────────────────────────────────────────

    def _indexing_config(self) -> IndexingConfig:
        cfg = self.cfg
        return IndexingConfig(
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
            embedding_model=cfg.embedding_model,
            embedding_provider=cfg.embedding_provider,
            request_batch_size=cfg.embeddings_request_batch_size,
            gpu_batch_size=cfg.gpu_batch_size,
            normalise_embeddings=True,
            trust_remote_code=True,
        )

    def _load_service(self, backend: RetrievalBackend, loaded_services: dict[str, object] | None = None) -> object:
        """Instantiate and load the RAG service for *backend*.

        Args:
            backend: Backend declaration to load.
            loaded_services: Dict of already-loaded services keyed by backend
                key.  Required when ``backend.type == "router"`` so the router
                can wrap already-loaded sub-backends.

        Returns:
            A loaded RAG service instance, or ``None`` for ``zero_shot``.

        Raises:
            ValueError: If the backend type is unknown or required paths are
                missing.
        """
        if backend.type == "zero_shot":
            return None

        # Return cached instance if available (avoids double-loading for router)
        if loaded_services is not None and backend.key in loaded_services:
            return loaded_services[backend.key]

        if backend.type == "elasticsearch":
            es_index = backend.es_index or self.cfg.collection_name
            service = ElasticsearchRagService(
                config=self._indexing_config(),
                es_url=os.getenv("ELASTICSEARCH_ENDPOINT", ""),
                es_user=os.getenv("ELASTICSEARCH_USERNAME", ""),
                es_password=os.getenv("ELASTICSEARCH_PASSWORD", ""),
                bm25_b=0,
            )
            service.load_index(es_index)
            return service

        if backend.type == "faiss":
            if backend.index_path is None:
                raise ValueError(
                    f"[{backend.key}] RetrievalBackend.index_path is required for type='faiss'"
                )
            faiss_kwargs = dict(backend.service_kwargs)
            docstore_path = faiss_kwargs.pop("docstore_path", None)
            service = FaissRagService(
                config=self._indexing_config(),
                strategy=backend.faiss_strategy,
                distance_strategy=backend.faiss_distance,
                **faiss_kwargs,
            )
            service.load_index(backend.index_path, docstore_path=docstore_path)
            return service

        if backend.type == "bm25":
            if backend.index_path is None:
                raise ValueError(
                    f"[{backend.key}] RetrievalBackend.index_path is required for type='bm25'"
                )
            service = BM25RagService(
                chunk=True,
                chunk_size=self.cfg.chunk_size,
                chunk_overlap=self.cfg.chunk_overlap,
            )
            service.load_index(backend.index_path)
            return service

        if backend.type == "router":
            if not backend.router_sub_keys or len(backend.router_sub_keys) < 2:
                raise ValueError(
                    f"[{backend.key}] router_sub_keys must have exactly 2 entries: "
                    "(sparse_key, dense_key)"
                )
            if loaded_services is None:
                raise ValueError(
                    f"[{backend.key}] loaded_services dict is required for type='router'"
                )
            sparse_key, dense_key = backend.router_sub_keys[:2]

            def _get(key: str) -> object:
                svc = loaded_services.get(key)
                if svc is None:
                    raise ValueError(
                        f"[{backend.key}] router sub-backend '{key}' not found in loaded services. "
                        "Make sure it appears before the router backend in RetrievalConfig.backends."
                    )
                return svc

            return RouterRagService(
                sparse_service=_get(sparse_key),
                dense_service=_get(dense_key),
                **backend.service_kwargs,
            )

        if backend.type == "hybrid_faiss":
            if not backend.router_sub_keys or len(backend.router_sub_keys) < 2:
                raise ValueError(
                    f"[{backend.key}] router_sub_keys must have exactly 2 entries: "
                    "(sparse_key, dense_key)"
                )
            if loaded_services is None:
                raise ValueError(
                    f"[{backend.key}] loaded_services dict is required for type='hybrid_faiss'"
                )
            sparse_key, dense_key = backend.router_sub_keys[:2]

            def _get_hybrid(key: str) -> object:
                svc = loaded_services.get(key)
                if svc is None:
                    raise ValueError(
                        f"[{backend.key}] hybrid sub-backend '{key}' not found in loaded_services. "
                        "Make sure it appears before the hybrid_faiss backend in RetrievalConfig.backends."
                    )
                return svc

            return HybridFaissRagService(
                sparse_service=_get_hybrid(sparse_key),
                dense_service=_get_hybrid(dense_key),
                **backend.service_kwargs,
            )

        raise ValueError(f"Unknown backend type: {backend.type!r}")

    # ── Per-backend retrieval ─────────────────────────────────────────────────

    def retrieve_for_backend(
        self,
        backend: RetrievalBackend,
        questions: list[str],
        question_ids: list[str] | None = None,
        popularities: list[float] | None = None,
        loaded_services: dict[str, object] | None = None,
    ) -> tuple[list[list[Document]], list[float]]:
        """Run retrieval for one backend.

        Args:
            backend: The backend to retrieve from.
            questions: Question strings in evaluation order.
            question_ids: Optional question IDs used for latency records.
                Defaults to positional string indices when not provided.
            popularities: Per-question popularity scores.  Required when
                ``backend.type == "router"``; ignored otherwise.
            loaded_services: Already-loaded service instances keyed by backend
                key.  Required when ``backend.type == "router"`` so the router
                can be constructed from pre-loaded sub-backends.

        Returns:
            Tuple of (list of document lists — one per question,
            list of per-question latency in ms).
        """
        _ids = question_ids or [str(i) for i in range(len(questions))]

        if backend.type == "zero_shot":
            return [[] for _ in questions], [0.0] * len(questions)

        service = self._load_service(backend, loaded_services=loaded_services)

        # Cache so later router backends can reuse without re-loading
        if loaded_services is not None and backend.key not in loaded_services and backend.type != "router":
            loaded_services[backend.key] = service

        if service.get_doc_count() == 0:
            logger.error(
                "[%s] Index is empty — build the index first and verify the path.", backend.key
            )
            return [[] for _ in questions], [0.0] * len(questions)

        if backend.type == "router":
            if not popularities or len(popularities) != len(questions):
                raise ValueError(
                    f"[{backend.key}] popularities must be provided for the router backend "
                    f"and must match the number of questions ({len(questions)})"
                )
            scored, latencies_ms = time_batch(
                lambda: service.batch_retrieve_with_scores(
                    questions,
                    top_k=self.cfg.top_k,
                    popularities=popularities,
                ),
                _ids,
            )
        elif backend.type == "elasticsearch":
            scored, latencies_ms = time_batch(
                lambda: service.batch_retrieve_with_scores(
                    questions,
                    top_k=self.cfg.top_k,
                    strategy=backend.es_strategy,
                    search_workers=6,
                    msearch_batch_size=5,
                    num_candidates=self.cfg.num_candidates,
                ),
                _ids,
            )
        else:
            scored, latencies_ms = time_batch(
                lambda: service.batch_retrieve_with_scores(questions, top_k=self.cfg.top_k),
                _ids,
            )

        return [[doc for doc, _ in docs] for docs in scored], latencies_ms

    # ── Main run ──────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Retrieve documents for all configured backends and save CSV checkpoints.

        Existing checkpoints are reused unless ``cfg.restart`` is True or the
        backend key appears in ``cfg.restart_keys``.
        """
        self._output_folder.mkdir(parents=True, exist_ok=True)

        corpus_handler = ParquetCorpusHandler(
            corpus_path=self._collection_folder / "wiki_corpus.parquet",
            metadata_path=self._collection_folder / "metadata.json",
        )
        question_input = HuggingFaceCyroInput(
            dataset_names=list(self.cfg.dataset_names),
            corpus_handler=corpus_handler,
            parquet_path=self._output_folder / "cyro_qa_cache.parquet",
            balance_deciles=True,
            balance_datasets=True,
            target_per_decile=self.cfg.questions_per_decile,
            shuffle=True,
            balance_decile_mode=self.cfg.balance_decile_mode,
        )
        question_input.load(force=self.cfg.restart)
        question_data = question_input.get_items()
        logger.info("Loaded %d questions", len(question_data))

        questions    = [item.question_text for item in question_data]
        question_ids = [item.question_id   for item in question_data]
        popularities = [item.popularity_avg if item.popularity_avg is not None else 0.0
                        for item in question_data]

        # Cache of already-loaded services so router can reuse sub-backends
        # without loading them a second time.
        loaded_services: dict[str, object] = {}

        # Pre-load any sub-backends required by a router or hybrid_faiss backend that is being
        # run, even if those sub-backends are not in --only-keys themselves.
        router_backends_to_run = [
            b for b in self.cfg.backends
            if b.type in ("router", "hybrid_faiss")
            and (not self.cfg.only_keys or b.key in self.cfg.only_keys)
        ]
        required_sub_keys: set[str] = set()
        for rb in router_backends_to_run:
            required_sub_keys.update(rb.router_sub_keys)

        backend_by_key = {b.key: b for b in self.cfg.backends}
        for sub_key in required_sub_keys:
            if sub_key in loaded_services:
                continue
            sub_backend = backend_by_key.get(sub_key)
            if sub_backend is None or sub_backend.type in ("zero_shot", "router", "hybrid_faiss"):
                continue
            logger.info("⚙ [%s] Pre-loading sub-backend for router…", sub_key)
            try:
                loaded_services[sub_key] = self._load_service(sub_backend, loaded_services)
            except Exception as e:
                logger.error("[%s] Failed to pre-load router sub-backend: %s", sub_key, e, exc_info=True)

        for backend in self.cfg.backends:
            if self.cfg.only_keys and backend.key not in self.cfg.only_keys:
                logger.info("⏭ [%s] %s — skipped (not in --only-keys)", backend.key, backend.label)
                continue

            checkpoint    = self._output_folder / f"retrieved_docs_{backend.key}.csv"
            force_restart = self.cfg.restart or (backend.key in self.cfg.restart_keys)

            if force_restart and checkpoint.exists():
                logger.info("♻ [%s] %s — overwriting checkpoint", backend.key, backend.label)
                checkpoint.unlink()

            # ── Incremental resume: find which question_ids are missing ────
            if checkpoint.exists():
                existing_docs = load_retrieved_docs_csv(checkpoint, question_ids)
                # load_retrieved_docs_csv returns [] for any ID not in the CSV
                missing_mask  = [len(docs) == 0 for docs in existing_docs]
                missing_ids   = [qid  for qid,  m in zip(question_ids, missing_mask) if m]
                missing_qs    = [q    for q,     m in zip(questions,    missing_mask) if m]
                missing_pops  = [pop  for pop,   m in zip(popularities, missing_mask) if m]

                if not missing_ids:
                    logger.info("✓ [%s] %s — all %d questions done, skipping", backend.key, backend.label, len(question_ids))
                    # Ensure service is cached so a later router backend can reference it
                    if backend.type not in ("zero_shot", "router", "hybrid_faiss") and backend.key not in loaded_services:
                        try:
                            loaded_services[backend.key] = self._load_service(backend, loaded_services)
                        except Exception as e:
                            logger.warning("[%s] Could not cache service for router: %s", backend.key, e)
                    continue

                logger.info(
                    "↻ [%s] %s — %d / %d questions missing, retrieving remainder…",
                    backend.key, backend.label, len(missing_ids), len(question_ids),
                )
                try:
                    new_docs, latencies_ms = self.retrieve_for_backend(
                        backend, missing_qs, missing_ids,
                        popularities=missing_pops,
                        loaded_services=loaded_services,
                    )
                except Exception as e:
                    logger.error("[%s] Retrieval failed: %s", backend.key, e, exc_info=True)
                    continue

                _append_retrieved_docs_csv(new_docs, missing_ids, checkpoint)
                logger.info("✓ [%s] Appended %d results → %s", backend.key, len(new_docs), checkpoint)

            else:
                logger.info("▶ [%s] %s — retrieving…", backend.key, backend.label)
                try:
                    retrieved, latencies_ms = self.retrieve_for_backend(
                        backend, questions, question_ids,
                        popularities=popularities,
                        loaded_services=loaded_services,
                    )
                except Exception as e:
                    logger.error("[%s] Retrieval failed: %s", backend.key, e, exc_info=True)
                    continue

                save_retrieved_docs_csv(retrieved, question_ids, checkpoint)
                logger.info("✓ [%s] Saved %d results → %s", backend.key, len(retrieved), checkpoint)

            # Cache non-router services for potential later router use
            if backend.type not in ("zero_shot", "router", "hybrid_faiss") and backend.key not in loaded_services:
                try:
                    loaded_services[backend.key] = self._load_service(backend, loaded_services)
                except Exception as e:
                    logger.warning("[%s] Could not cache service for router: %s", backend.key, e)

            # Cache non-router services for potential later router use
            if backend.type not in ("zero_shot", "router", "hybrid_faiss") and backend.key not in loaded_services:
                try:
                    loaded_services[backend.key] = self._load_service(backend, loaded_services)
                except Exception as e:
                    logger.warning("[%s] Could not cache service for router: %s", backend.key, e)

            latency_path = self._output_folder / f"latency_retrieval_{backend.key}.json"
            save_latency(
                path=latency_path,
                backend_key=backend.key,
                stage="retrieval",
                question_ids=question_ids,
                latencies_ms=latencies_ms,
            )

    # ── Entry point ───────────────────────────────────────────────────────────

    @classmethod
    def main(cls, argv: list[str] | None = None) -> None:
        """Parse CLI arguments, instantiate the runner, and call :meth:`run`."""
        p = argparse.ArgumentParser(
            description="Stage 1 — retrieve documents for all backends and save CSV checkpoints.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        _d = RetrievalConfig()
        p.add_argument("--collection", "-c", default=_d.collection_name)
        p.add_argument("--output-dir", "-o", default=_d.output_dir)
        p.add_argument("--top-k", type=int, default=_d.top_k)
        p.add_argument("--questions-per-decile", type=int, default=_d.questions_per_decile)
        p.add_argument("--restart", action="store_true",
                       help="Overwrite all existing retrieval checkpoints.")
        p.add_argument("--restart-keys", nargs="+", default=[],
                       help="Overwrite checkpoints only for these backend keys.")
        p.add_argument("--only-keys", nargs="+", default=[],
                       help="Run only these backend keys; skip all others.")
        args = p.parse_args(argv)

        cfg = RetrievalConfig(
            collection_name=args.collection,
            output_dir=args.output_dir,
            top_k=args.top_k,
            questions_per_decile=args.questions_per_decile,
            restart=args.restart,
            restart_keys=tuple(args.restart_keys),
            only_keys=tuple(args.only_keys),
        )
        cls(cfg).run()


if __name__ == "__main__":
    RetrievalRunner.main()
