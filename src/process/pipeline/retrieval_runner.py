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
    RetrievalBackend(
        key             = "neural_router_strict",
        label           = "Neural Router (Strict - Argmax)",
        type            = "neural_router",
        router_sub_keys = ("bm25", "faiss_high"),
        service_kwargs  = {
            "model_path": "models/router_mrr20.pt",
            "backend_order": ["bm25", "faiss_high"],
            "strict": True,
        },
    ),
    RetrievalBackend(
        key             = "neural_router_hybrid",
        label           = "Neural Router (Hybrid - Probability Weighted RRF)",
        type            = "neural_router",
        router_sub_keys = ("bm25", "faiss_high"),
        service_kwargs  = {
            "model_path": "models/router_mrr20.pt",
            "backend_order": ["bm25", "faiss_high"],
            "strict": False,
            "rrf_k": 60,
            "rrf_depth": 60,
        },
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
import gc
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd
from langchain.schema import Document
from tqdm import tqdm

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

from config import DATA_DIR, ROOT_DIR
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
    type:           Literal["elasticsearch", "faiss", "bm25", "zero_shot", "router", "hybrid_faiss", "neural_router"]
    index_path:     Path | None = None
    es_index:       str | None  = None
    es_strategy:    str         = "approximation"
    faiss_strategy: str         = "ivfpq"
    faiss_distance: str         = "cosine"
    service_kwargs: dict        = field(default_factory=dict)
    router_sub_keys: tuple[str, ...] = ()
    """Keys of other backends (in the same config) to wrap.

    For router/hybrid_faiss: tuple of 2 backend keys (sparse_key, dense_key).
    For neural_router: tuple of 2+ backend keys in the order model was trained.
    The referenced backends must appear earlier in ``RetrievalConfig.backends``
    so their services are already loaded.
    """

    def __post_init__(self) -> None:
        if self.index_path is not None:
            self.index_path = Path(self.index_path)


def router_backends_from_models_dir(
    models_dir: Path | None = None,
    sub_keys: tuple[str, str] = ("bm25_plus", "ivfpq_high"),
) -> list[RetrievalBackend]:
    """Build RetrievalBackend entries for every router*.pt in a directory.

    Each ``router<name>.pt`` file yields two backends so both routing
    strategies can be compared:

      - ``<stem>``       — strict (argmax) routing
      - ``<stem>_hybrid`` — probability-weighted RRF routing

    The backend key equals the file stem (e.g. ``router_mrr_filter_e80``
    or ``router60k_frozen_wd5e-3_drop60_mrr20_s42``).  For backward
    compatibility, the old ``neural_<stem>`` prefix is resolved as an
    alias in :func:`full_pipeline._retrieval_backends_for_keys`.

    The referenced *sub_keys* (default: ``bm25_plus`` and ``ivfpq_high``)
    must be declared earlier in :class:`RetrievalConfig.backends` so their
    services are loaded before the router is constructed.

    Args:
        models_dir: Directory containing the trained ``router*.pt`` files.
            Defaults to ``ROOT_DIR / "models"``.
        sub_keys: Backend keys the router chooses between. Order matters —
            it must match the ``backend_order`` the model was trained on.

    Returns:
        Ordered list of :class:`RetrievalBackend` entries (two per model).

    Raises:
        FileNotFoundError: If *models_dir* does not exist.
    """
    models_dir = Path(models_dir) if models_dir else ROOT_DIR / "models"
    if not models_dir.exists():
        raise FileNotFoundError(
            f"router_backends_from_models_dir: '{models_dir}' does not exist"
        )

    pt_files = sorted(models_dir.glob("router*.pt"))
    if not pt_files:
        logger.warning("No router*.pt files found in %s", models_dir)
        return []

    backends: list[RetrievalBackend] = []
    for pt in pt_files:
        stem        = pt.stem                    # e.g. "router_mrr_filter_e80"
        model_path  = str(pt.resolve())          # absolute — survives CWD changes
        backend_ord = list(sub_keys)

        backends.append(RetrievalBackend(
            key             = stem,
            label           = f"Neural Router ({stem}) — strict",
            type            = "neural_router",
            router_sub_keys = sub_keys,
            service_kwargs  = {
                "model_path":    model_path,
                "backend_order": backend_ord,
                "strict":        True,
            },
        ))
        backends.append(RetrievalBackend(
            key             = f"{stem}_hybrid",
            label           = f"Neural Router ({stem}) — hybrid RRF",
            type            = "neural_router",
            router_sub_keys = sub_keys,
            service_kwargs  = {
                "model_path":    model_path,
                "backend_order": backend_ord,
                "strict":        False,
                "rrf_k":         60,
                "rrf_depth":     60,
            },
        ))

    logger.info(
        "Discovered %d router models in %s → %d backends (strict + hybrid)",
        len(pt_files), models_dir, len(backends),
    )
    return backends


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
        RetrievalBackend(
            key             = "neural_router_strict",
            label           = "Neural Router (Strict - Argmax)",
            type            = "neural_router",
            router_sub_keys = ("bm25_plus", "ivfpq_high"),
            service_kwargs  = {
                "model_path": str(DATA_DIR / "wiki_full_bil" / "models" / "router_pop_after_bert.pt"),
                "backend_order": ["bm25_plus", "ivfpq_high"],
                "strict": True,
            },
        ),
        RetrievalBackend(
            key             = "neural_router_hybrid",
            label           = "Neural Router (Hybrid - Probability Weighted RRF)",
            type            = "neural_router",
            router_sub_keys = ("bm25_plus", "ivfpq_high"),
            service_kwargs  = {
                "model_path": str(DATA_DIR / "wiki_full_bil" / "models" / "router_pop_after_bert.pt"),
                "backend_order": ["bm25_plus", "ivfpq_high"],
                "strict": False,
                "rrf_k": 60,
                "rrf_depth": 60,
            },
        ),
        RetrievalBackend(
            key             = "neural_router_plain_bert",
            label           = "Neural Router (Plain BERT - mrr@60 with popularity)",
            type            = "neural_router",
            router_sub_keys = ("bm25_plus", "ivfpq_high"),
            service_kwargs  = {
                "model_path": "models/router_plain_bert.pt",
                "backend_order": ["bm25_plus", "ivfpq_high"],
                "strict": True,
            },
        ),
        RetrievalBackend(
            key             = "neural_router_plain_bert_hybrid",
            label           = "Neural Router (Plain BERT Hybrid - mrr@60 with popularity)",
            type            = "neural_router",
            router_sub_keys = ("bm25_plus", "ivfpq_high"),
            service_kwargs  = {
                "model_path": "models/router_plain_bert.pt",
                "backend_order": ["bm25_plus", "ivfpq_high"],
                "strict": False,
                "rrf_k": 60,
                "rrf_depth": 60,
            },
        ),
        RetrievalBackend(
            key             = "neural_router_no_pop_answer",
            label           = "Neural Router (No Pop - answer mode, no popularity)",
            type            = "neural_router",
            router_sub_keys = ("bm25_plus", "ivfpq_high"),
            service_kwargs  = {
                "model_path": "models/router_plain_bert_no_pop_answer.pt",
                "backend_order": ["bm25_plus", "ivfpq_high"],
                "strict": True,
            },
        ),
        RetrievalBackend(
            key             = "neural_router_no_pop_answer_hybrid",
            label           = "Neural Router (No Pop Hybrid - answer mode, no popularity)",
            type            = "neural_router",
            router_sub_keys = ("bm25_plus", "ivfpq_high"),
            service_kwargs  = {
                "model_path": "models/router_plain_bert_no_pop_answer.pt",
                "backend_order": ["bm25_plus", "ivfpq_high"],
                "strict": False,
                "rrf_k": 60,
                "rrf_depth": 60,
            },
        ),
        RetrievalBackend(
            key             = "neural_router_pop_after_bert",
            label           = "Neural Router (Pop After BERT - mrr@20 with popularity)",
            type            = "neural_router",
            router_sub_keys = ("bm25_plus", "ivfpq_high"),
            service_kwargs  = {
                "model_path": "models/router_pop_after_bert.pt",
                "backend_order": ["bm25_plus", "ivfpq_high"],
                "strict": True,
            },
        ),
        RetrievalBackend(
            key             = "neural_router_pop_after_bert_hybrid",
            label           = "Neural Router (Pop After BERT Hybrid - mrr@20 with popularity)",
            type            = "neural_router",
            router_sub_keys = ("bm25_plus", "ivfpq_high"),
            service_kwargs  = {
                "model_path": "models/router_pop_after_bert.pt",
                "backend_order": ["bm25_plus", "ivfpq_high"],
                "strict": False,
                "rrf_k": 60,
                "rrf_depth": 60,
            },
        ),
        RetrievalBackend(
            key             = "neural_router_unfreeze1_answer",
            label           = "Neural Router (Unfreeze1 - answer mode, 1 BERT layer)",
            type            = "neural_router",
            router_sub_keys = ("bm25_plus", "ivfpq_high"),
            service_kwargs  = {
                "model_path": "models/router_unfreeze1_bert_answer.pt",
                "backend_order": ["bm25_plus", "ivfpq_high"],
                "strict": True,
            },
        ),
        RetrievalBackend(
            key             = "neural_router_unfreeze1_answer_hybrid",
            label           = "Neural Router (Unfreeze1 Hybrid - answer mode, 1 BERT layer)",
            type            = "neural_router",
            router_sub_keys = ("bm25_plus", "ivfpq_high"),
            service_kwargs  = {
                "model_path": "models/router_unfreeze1_bert_answer.pt",
                "backend_order": ["bm25_plus", "ivfpq_high"],
                "strict": False,
                "rrf_k": 60,
                "rrf_depth": 60,
            },
        ),
        RetrievalBackend(
            key             = "neural_router_bert_answer",
            label           = "Neural Router (BERT answer)",
            type            = "neural_router",
            router_sub_keys = ("bm25_plus", "ivfpq_high"),
            service_kwargs  = {
                "model_path": "models/router_bert_answer.pt",
                "backend_order": ["bm25_plus", "ivfpq_high"],
                "strict": True,
            },
        ),
        RetrievalBackend(
            key             = "neural_router_bert_answer_hybrid",
            label           = "Neural Router (BERT answer hybrid)",
            type            = "neural_router",
            router_sub_keys = ("bm25_plus", "ivfpq_high"),
            service_kwargs  = {
                "model_path": "models/router_bert_answer.pt",
                "backend_order": ["bm25_plus", "ivfpq_high"],
                "strict": False,
                "rrf_k": 60,
                "rrf_depth": 60,
            },
        ),
        RetrievalBackend(
            key             = "neural_router_mrr_no_pop",
            label           = "Neural Router (MRR no-pop)",
            type            = "neural_router",
            router_sub_keys = ("bm25_plus", "ivfpq_high"),
            service_kwargs  = {
                "model_path": "models/router_mrr_no_pop.pt",
                "backend_order": ["bm25_plus", "ivfpq_high"],
                "strict": True,
            },
        ),
        RetrievalBackend(
            key             = "neural_router_mrr_no_pop_hybrid",
            label           = "Neural Router (MRR no-pop hybrid)",
            type            = "neural_router",
            router_sub_keys = ("bm25_plus", "ivfpq_high"),
            service_kwargs  = {
                "model_path": "models/router_mrr_no_pop.pt",
                "backend_order": ["bm25_plus", "ivfpq_high"],
                "strict": False,
                "rrf_k": 60,
                "rrf_depth": 60,
            },
        ),
        RetrievalBackend(
            key             = "neural_router_plain_bert_mrr_filter",
            label           = "Neural Router (Plain BERT MRR filter)",
            type            = "neural_router",
            router_sub_keys = ("bm25_plus", "ivfpq_high"),
            service_kwargs  = {
                "model_path": "models/router_plain_bert_mrr_filter.pt",
                "backend_order": ["bm25_plus", "ivfpq_high"],
                "strict": True,
            },
        ),
        RetrievalBackend(
            key             = "neural_router_plain_bert_mrr_filter_hybrid",
            label           = "Neural Router (Plain BERT MRR filter hybrid)",
            type            = "neural_router",
            router_sub_keys = ("bm25_plus", "ivfpq_high"),
            service_kwargs  = {
                "model_path": "models/router_plain_bert_mrr_filter.pt",
                "backend_order": ["bm25_plus", "ivfpq_high"],
                "strict": False,
                "rrf_k": 60,
                "rrf_depth": 60,
            },
        ),
        RetrievalBackend(
            key             = "neural_router_unfreeze1_no_pop_answer",
            label           = "Neural Router (Unfreeze1 no-pop answer)",
            type            = "neural_router",
            router_sub_keys = ("bm25_plus", "ivfpq_high"),
            service_kwargs  = {
                "model_path": "models/router_unfreeze1_no_pop_answer.pt",
                "backend_order": ["bm25_plus", "ivfpq_high"],
                "strict": True,
            },
        ),
        RetrievalBackend(
            key             = "neural_router_unfreeze1_no_pop_answer_hybrid",
            label           = "Neural Router (Unfreeze1 no-pop answer hybrid)",
            type            = "neural_router",
            router_sub_keys = ("bm25_plus", "ivfpq_high"),
            service_kwargs  = {
                "model_path": "models/router_unfreeze1_no_pop_answer.pt",
                "backend_order": ["bm25_plus", "ivfpq_high"],
                "strict": False,
                "rrf_k": 60,
                "rrf_depth": 60,
            },
        ),
    ])
    dataset_names:                 tuple[str, ...] = (
        "natural_questions", "trivia_qa", "pop_qa", "fever", "hotpot_qa", "trex"
    )
    qa_file:                       Path | None = None  # If set, load from local parquet instead of HF
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
        df = pd.read_csv(path, dtype={"question_id": str})
        if df.empty or "question_id" not in df.columns:
            return [[] for _ in question_ids]

        id_to_docs: dict[str, list[Document]] = {}
        
        # Sort entire dataframe by question_id and doc_rank at once
        df = df.sort_values(["question_id", "doc_rank"])
        
        metadata_cols = [col for col in df.columns if col.startswith("metadata_")]
        
        for qid, group in df.groupby("question_id"):
            doc_list: list[Document] = []
            for row in group.itertuples(index=False):
                row_dict = row._asdict()
                content = row_dict["page_content"]
                if pd.isna(content):
                    content = ""
                metadata = {
                    col[len("metadata_"):]: row_dict[col]
                    for col in metadata_cols
                    if pd.notna(row_dict[col])
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
        self._precomputed_cache: dict[str, dict[str, list[list[Document]]]] = {}
        self._shared_precomputed_docs: dict[str, list[list[Document]]] = {}
        self._shared_precomputed_depths: dict[str, int] = {}
        self._shared_precomputed_refcounts: dict[str, int] = {}

    # ── Question loading helper ───────────────────────────────────────────────

    def _load_from_local_parquet(self, qa_file: Path) -> list:
        """Load questions from a local parquet file (e.g., from prepare_qa.py).

        Args:
            qa_file: Path to the parquet file containing questions.

        Returns:
            List of QuestionItem instances.
        """
        import pandas as pd
        from src.question_input.base import QuestionItem
        from src.metrics.decile_utils import COL_DECILE_UNWEIGHTED, COL_DECILE_CHUNK_WEIGHTED, COL_POPULARITY

        if not qa_file.exists():
            raise FileNotFoundError(f"QA file not found: {qa_file}")

        df = pd.read_parquet(qa_file)
        logger.info("Loaded %d rows from %s", len(df), qa_file)

        # Build QuestionItem list
        items = []
        for _, row in df.iterrows():
            # Handle both old and new column names
            decile_uw = row.get(COL_DECILE_UNWEIGHTED, row.get("decile", -1))
            decile_cw = row.get(COL_DECILE_CHUNK_WEIGHTED, row.get("decile", -1))
            legacy_decile = row.get("decile", decile_uw if decile_uw != -1 else -1)
            
            items.append(QuestionItem(
                question_id=str(row.get("question_id", "")),
                question_text=str(row.get("question_text", "")),
                answer_texts=row.get("answer_texts", []) if isinstance(row.get("answer_texts"), list) else [],
                wikipedia_id=str(row.get("wikipedia_id", "")),
                wikipedia_title=str(row.get("wikipedia_title", "")),
                decile=int(legacy_decile) if pd.notna(legacy_decile) else -1,
                decile_unweighted=int(decile_uw) if pd.notna(decile_uw) else -1,
                decile_chunk_weighted=int(decile_cw) if pd.notna(decile_cw) else -1,
                dataset=str(row.get("dataset", "")),
                popularity_avg=float(row.get(COL_POPULARITY, 0.0)) if pd.notna(row.get(COL_POPULARITY)) else None,
            ))

        return items

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
            sparse_key, dense_key = backend.router_sub_keys[:2]

            # ── Pre-computed path: skip live sub-backend loading ────────────
            precomputed = self._precomputed_cache.get(backend.key)
            if precomputed is not None:
                logger.info("⚡ [%s] Using pre-computed sub-backend results", backend.key)
                service = RouterRagService(**backend.service_kwargs)
                service.set_precomputed_results({
                    "dense":  precomputed.get(dense_key, []),
                    "sparse": precomputed.get(sparse_key, []),
                })
                return service

            # ── Live path: requires loaded sub-backend services ──────────────
            if loaded_services is None:
                raise ValueError(
                    f"[{backend.key}] loaded_services dict is required for type='router'"
                )

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
            sparse_key, dense_key = backend.router_sub_keys[:2]

            # ── Pre-computed path: skip live sub-backend loading ────────────
            precomputed = self._precomputed_cache.get(backend.key)
            if precomputed is not None:
                logger.info("⚡ [%s] Using pre-computed sub-backend results", backend.key)
                service = HybridFaissRagService(**backend.service_kwargs)
                service.set_precomputed_results({
                    "dense":  precomputed.get(dense_key, []),
                    "sparse": precomputed.get(sparse_key, []),
                })
                return service

            # ── Live path: requires loaded sub-backend services ──────────────
            if loaded_services is None:
                raise ValueError(
                    f"[{backend.key}] loaded_services dict is required for type='hybrid_faiss'"
                )

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
        
        if backend.type == "neural_router":
            # Neural router requires: router_sub_keys (backend names in order),
            # model_path, backend_order, strict, rrf_k, rrf_depth
            if not backend.router_sub_keys or len(backend.router_sub_keys) < 2:
                raise ValueError(
                    f"[{backend.key}] router_sub_keys must have 2+ entries for neural_router"
                )

            # Extract configuration from service_kwargs
            kwargs = dict(backend.service_kwargs)
            model_path = kwargs.pop("model_path", None)
            backend_order = kwargs.pop("backend_order", None)

            if model_path is None:
                raise ValueError(
                    f"[{backend.key}] service_kwargs must include 'model_path' for neural_router"
                )
            if backend_order is None:
                raise ValueError(
                    f"[{backend.key}] service_kwargs must include 'backend_order' for neural_router"
                )

            # ── Pre-computed path: skip live sub-backend loading ────────────
            precomputed = self._precomputed_cache.get(backend.key)
            if precomputed is not None:
                logger.info("⚡ [%s] Using pre-computed sub-backend results", backend.key)
                from src.rag.neural_router_rag_service import NeuralRouterRagService
                service = NeuralRouterRagService(
                    backends={},
                    backend_order=backend_order,
                    model_path=model_path,
                    **kwargs,
                )
                service.set_precomputed_results(precomputed)
                return service

            # ── Live path: requires loaded sub-backend services ──────────────
            if loaded_services is None:
                raise ValueError(
                    f"[{backend.key}] loaded_services dict is required for type='neural_router'"
                )

            backends_dict = {}
            for sub_key in backend.router_sub_keys:
                svc = loaded_services.get(sub_key)
                if svc is None:
                    raise ValueError(
                        f"[{backend.key}] neural_router sub-backend '{sub_key}' not found. "
                        "Make sure it appears before neural_router in RetrievalConfig.backends."
                    )
                backends_dict[sub_key] = svc
            
            from src.rag.neural_router_rag_service import NeuralRouterRagService
            
            return NeuralRouterRagService(
                backends=backends_dict,
                backend_order=backend_order,
                model_path=model_path,
                **kwargs,  # Remaining: strict, rrf_k, rrf_depth, predict_batch_size, device
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
        *,
        override_top_k: int | None = None,
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
            override_top_k: When provided, use this instead of ``cfg.top_k``.
                Used by the auto-upgrade path to retrieve sub-backends at a
                deeper depth for RRF fusion.

        Returns:
            Tuple of (list of document lists — one per question,
            list of per-question latency in ms).
        """
        _ids = question_ids or [str(i) for i in range(len(questions))]
        _top_k = override_top_k or self.cfg.top_k

        if backend.type == "zero_shot":
            return [[] for _ in questions], [0.0] * len(questions)

        service = self._load_service(backend, loaded_services=loaded_services)

        # Cache so later router backends can reuse without re-loading
        if loaded_services is not None and backend.key not in loaded_services and backend.type not in ("router", "neural_router"):
            loaded_services[backend.key] = service

        if service.get_doc_count() == 0:
            logger.error(
                "[%s] Index is empty — build the index first and verify the path.", backend.key
            )
            return [[] for _ in questions], [0.0] * len(questions)

        if backend.type in ("router", "neural_router"):
            if not popularities or len(popularities) != len(questions):
                raise ValueError(
                    f"[{backend.key}] popularities must be provided for router backends "
                    f"and must match the number of questions ({len(questions)})"
                )
            scored, latencies_ms = time_batch(
                lambda: service.batch_retrieve_with_scores(
                    questions,
                    top_k=_top_k,
                    popularities=popularities,
                ),
                _ids,
            )
        elif backend.type == "elasticsearch":
            scored, latencies_ms = time_batch(
                lambda: service.batch_retrieve_with_scores(
                    questions,
                    top_k=_top_k,
                    strategy=backend.es_strategy,
                    search_workers=6,
                    msearch_batch_size=5,
                    num_candidates=self.cfg.num_candidates,
                ),
                _ids,
            )
        else:
            scored, latencies_ms = time_batch(
                lambda: service.batch_retrieve_with_scores(questions, top_k=_top_k),
                _ids,
            )

        return [[doc for doc, _ in docs] for docs in scored], latencies_ms

    # ── Pre-computed sub-backend reuse helpers ─────────────────────────────────

    def _compute_required_depth(self, backend: RetrievalBackend) -> int:
        """Determine the required retrieval depth for a backend's sub-backends.

        Hybrid/RRF modes need ``max(rrf_depth, top_k)`` docs per sub-backend
        for proper fusion. Strict modes only need ``top_k``.

        Args:
            backend: A composite backend declaration.

        Returns:
            Required number of docs per question per sub-backend.
        """
        top_k = self.cfg.top_k
        if backend.type == "neural_router":
            strict = backend.service_kwargs.get("strict", True)
            if not strict:
                rrf_depth = backend.service_kwargs.get("rrf_depth", 60)
                return max(rrf_depth, top_k)
        elif backend.type == "hybrid_faiss":
            rrf_depth = backend.service_kwargs.get("rrf_depth", 60)
            return max(rrf_depth, top_k)
        # router (popularity) and strict neural_router only need top_k
        return top_k

    def _try_load_precomputed(
        self,
        sub_keys: tuple[str, ...],
        question_ids: list[str],
        required_depth: int,
    ) -> tuple[dict[str, list[list[Document]]] | None, list[str]]:
        """Try to load pre-computed sub-backend results from CSVs.

        Args:
            sub_keys: Sub-backend keys (e.g., ``("bm25_plus", "ivfpq_high")``).
            question_ids: All question IDs in evaluation order.
            required_depth: Minimum docs per question needed.

        Returns:
            Tuple of ``(precomputed_dict, insufficient_keys)``.
            ``precomputed_dict`` is ``None`` if any CSV is missing entirely.
            ``insufficient_keys`` lists sub-keys whose CSV exists but has
            fewer docs than ``required_depth`` (candidates for auto-upgrade).
        """
        precomputed: dict[str, list[list[Document]]] = {}
        insufficient: list[str] = []
        for sub_key in sub_keys:
            if sub_key in self._shared_precomputed_docs:
                docs = self._shared_precomputed_docs[sub_key]
                max_docs = self._shared_precomputed_depths.get(sub_key, 0)
                if max_docs < required_depth:
                    insufficient.append(sub_key)
                precomputed[sub_key] = docs
                continue

            path = self._output_folder / f"retrieved_docs_{sub_key}.csv"
            if not path.exists():
                return None, list(sub_keys)

            logger.info("📥 [%s] Loading pre-computed CSV into memory: %s", sub_key, path.name)
            docs = load_retrieved_docs_csv(path, question_ids)
            if docs is None:
                return None, list(sub_keys)

            max_docs = max((len(d) for d in docs), default=0)
            self._shared_precomputed_docs[sub_key] = docs
            self._shared_precomputed_depths[sub_key] = max_docs
            if max_docs < required_depth:
                insufficient.append(sub_key)
            precomputed[sub_key] = docs
        return precomputed, insufficient

    def _upgrade_sub_backend(
        self,
        sub_key: str,
        required_depth: int,
        questions: list[str],
        question_ids: list[str],
        popularities: list[float],
        backend_by_key: dict[str, RetrievalBackend],
        loaded_services: dict[str, object],
    ) -> list[list[Document]]:
        """Re-retrieve a sub-backend at a deeper depth and overwrite its CSV.

        Used when an existing CSV has fewer docs than needed for RRF fusion.

        Args:
            sub_key: Sub-backend key to upgrade.
            required_depth: Target number of docs per question.
            questions: All question texts.
            question_ids: All question IDs.
            popularities: Per-question popularity scores.
            backend_by_key: Mapping from backend key to declaration.
            loaded_services: Cache of loaded services.

        Returns:
            The deeper retrieval results as ``list[list[Document]]``.
        """
        sub_backend = backend_by_key.get(sub_key)
        if sub_backend is None:
            raise ValueError(f"Sub-backend '{sub_key}' not found in config")

        old_path = self._output_folder / f"retrieved_docs_{sub_key}.csv"
        old_count = 0
        if old_path.exists():
            old_docs = load_retrieved_docs_csv(old_path, question_ids)
            if old_docs:
                old_count = max((len(d) for d in old_docs), default=0)

        logger.info(
            "♻ [%s] Auto-upgrading CSV: depth %d → %d (needed for RRF fusion)",
            sub_key, old_count, required_depth,
        )

        # Delete old checkpoint so it's fully re-retrieved (not appended)
        if old_path.exists():
            old_path.unlink()

        retrieved, latencies_ms = self.retrieve_for_backend(
            sub_backend, questions, question_ids,
            popularities=popularities,
            loaded_services=loaded_services,
            override_top_k=required_depth,
        )

        save_retrieved_docs_csv(retrieved, question_ids, old_path)
        logger.info(
            "✓ [%s] Upgraded CSV saved (%d questions × %d docs) → %s",
            sub_key, len(retrieved), required_depth, old_path,
        )

        latency_path = self._output_folder / f"latency_retrieval_{sub_key}.json"
        save_latency(
            path=latency_path,
            backend_key=sub_key,
            stage="retrieval",
            question_ids=question_ids,
            latencies_ms=latencies_ms,
        )
        self._shared_precomputed_docs[sub_key] = retrieved
        self._shared_precomputed_depths[sub_key] = required_depth
        return retrieved

    def _release_precomputed_for_backend(self, backend_key: str) -> None:
        """Release shared pre-computed data once no remaining backend needs it."""
        precomputed = self._precomputed_cache.pop(backend_key, None)
        if precomputed is None:
            return

        freed_any = False
        for sub_key in precomputed:
            remaining = self._shared_precomputed_refcounts.get(sub_key, 0) - 1
            if remaining > 0:
                self._shared_precomputed_refcounts[sub_key] = remaining
                continue
            self._shared_precomputed_refcounts.pop(sub_key, None)
            if self._shared_precomputed_docs.pop(sub_key, None) is not None:
                freed_any = True
            self._shared_precomputed_depths.pop(sub_key, None)

        if freed_any:
            gc.collect()

    def _prepare_precomputed(
        self,
        composite_backends: list[RetrievalBackend],
        questions: list[str],
        question_ids: list[str],
        popularities: list[float],
        backend_by_key: dict[str, RetrievalBackend],
        loaded_services: dict[str, object],
    ) -> set[str]:
        """Check and prepare pre-computed results for composite backends.

        For each composite backend, tries to load sub-backend CSVs. If CSVs
        exist with sufficient depth, stores them in ``self._precomputed_cache``.
        If depth is insufficient, auto-upgrades the sub-backend CSV. If CSVs
        are missing entirely, adds the sub-keys to a set for live loading.

        Args:
            composite_backends: Composite backends that will run.
            questions: All question texts.
            question_ids: All question IDs.
            popularities: Per-question popularity scores.
            backend_by_key: Mapping from backend key to declaration.
            loaded_services: Cache of loaded services.

        Returns:
            Set of sub-backend keys that need live loading (no CSVs available).
        """
        needs_live: set[str] = set()
        progress = tqdm(
            composite_backends,
            total=len(composite_backends),
            desc="Preparing composite caches",
            unit="backend",
            dynamic_ncols=True,
        )

        for cb in progress:
            progress.set_postfix_str(cb.key)
            required_depth = self._compute_required_depth(cb)
            precomputed, insufficient = self._try_load_precomputed(
                cb.router_sub_keys, question_ids, required_depth,
            )

            if precomputed is None:
                # CSVs missing — need live sub-backends
                needs_live.update(cb.router_sub_keys)
                continue

            # Auto-upgrade insufficient sub-backends
            for sub_key in insufficient:
                upgraded = self._upgrade_sub_backend(
                    sub_key, required_depth,
                    questions, question_ids, popularities,
                    backend_by_key, loaded_services,
                )
                precomputed[sub_key] = upgraded

            self._precomputed_cache[cb.key] = precomputed
            for sub_key in precomputed:
                self._shared_precomputed_refcounts[sub_key] = (
                    self._shared_precomputed_refcounts.get(sub_key, 0) + 1
                )
            logger.info(
                "⚡ [%s] Pre-computed sub-backend results ready (depth=%d) — "
                "will skip live %s retrieval",
                cb.key, required_depth, list(cb.router_sub_keys),
            )

        progress.close()
        return needs_live

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
        
        # ── Load questions: either from local file or HuggingFace ─────────
        if self.cfg.qa_file is not None:
            # Load from local parquet file
            logger.info("Loading questions from local file: %s", self.cfg.qa_file)
            question_data = self._load_from_local_parquet(self.cfg.qa_file)
            logger.info("Loaded %d questions from local file", len(question_data))
        else:
            # Original HuggingFace loading path
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

        # ── Determine which composite backends will run ─────────────────────
        backend_by_key = {b.key: b for b in self.cfg.backends}
        composite_backends_to_run = [
            b for b in self.cfg.backends
            if b.type in ("router", "hybrid_faiss", "neural_router")
            and (not self.cfg.only_keys or b.key in self.cfg.only_keys)
        ]

        # ── Prepare pre-computed sub-backend results (auto-upgrade if needed) ──
        needs_live = self._prepare_precomputed(
            composite_backends_to_run,
            questions, question_ids, popularities,
            backend_by_key, loaded_services,
        )

        # ── Pre-load live sub-backends for composite backends without CSVs ──
        for sub_key in needs_live:
            if sub_key in loaded_services:
                continue
            sub_backend = backend_by_key.get(sub_key)
            if sub_backend is None or sub_backend.type in ("zero_shot", "router", "hybrid_faiss", "neural_router"):
                continue
            logger.info("⚙ [%s] Pre-loading sub-backend for router…", sub_key)
            try:
                loaded_services[sub_key] = self._load_service(sub_backend, loaded_services)
            except Exception as e:
                logger.error("[%s] Failed to pre-load router sub-backend: %s", sub_key, e, exc_info=True)

        total_backends = len(self.cfg.backends)
        progress = tqdm(
            self.cfg.backends,
            total=total_backends,
            desc="Stage 1 backends",
            unit="backend",
            dynamic_ncols=True,
        )

        for backend in progress:
            progress.set_postfix_str(backend.key)
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
                    self._release_precomputed_for_backend(backend.key)
                    # Ensure service is cached so a later router backend can reference it
                    if backend.type not in ("zero_shot", "router", "hybrid_faiss", "neural_router") and backend.key not in loaded_services:
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
            if backend.type not in ("zero_shot", "router", "hybrid_faiss", "neural_router") and backend.key not in loaded_services:
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
            self._release_precomputed_for_backend(backend.key)

        progress.close()

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
        p.add_argument("--datasets", nargs="+", default=None,
                       help="QA dataset names to load from HuggingFace (default: all 6 datasets).")
        p.add_argument("--qa-file", type=Path, default=None,
                       help="Path to local QA parquet file (skips HuggingFace download).")
        args = p.parse_args(argv)

        cfg = RetrievalConfig(
            collection_name=args.collection,
            output_dir=args.output_dir,
            top_k=args.top_k,
            questions_per_decile=args.questions_per_decile,
            restart=args.restart,
            restart_keys=tuple(args.restart_keys),
            only_keys=tuple(args.only_keys),
            dataset_names=tuple(args.datasets) if args.datasets else _d.dataset_names,
            qa_file=args.qa_file,
        )
        cls(cfg).run()


if __name__ == "__main__":
    RetrievalRunner.main()
