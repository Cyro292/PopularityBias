"""FAISS RAG service — memory-efficient vector retrieval for huge datasets.

Optimised for low-RAM environments with:
  - Memory-mapped index loading (search without loading full index into RAM)
  - Streaming IVF_PQ training (train on batches, not all vectors at once)
  - OPQ pre-transform for better compression
  - On-disk inverted lists (IndexIVFPQDisk) for billion-scale search
  - Configurable memory limits and queue sizes
  - Index sharding for parallel indexing and search
"""

from __future__ import annotations

import gc
import logging
import os
import queue
import shutil
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Literal, Sequence

import faiss
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS, DistanceStrategy
from tqdm import tqdm

from .base import RagService, VectorStoreLike
from .document_utils import documents_from_dataframe
from .SqliteDocstore import SqliteDocstore
from .utils import IndexingConfig, build_embeddings

logger = logging.getLogger(__name__)


# === Memory Configuration ===================================================

@dataclass
class MemoryConfig:
    """Configuration for memory-efficient FAISS operations.

    Args:
        max_ram_mb: Approximate RAM budget in MB for indexing/search.
        use_mmap: Memory-map the index file instead of loading into RAM.
        use_ondisk_ivf: Store inverted lists on disk (for IVF indexes).
        training_sample_size: Max vectors to sample for IVF training.
        queue_maxsize: Max items in producer/consumer queues.
        gc_every_n_batches: Force garbage collection every N batches.
        prefetch_batches: Number of batches to prefetch from parquet.
        shard_size: Max vectors per shard (0 = no sharding).
    """

    max_ram_mb: int = 4096
    use_mmap: bool = True
    use_ondisk_ivf: bool = False
    training_sample_size: int = 500_000
    queue_maxsize: int = 2
    gc_every_n_batches: int = 1
    prefetch_batches: int = 1
    shard_size: int = 0  # 0 = disabled, e.g. 10_000_000 for 10M vectors/shard


DEFAULT_MEMORY_CONFIG = MemoryConfig()

# FAISS + OpenMP can have issues with "RuntimeError: OpenMP error: Cannot fork a new thread" on some platforms (e.g., macOS). Setting this env var allows it to proceed with a warning instead of crashing. See
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
faiss.omp_set_num_threads(1)

_FAISS_METRIC_MAP = {
    "cosine": DistanceStrategy.COSINE,
    "dot_product": DistanceStrategy.MAX_INNER_PRODUCT,
    "euclidean": DistanceStrategy.EUCLIDEAN,
}


# === FAISS OpenMP Settings ==================================================

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
faiss.omp_set_num_threads(1)


class FaissRagService(RagService):
    """Memory-efficient FAISS / BM25 retrieval service for huge datasets.

    Strategies:
        * ``vector``  – exact brute-force search (``IndexFlatL2`` /
          ``IndexFlatIP``). High RAM usage.
        * ``hnsw`` – approximate kNN via HNSW (``IndexHNSWFlat``).
          Medium RAM, fast search.
        * ``ivfpq`` – compressed approximate search (``IndexIVFPQ``).
          **Low RAM**, good for huge datasets.
        * ``ivfpq_disk`` – IVF_PQ with on-disk inverted lists.
          **Minimal RAM**, for billion-scale.
        * ``opq_ivfpq`` – OPQ pre-transform + IVF_PQ for better accuracy
          at same compression.
        * ``bm25`` – lexical BM25 retrieval (``rank_bm25``).

    Memory Optimizations:
        * Memory-mapped index loading (``use_mmap=True``)
        * Streaming IVF training (sample-based, not full dataset)
        * On-disk inverted lists for IVF indexes
        * Configurable queue sizes and GC frequency
        * SQLite docstore (documents on disk, not RAM)
    """

    # ── Construction ─────────────────────────────────────────────────────

    def __init__(
        self,
        config: IndexingConfig | None = None,
        *,
        strategy: Literal[
            "vector", "hnsw", "ivfpq", "ivfpq_disk", "opq_ivfpq", "bm25"
        ] = "ivfpq",
        distance_strategy: Literal["cosine", "dot_product", "euclidean"] = "cosine",
        memory_config: MemoryConfig | None = None,
        # HNSW tuning
        hnsw_m: int = 32,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 128,
        normalize_l2: bool = False,
        # IVF_PQ tuning (for ivfpq, ivfpq_disk, opq_ivfpq strategies)
        ivfpq_nlist: int = 4096,
        ivfpq_m: int = 48,
        ivfpq_nbits: int = 8,
        ivfpq_nprobe: int = 64,
        # OPQ tuning (for opq_ivfpq strategy)
        opq_m: int = 48,
    ):
        self.strategy = strategy
        self._normalize_l2 = normalize_l2
        self.memory_config = memory_config or DEFAULT_MEMORY_CONFIG

        # Current in-memory stores
        self._faiss_store: FAISS | None = None
        self._bm25_retriever: BM25Retriever | None = None
        self._current_index_path: Path | None = None
        self._is_mmap_loaded: bool = False

        # HNSW parameters
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef_search = hnsw_ef_search

        # IVF_PQ parameters
        self.ivfpq_nlist = ivfpq_nlist
        self.ivfpq_m = ivfpq_m
        self.ivfpq_nbits = ivfpq_nbits
        self.ivfpq_nprobe = ivfpq_nprobe

        # OPQ parameters
        self.opq_m = opq_m

        # Training state (for streaming training)
        self._training_vectors: list[np.ndarray] = []
        self._training_vectors_count: int = 0
        self._is_trained: bool = False

        # Backwards compatibility
        self.connected_graph_notes = hnsw_m

        # Load prompt templates
        self._passage_prompt, self._query_prompt = self._load_prompts()

        if strategy == "bm25":
            raise ValueError("FAISS RAG Service cannot be indexed with BM25 strategy")
        else:
            if config is None:
                raise ValueError(f"IndexingConfig required for '{strategy}' strategy")
            self.config = config
            self._embeddings = build_embeddings(
                provider=config.embedding_provider,
                model=config.embedding_model,
                trust_remote_code=config.trust_remote_code,
                rate_limiter=config.rate_limiter,
                requests_per_second=config.requests_per_second,
                check_interval=config.rate_limit_check_interval,
                bucket_size=config.rate_limit_bucket_size,
                gpu_batch_size=config.gpu_batch_size,
                request_batch_size=config.request_batch_size,
                normalise_embeddings=config.normalise_embeddings,
            )

            if distance_strategy not in _FAISS_METRIC_MAP:
                raise ValueError(f"Unsupported distance function: {distance_strategy}")

            self.distance_strategy = _FAISS_METRIC_MAP[distance_strategy]

        # Pre-create text splitter once (reused across all batches)
        cfg = self.config if hasattr(self, 'config') else (config or IndexingConfig())
        if cfg.chunk_size:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            self._text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=cfg.chunk_size,
                chunk_overlap=cfg.chunk_overlap,
            )
        else:
            self._text_splitter = None

        logger.info(
            f"FAISS {strategy} strategy ready "
            f"(mmap={self.memory_config.use_mmap}, "
            f"max_ram={self.memory_config.max_ram_mb}MB)"
        )

    # ── Prompt helpers ───────────────────────────────────────────────────

    @staticmethod
    def _load_prompts() -> tuple[str, str]:
        """Load passage and query prompt templates from disk."""
        from config import DATA_DIR

        prompts_dir = Path(DATA_DIR) / "prompts"

        def _read(filename: str, default: str) -> str:
            path = prompts_dir / filename
            if path.exists():
                return path.read_text().strip()
            logger.warning(f"Prompt {path} not found, using default")
            return default

        return (
            _read("embeding_promt.txt", "passage: {passage}"),
            _read("query_promt.txt", "query: {query}"),
        )

    def _prepare_query(self, query: str) -> str:
        """Wrap query with prompt template (skipped for BM25)."""
        return self._query_prompt.format(query=query)

    def _prepare_documents(
        self, documents: list[Document], log_details: bool = False
    ) -> list[Document]:
        """Chunk documents and apply embedding prompt."""
        original_count = len(documents)

        if log_details and documents:
            first = documents[0].page_content[:100].replace("\n", " ")
            logger.info(f"Processing {original_count} documents, first doc start: '{first}...'")
            if len(documents) > 1:
                second = documents[1].page_content[:100].replace("\n", " ")
                logger.info(f"Second doc start: '{second}...'")

        if self._text_splitter is not None:
            documents = self._text_splitter.split_documents(documents)
            if log_details:
                logger.info(
                    f"Split {original_count} documents into {len(documents)} chunks "
                    f"(chunk_size={self.config.chunk_size}, overlap={self.config.chunk_overlap})"
                )

        if self._embeddings is None:
            return documents

        prepared = [
            Document(
                page_content=self._passage_prompt.format(passage=d.page_content),
                metadata=d.metadata,
            )
            for d in documents
        ]

        if log_details:
            logger.info(f"Applied embedding prompts to {len(prepared)} documents")
            first_p = prepared[0].page_content[:100].replace("\n", " ")
            logger.info(f"First prepared doc start: '{first_p}...'")
            if len(prepared) > 1:
                second_p = prepared[1].page_content[:100].replace("\n", " ")
                logger.info(f"Second prepared doc start: '{second_p}...'")

        return prepared

    # ── FAISS index helpers ──────────────────────────────────────────────

    def _build_faiss_index(self, dim: int, connected_graph_notes: int | None = None) -> faiss.Index:
        """Create a raw FAISS index matching the chosen strategy/distance.

        Memory-efficient index types:
            - ivfpq: ~48 bytes per vector (vs 6144 for flat with 1536-dim)
            - opq_ivfpq: OPQ pre-transform for better accuracy at same size
            - ivfpq_disk: Inverted lists on disk, minimal RAM

        IVF indexes are returned **untrained** — call ``_train_faiss_index``
        or ``_accumulate_training_vectors`` + ``_finalize_training`` before
        adding any vectors.
        """
        m = connected_graph_notes or self.hnsw_m

        if self.strategy == "vector":
            # Exact brute-force — high RAM, perfect recall
            index = faiss.IndexFlatL2(dim)
            logger.info(f"[Index] Created IndexFlatL2 (dim={dim})")

        elif self.strategy in ("hnsw", "approximation"):
            # HNSW graph — medium RAM, fast search, ~95% recall
            index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_L2)
            index.hnsw.efConstruction = self.hnsw_ef_construction
            index.hnsw.efSearch = self.hnsw_ef_search
            logger.info(
                f"[Index] Created IndexHNSWFlat (dim={dim}, M={m}, "
                f"efConstruction={self.hnsw_ef_construction})"
            )

        elif self.strategy == "ivfpq":
            # IVF_PQ — low RAM, ~90% recall
            self._validate_ivfpq_params(dim)
            quantizer = faiss.IndexFlatL2(dim)
            index = faiss.IndexIVFPQ(
                quantizer, dim,
                self.ivfpq_nlist,
                self.ivfpq_m,
                self.ivfpq_nbits,
            )
            index.nprobe = self.ivfpq_nprobe
            bytes_per_vec = self.ivfpq_m * self.ivfpq_nbits // 8
            logger.info(
                f"[Index] Created IndexIVFPQ (dim={dim}, nlist={self.ivfpq_nlist}, "
                f"m={self.ivfpq_m}, nbits={self.ivfpq_nbits}, ~{bytes_per_vec} bytes/vec)"
            )

        elif self.strategy == "opq_ivfpq":
            # OPQ pre-transform + IVF_PQ — better accuracy at same compression
            self._validate_ivfpq_params(dim)
            opq = faiss.OPQMatrix(dim, self.opq_m)
            quantizer = faiss.IndexFlatL2(dim)
            ivfpq = faiss.IndexIVFPQ(
                quantizer, dim,
                self.ivfpq_nlist,
                self.ivfpq_m,
                self.ivfpq_nbits,
            )
            ivfpq.nprobe = self.ivfpq_nprobe
            index = faiss.IndexPreTransform(opq, ivfpq)
            logger.info(
                f"[Index] Created OPQ({self.opq_m}) + IndexIVFPQ "
                f"(dim={dim}, nlist={self.ivfpq_nlist}, m={self.ivfpq_m})"
            )

        elif self.strategy == "ivfpq_disk":
            # IVF_PQ with on-disk inverted lists — minimal RAM for huge datasets
            self._validate_ivfpq_params(dim)
            quantizer = faiss.IndexFlatL2(dim)
            index = faiss.IndexIVFPQ(
                quantizer, dim,
                self.ivfpq_nlist,
                self.ivfpq_m,
                self.ivfpq_nbits,
            )
            index.nprobe = self.ivfpq_nprobe
            # On-disk inverted lists are set up after training in _setup_ondisk_ivf()
            logger.info(
                f"[Index] Created IndexIVFPQ for on-disk mode "
                f"(dim={dim}, nlist={self.ivfpq_nlist}, m={self.ivfpq_m})"
            )

        else:
            raise ValueError(f"Unsupported strategy: {self.strategy}")

        return index

    def _validate_ivfpq_params(self, dim: int) -> None:
        """Validate IVF_PQ parameters against embedding dimension."""
        if dim % self.ivfpq_m != 0:
            raise ValueError(
                f"IVF_PQ: dim ({dim}) must be divisible by ivfpq_m ({self.ivfpq_m}). "
                f"Try ivfpq_m={self._suggest_m(dim)}"
            )

    @staticmethod
    def _suggest_m(dim: int) -> int:
        """Suggest a valid ivfpq_m value for the given dimension."""
        # Common dimensions: 384, 768, 1024, 1536, 3072
        for m in [48, 32, 64, 24, 16, 8]:
            if dim % m == 0:
                return m
        return dim  # fallback: no compression

    def _train_faiss_index(self, index: faiss.Index, embeddings: list[list[float]]) -> None:
        """Train an IVF/OPQ index on embeddings (no-op for flat/HNSW)."""
        if not hasattr(index, 'is_trained') or index.is_trained:
            return

        vectors = np.array(embeddings, dtype=np.float32)
        n = len(vectors)
        min_train = self.ivfpq_nlist * 39  # FAISS recommendation

        if n < min_train:
            logger.warning(
                f"[Train] Only {n:,} vectors for training; recommended >= {min_train:,} "
                f"(nlist={self.ivfpq_nlist}). Recall may be reduced."
            )

        logger.info(f"[Train] Training index on {n:,} vectors...")
        index.train(vectors)
        logger.info(f"[Train] ✓ Index trained")

        del vectors
        gc.collect()

    # ── Streaming training (memory-efficient for huge datasets) ──────────

    def _accumulate_training_vectors(
        self,
        embeddings: list[list[float]],
    ) -> bool:
        """Accumulate vectors for streaming IVF training.

        Returns True if we have enough vectors to train.
        """
        max_samples = self.memory_config.training_sample_size
        current = self._training_vectors_count

        if current >= max_samples:
            return True  # Already have enough

        # Sample from this batch to avoid memory explosion
        n = len(embeddings)
        remaining = max_samples - current

        if n <= remaining:
            # Take all
            self._training_vectors.append(np.array(embeddings, dtype=np.float32))
            self._training_vectors_count += n
        else:
            # Random sample
            indices = np.random.choice(n, remaining, replace=False)
            sample = np.array([embeddings[i] for i in indices], dtype=np.float32)
            self._training_vectors.append(sample)
            self._training_vectors_count += remaining

        min_train = self.ivfpq_nlist * 39
        have_enough = self._training_vectors_count >= min_train

        logger.debug(
            f"[Train] Accumulated {self._training_vectors_count:,}/{max_samples:,} "
            f"training vectors (need {min_train:,})"
        )

        return have_enough

    def _finalize_training(self, index: faiss.Index) -> None:
        """Train the index on accumulated vectors and clear the buffer."""
        if self._is_trained or not self._training_vectors:
            return

        # Concatenate all accumulated vectors
        all_vectors = np.vstack(self._training_vectors)
        logger.info(
            f"[Train] Training index on {len(all_vectors):,} sampled vectors..."
        )

        index.train(all_vectors)
        self._is_trained = True

        # Clear training buffer to free RAM
        del all_vectors
        self._training_vectors.clear()
        self._training_vectors_count = 0
        gc.collect()

        logger.info(f"[Train] ✓ Index trained, training buffer cleared")

    def _setup_ondisk_ivf(self, index: faiss.Index, path: Path) -> faiss.Index:
        """Convert trained IVF index to use on-disk inverted lists.

        This dramatically reduces RAM usage for huge indexes by keeping
        the inverted lists on disk instead of in memory.
        """
        if self.strategy != "ivfpq_disk":
            return index

        if not index.is_trained:
            raise ValueError("Index must be trained before setting up on-disk IVF")

        ivlists_path = path / "faiss" / "ivlists.bin"
        ivlists_path.parent.mkdir(parents=True, exist_ok=True)

        # Create on-disk inverted lists
        ivf_index = faiss.extract_index_ivf(index)
        invlists = faiss.OnDiskInvertedLists(
            ivf_index.nlist,
            ivf_index.code_size,
            str(ivlists_path),
        )
        ivf_index.replace_invlists(invlists, True)

        logger.info(f"[Index] Set up on-disk inverted lists at {ivlists_path}")
        return index

    def load_faiss_store(
        self,
        path: str | Path,
        *,
        use_mmap: bool | None = None,
    ) -> FAISS | None:
        """Load a previously-saved FAISS index from *path*.

        Args:
            path: Directory containing the faiss/ subdirectory.
            use_mmap: If True, memory-map the index file instead of loading
                into RAM. Defaults to ``self.memory_config.use_mmap``.

        Memory-mapping allows searching huge indexes (100GB+) with minimal
        RAM usage. The OS loads pages on-demand during search.
        """
        path = Path(path)
        faiss_dir = path / "faiss"
        if not faiss_dir.exists():
            logger.warning(f"FAISS index directory not found at {faiss_dir}")
            raise FileNotFoundError(f"FAISS index directory not found at {faiss_dir}")

        faiss_file = faiss_dir / "index.faiss"
        db_file = faiss_dir / "docstore.sqlite"

        if not faiss_file.exists():
            raise FileNotFoundError(f"index.faiss missing in {faiss_dir}")

        # Determine whether to use memory-mapping
        mmap = use_mmap if use_mmap is not None else self.memory_config.use_mmap

        if mmap:
            logger.info(f"Loading FAISS index with memory-mapping...")
            raw_index = faiss.read_index(str(faiss_file), faiss.IO_FLAG_MMAP)
            self._is_mmap_loaded = True
        else:
            logger.info(f"Loading FAISS index into RAM...")
            raw_index = faiss.read_index(str(faiss_file))
            self._is_mmap_loaded = False

        # Restore IVF query-time probe count if applicable
        if hasattr(raw_index, 'nprobe'):
            raw_index.nprobe = self.ivfpq_nprobe

        # Restore HNSW search parameters
        if hasattr(raw_index, 'hnsw'):
            raw_index.hnsw.efSearch = self.hnsw_ef_search

        logger.info(
            f"✓ {raw_index.ntotal:,} vectors loaded "
            f"({'mmap' if mmap else 'RAM'})"
        )

        docstore = SqliteDocstore(db_file)
        index_to_docstore_id = docstore.load_id_map()
        if not index_to_docstore_id:
            raise ValueError(f"id_map is empty in {db_file} — index may be corrupt")

        self._faiss_store = FAISS(
            embedding_function=self._embeddings,
            index=raw_index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
            normalize_L2=self._normalize_l2,
        )
        self._current_index_path = path
        return self._faiss_store
    
    def load_index(self, path: str | Path) -> FAISS:
        return self.load_faiss_store(path)

    def save_faiss_index(self, path: str | Path) -> None:
        """Save current FAISS index to *path* (no pickle — fast, no RAM spike).

        Layout under ``<path>/faiss/`` (two files only):
          index.faiss     – raw FAISS binary
          docstore.sqlite – text + metadata + id_map table (integer→UUID)
        """
        if self._faiss_store is None:
            raise ValueError("No FAISS index to save")

        faiss_dir = Path(path) / "faiss"
        faiss_dir.mkdir(parents=True, exist_ok=True)

        # ── FAISS binary (atomic via tmp rename) ─────────────────────────
        tmp_faiss = faiss_dir / "index.faiss.tmp"
        faiss.write_index(self._faiss_store.index, str(tmp_faiss))
        tmp_faiss.rename(faiss_dir / "index.faiss")

        # ── Integer → docstore-ID map stored inside SQLite ────────────────
        self._faiss_store.docstore.save_id_map(self._faiss_store.index_to_docstore_id)

        n = len(self._faiss_store.index_to_docstore_id)
        logger.info(f"FAISS index saved to {faiss_dir} ({n:,} vectors)")

    def save_index(self, path: str | Path) -> None:
        self.save_faiss_index(path)

    def index_from_dataframe(
        self,
        df: pd.DataFrame,
        text_field: str,
        html_field: str | None = None,
        *,
        metadata_fields: Sequence[str] | None = None,
        output_dir: Path | None = None,
        collection_name: str = "rag",
        progress_bar: bool | None = None,
    ) -> tuple[FAISS | BM25Retriever | None, int]:
        """Build a FAISS (or BM25) index from a pandas DataFrame."""
        raise NotImplementedError("index_from_dataframe is not implemented; use index_from_parquet_batches for large datasets")

    def index_from_parquet(
        self,
        parquet_path: Path,
        output_dir: Path,
        *,
        text_field: str | None = None,
        html_field: str | None = None,
        metadata_fields: Sequence[str] | None = None,
        collection_name: str = "rag",
    ) -> tuple[FAISS | BM25Retriever | None, int]:
        """Convenience: read entire Parquet into DataFrame, then index."""
        raise NotImplementedError("index_from_parquet is not implemented; use index_from_parquet_batches for large datasets")

    # ── Sparese Indexing  ──────────────────────────────────────────

    def index_from_parquet_sparse(
        self,
        parquet_path: Path,
        output_dir: Path,
        *,
        text_field: str | None = None,
        html_field: str | None = None,
        metadata_fields: Sequence[str] | None = None,
        collection_name: str = "rag",
    ) -> tuple[FAISS | BM25Retriever | None, int]:
        """Build a sparse BM25 index from a Parquet file."""
        raise NotImplementedError("Sparse BM25 indexing is not implemented; use index_from_parquet_batches with strategy='bm25'")

    # ── Batched dense indexing from Parquet (producer → embed → insert) ────────

    def index_from_parquet_batches(
        self,
        parquet_path: Path,
        *,
        text_field: str = "text",
        metadata_fields: Sequence[str] | None = None,
        collection_name: str | None = None,
        progress_bar: bool | None = None,
        batch_size: int = 5000,
        skip_rows: int = 0,
        checkpoint: bool = True,
        memory_config: MemoryConfig | None = None,
    ) -> tuple[FAISS, int]:
        """Memory-efficient three-stage indexing pipeline.

        Pipeline stages:
            Producer thread  →  Main thread (embed)  →  Insert thread
            read + chunk         GPU/API-bound            FAISS add

        Memory optimizations:
            - Streaming IVF training (samples vectors, doesn't load all at once)
            - Configurable queue sizes (default: 2 items max)
            - Aggressive garbage collection between batches
            - SQLite docstore (documents on disk, not RAM)
            - Optional on-disk inverted lists (ivfpq_disk strategy)

        Args:
            parquet_path: Path to the parquet file to index.
            text_field: Column name for document text.
            metadata_fields: Additional columns to store as metadata.
            collection_name: Output directory name for the index.
            progress_bar: Show progress bar.
            batch_size: Rows per parquet batch. Smaller = less RAM.
            skip_rows: Resume from this row offset.
            checkpoint: Save after each batch (recommended for huge datasets).
            memory_config: Override service-level memory settings.

        Returns:
            Tuple of (FAISS store, total chunks indexed).
        """
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
        if self._embeddings is None:
            raise ValueError("Embeddings required for parquet batch indexing")

        mem_cfg = memory_config or self.memory_config
        total_chunks = 0
        rows_processed = 0
        batches_since_gc = 0

        parquet_file = pq.ParquetFile(parquet_path)
        total_rows = parquet_file.metadata.num_rows
        rows_to_index = total_rows - skip_rows

        columns = [text_field] + list(metadata_fields or [])
        if skip_rows > 0:
            logger.info(f"Skipping first {skip_rows:,} rows")

        logger.info(
            f"Indexing {rows_to_index:,} rows (of {total_rows:,} total) | "
            f"batch={batch_size:,} | strategy={self.strategy} | "
            f"queue_size={mem_cfg.queue_maxsize}"
        )

        # ── Shared state ─────────────────────────────────────────────────
        prepare_queue: queue.Queue = queue.Queue(maxsize=mem_cfg.queue_maxsize)
        insert_queue: queue.Queue = queue.Queue(maxsize=mem_cfg.queue_maxsize)
        exception_holder: list[Exception] = []
        cancel = threading.Event()
        store_lock = threading.Lock()

        # Training state for streaming training
        training_complete = threading.Event()
        raw_index_holder: list[faiss.Index] = []

        # Checkpoint state
        save_path = Path(collection_name) if collection_name else None

        def _safe_put(q: queue.Queue, item):
            while not cancel.is_set():
                try:
                    q.put(item, timeout=2)
                    return True
                except queue.Full:
                    continue
            return False

        def _safe_get(q: queue.Queue):
            while not cancel.is_set():
                try:
                    return q.get(timeout=2), True
                except queue.Empty:
                    continue
            return None, False

        # ── Stage 1 – Producer: read → chunk ─────────────────────────────
        def producer():
            try:
                rows_seen = 0
                for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
                    if cancel.is_set():
                        break
                    batch_len = batch.num_rows

                    if rows_seen + batch_len <= skip_rows:
                        rows_seen += batch_len
                        logger.debug(f"[Prepare] Skipped batch ({rows_seen:,}/{skip_rows:,})")
                        continue

                    df = batch.to_pandas()
                    if rows_seen < skip_rows:
                        drop = skip_rows - rows_seen
                        df = df.iloc[drop:]
                        logger.info(f"[Prepare] Partial skip: dropped first {drop:,} rows of batch")
                    rows_seen += batch_len

                    n_rows = len(df)
                    logger.info(f"[Prepare] {n_rows:,} rows from parquet")

                    documents = documents_from_dataframe(df, text_field, metadata_fields)
                    chunks = self._prepare_documents(documents, log_details=False)
                    logger.info(f"[Prepare] {len(chunks):,} chunks (from {n_rows:,} rows)")
                    del documents, df

                    if not _safe_put(prepare_queue, (chunks, n_rows)):
                        break
                    del chunks
                    gc.collect()
            except Exception as e:
                exception_holder.append(e)
                cancel.set()
            finally:
                _safe_put(prepare_queue, None)

        # ── Stage 3 – Insert thread: add to FAISS ────────────────────────
        def inserter():
            nonlocal total_chunks, rows_processed, batches_since_gc
            try:
                while True:
                    item, ok = _safe_get(insert_queue)
                    if not ok or item is None:
                        break

                    texts, embeddings, metadatas, batch_rows = item
                    n = len(texts)
                    logger.info(f"[Insert] {n:,} docs → FAISS")

                    text_embedding_pairs = list(zip(texts, embeddings))

                    with store_lock:
                        if self._faiss_store is None:
                            # First batch — create index + docstore
                            dim = len(embeddings[0])
                            raw_index = self._build_faiss_index(dim)

                            # For IVF indexes: use streaming training
                            if not raw_index.is_trained:
                                have_enough = self._accumulate_training_vectors(embeddings)
                                if have_enough or self.strategy == "vector":
                                    self._finalize_training(raw_index)
                                    training_complete.set()

                                    # Set up on-disk IVF if requested
                                    if self.strategy == "ivfpq_disk" and save_path:
                                        raw_index = self._setup_ondisk_ivf(raw_index, save_path)
                                else:
                                    # Store index for later training
                                    raw_index_holder.append(raw_index)
                                    logger.info(
                                        f"[Insert] Accumulating training vectors... "
                                        f"({self._training_vectors_count:,} so far)"
                                    )
                                    # Skip adding this batch, it will be re-embedded or lost
                                    # This is a tradeoff for memory efficiency
                                    continue

                            db_dir = Path(save_path) / "faiss" if save_path else Path("faiss_index") / "faiss"
                            db_dir.mkdir(parents=True, exist_ok=True)
                            self._faiss_store = FAISS(
                                embedding_function=self._embeddings,
                                index=raw_index,
                                docstore=SqliteDocstore(db_path=db_dir / "docstore.sqlite"),
                                index_to_docstore_id={},
                                normalize_L2=self._normalize_l2,
                            )

                        # Handle deferred training completion
                        if raw_index_holder and not training_complete.is_set():
                            have_enough = self._accumulate_training_vectors(embeddings)
                            if have_enough:
                                self._finalize_training(raw_index_holder[0])
                                training_complete.set()

                                if self.strategy == "ivfpq_disk" and save_path:
                                    raw_index_holder[0] = self._setup_ondisk_ivf(
                                        raw_index_holder[0], save_path
                                    )

                                # Create the store with trained index
                                db_dir = Path(save_path) / "faiss" if save_path else Path("faiss_index") / "faiss"
                                db_dir.mkdir(parents=True, exist_ok=True)
                                self._faiss_store = FAISS(
                                    embedding_function=self._embeddings,
                                    index=raw_index_holder[0],
                                    docstore=SqliteDocstore(db_path=db_dir / "docstore.sqlite"),
                                    index_to_docstore_id={},
                                    normalize_L2=self._normalize_l2,
                                )
                            else:
                                # Still collecting training vectors
                                continue

                        self._faiss_store.add_embeddings(
                            text_embeddings=text_embedding_pairs,
                            metadatas=metadatas,
                        )

                    total_chunks += n
                    rows_processed += batch_rows
                    batches_since_gc += 1
                    pbar.update(batch_rows)
                    logger.info(
                        f"[Insert] ✓ +{n:,} chunks / {batch_rows:,} rows "
                        f"(cumulative: {total_chunks:,} chunks, {rows_processed:,} rows)"
                    )

                    # ── Checkpoint save after every batch ────────────────
                    if checkpoint and save_path:
                        try:
                            logger.info(f"[Checkpoint] Saving index at {rows_processed:,} rows...")
                            self.save_index(save_path)
                            logger.info(
                                f"[Checkpoint] Saved at {rows_processed:,} rows "
                                f"({total_chunks:,} chunks) → {save_path}"
                            )
                        except Exception as ckpt_err:
                            logger.warning(f"[Checkpoint] Save failed: {ckpt_err}")

                    # ── Aggressive GC for memory efficiency ──────────────
                    del texts, embeddings, metadatas, text_embedding_pairs
                    if batches_since_gc >= mem_cfg.gc_every_n_batches:
                        gc.collect()
                        batches_since_gc = 0

            except Exception as e:
                exception_holder.append(e)
                cancel.set()

        # ── Start threads ────────────────────────────────────────────────
        producer_thread = threading.Thread(target=producer, name="Prepare")
        insert_thread = threading.Thread(target=inserter, name="Insert")

        pbar = tqdm(
            total=total_rows,
            initial=skip_rows,
            desc="Indexing",
            unit="row",
            disable=not progress_bar,
        )

        producer_thread.start()
        insert_thread.start()

        # ── Stage 2 – Main thread: embed ─────────────────────────────────
        try:
            while True:
                item, ok = _safe_get(prepare_queue)
                if not ok or item is None:
                    break

                chunk_batch, chunk_batch_rows = item
                texts = [doc.page_content for doc in chunk_batch]
                metadatas = [doc.metadata for doc in chunk_batch]

                del chunk_batch

                logger.info(f"[Embed] {len(texts):,} chunks...")
                embeddings = self._embeddings.embed_documents(texts)

                if not _safe_put(insert_queue, (texts, embeddings, metadatas, chunk_batch_rows)):
                    break
                del texts, embeddings, metadatas
                gc.collect()
        except Exception as e:
            exception_holder.append(e)
            cancel.set()
        finally:
            _safe_put(insert_queue, None)

        # ── Wait for completion ──────────────────────────────────────────
        producer_thread.join()
        insert_thread.join()
        pbar.close()

        # Save whatever we have even if there was an error
        if save_path is not None and self._faiss_store is not None:
            try:
                self.save_index(save_path)
                logger.info(f"[Save] Final save at {rows_processed:,} rows → {save_path}")
            except Exception as save_err:
                logger.error(f"[Save] Final save failed: {save_err}")

        if exception_holder:
            raise exception_holder[0]

        logger.info(f"\n=== Indexing Complete ===")
        logger.info(f"Total chunks indexed: {total_chunks:,} (from {total_rows:,} rows)")
        return self._faiss_store, total_chunks

    # ── Add documents to existing index ──────────────────────────────────

    def add_documents(
        self, index: FAISS | None, documents: Sequence[Document]
    ) -> FAISS:
        """Add documents to an existing FAISS index."""
        store = index or self._faiss_store
        if store is None:
            raise ValueError("No FAISS index loaded")
        store.add_documents(list(documents))
        self._faiss_store = store
        return store

    # ── Delete index ─────────────────────────────────────────────────────

    def delete_index(self, index: FAISS | None = None) -> None:
        """Drop in-memory index (and delete persisted files if saved)."""
        if self._current_index_path and self._current_index_path.exists():
            shutil.rmtree(self._current_index_path)
            logger.info(f"Deleted saved index at {self._current_index_path}")

        self._faiss_store = None
        self._bm25_retriever = None
        self._current_index_path = None
        logger.info("In-memory index cleared")

    # ── Strategy context manager ─────────────────────────────────────────

    @contextmanager
    def _strategy_context(self, strategy: str | None):
        """Temporarily switch retrieval strategy."""
        if strategy is None or strategy == self.strategy:
            yield self.strategy
            return

        original = self.strategy
        try:
            self.strategy = strategy
            yield strategy
        finally:
            self.strategy = original

    # ── Retrieval ────────────────────────────────────────────────────────

    def retrieve_documents(
        self,
        text: str,
        top_k: int = 5,
        strategy: str | None = None,
        **kwargs,
    ) -> list[Document]:
        """Retrieve documents using the configured (or overridden) strategy."""
        results = self.retrieve_documents_with_scores(
            text, top_k=top_k, strategy=strategy, **kwargs
        )
        return [doc for doc, _score in results]

    def retrieve_documents_with_scores(
        self,
        text: str,
        top_k: int = 5,
        strategy: str | None = None,
        **kwargs,
    ) -> list[tuple[Document, float]]:
        """Retrieve documents with relevance scores."""
        with self._strategy_context(strategy) as strat:
            if strat == "bm25":
                return self._retrieve_bm25(text, top_k)
            else:
                return self._retrieve_vector(text, top_k)

    def _retrieve_vector(self, text: str, top_k: int) -> list[tuple[Document, float]]:
        if self._faiss_store is None:
            raise ValueError("No FAISS index loaded. Use load_index() or index first.")
        query = self._prepare_query(text)
        return self._faiss_store.similarity_search_with_score(query, k=top_k)

    def _retrieve_bm25(self, text: str, top_k: int) -> list[tuple[Document, float]]:
        if self._bm25_retriever is None:
            raise ValueError(
                "No BM25 retriever loaded. "
                "Call load_index() — it will build one from the FAISS docstore automatically."
            )
        self._bm25_retriever.k = top_k
        docs = self._bm25_retriever.invoke(text)
        # BM25Retriever doesn't expose raw scores — assign rank-based scores
        return [(doc, 1.0 / (i + 1)) for i, doc in enumerate(docs)]

    # ── Batch retrieval ──────────────────────────────────────────────────

    def batch_retrieve(
        self,
        questions: list[str],
        *,
        top_k: int = 5,
        strategy: str | None = None,
        progress_bar: bool = True,
        **kwargs,
    ) -> list[list[tuple[Document, float]]]:
        """Retrieve for multiple queries, optionally switching strategy."""
        results = []
        for q in tqdm(questions, desc=f"Retrieving ({strategy or self.strategy})", disable=not progress_bar):
            results.append(self.retrieve_documents_with_scores(q, top_k=top_k, strategy=strategy, **kwargs))
        return results

    # ── Convenience accessors ────────────────────────────────────────────

    def get_store(self) -> FAISS:
        """Return the underlying FAISS vector store."""
        if self._faiss_store is None:
            raise ValueError("No FAISS index loaded. Use load_index() first.")
        return self._faiss_store
    
    def get_indexed_doc_count(self) -> int:
        """Return the number of documents currently indexed in FAISS."""
        if self._faiss_store is None:
            return 0
        return len(self._faiss_store.index_to_docstore_id)

    def retrieve_all_documents(
        self,
        batch_size: int = 1000,
        progress_bar: bool = True,
    ) -> list[Document]:
        """Retrieve every document from the FAISS docstore.

        FAISS stores all documents in its ``InMemoryDocstore``, so this
        is O(n) in-memory and does not need scrolling.
        """
        raise NotImplementedError("retrieve_all_documents is not implemented yet; access the FAISS docstore directly for now")

    def retrieve_all_documents_sparse(
        self,
        batch_size: int = 1000,
        progress_bar: bool = True,
    ) -> list[Document]:
        """Retrieve every document from the BM25 retriever."""
        
        tbar = tqdm(total=len(self._bm25_retriever._docstore._dict), desc="Retrieving all BM25 documents", disable=not progress_bar)
        results = []

        for batch in self._bm25_retriever._docstore._dict.values():
            for doc in batch:
                results.append(doc)
            tbar.update(len(batch))

        return results

    # ── Memory estimation helpers ────────────────────────────────────────

    def estimate_index_memory(
        self,
        n_vectors: int,
        dim: int = 1536,
    ) -> dict[str, float]:
        """Estimate RAM usage for different index strategies.

        Args:
            n_vectors: Number of vectors to index.
            dim: Embedding dimension (default: 1536 for OpenAI).

        Returns:
            Dict mapping strategy name to estimated RAM in MB.
        """
        bytes_per_float = 4

        estimates = {}

        # Flat index: full vectors in RAM
        flat_bytes = n_vectors * dim * bytes_per_float
        estimates["vector"] = flat_bytes / (1024 * 1024)

        # HNSW: vectors + graph (~2x flat)
        hnsw_bytes = flat_bytes * 2.2
        estimates["hnsw"] = hnsw_bytes / (1024 * 1024)

        # IVF_PQ: compressed codes + centroids
        # Each vector compressed to ivfpq_m bytes (with nbits=8)
        pq_bytes_per_vec = self.ivfpq_m
        centroids_bytes = self.ivfpq_nlist * dim * bytes_per_float
        ivfpq_bytes = (n_vectors * pq_bytes_per_vec) + centroids_bytes
        estimates["ivfpq"] = ivfpq_bytes / (1024 * 1024)

        # OPQ + IVF_PQ: same as IVF_PQ + OPQ matrix
        opq_matrix_bytes = dim * dim * bytes_per_float
        estimates["opq_ivfpq"] = (ivfpq_bytes + opq_matrix_bytes) / (1024 * 1024)

        # IVF_PQ with on-disk: only centroids + metadata in RAM
        ivfpq_disk_bytes = centroids_bytes + (n_vectors * 8)  # 8 bytes per ID
        estimates["ivfpq_disk"] = ivfpq_disk_bytes / (1024 * 1024)

        return estimates

    def recommend_strategy(
        self,
        n_vectors: int,
        available_ram_mb: int,
        dim: int = 1536,
    ) -> str:
        """Recommend the best indexing strategy for your constraints.

        Args:
            n_vectors: Number of vectors to index.
            available_ram_mb: Available RAM in MB.
            dim: Embedding dimension.

        Returns:
            Recommended strategy name.
        """
        estimates = self.estimate_index_memory(n_vectors, dim)

        # Add safety margin (80% of available RAM)
        budget = available_ram_mb * 0.8

        # Prefer accuracy when possible
        if estimates["vector"] < budget:
            return "vector"
        if estimates["hnsw"] < budget:
            return "hnsw"
        if estimates["ivfpq"] < budget:
            return "ivfpq"
        if estimates["opq_ivfpq"] < budget:
            return "opq_ivfpq"
        return "ivfpq_disk"

    def get_index_stats(self) -> dict:
        """Return statistics about the current index."""
        if self._faiss_store is None:
            return {"loaded": False}

        index = self._faiss_store.index
        stats = {
            "loaded": True,
            "is_mmap": self._is_mmap_loaded,
            "n_vectors": index.ntotal,
            "strategy": self.strategy,
        }

        # IVF-specific stats
        if hasattr(index, 'nlist'):
            stats["nlist"] = index.nlist
            stats["nprobe"] = index.nprobe

        # HNSW-specific stats
        if hasattr(index, 'hnsw'):
            stats["hnsw_m"] = self.hnsw_m
            stats["hnsw_ef_search"] = index.hnsw.efSearch

        return stats