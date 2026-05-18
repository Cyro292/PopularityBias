"""FAISS RAG service — memory-efficient local vector retrieval.

Optimised for large datasets with configurable memory constraints:

Index strategies
----------------
``vector``      Exact brute-force (``IndexFlatL2``).  Perfect recall, high RAM.
``hnsw``        Approximate kNN via HNSW graph.  Medium RAM, fast search.
``ivfpq``       Compressed IVF_PQ.  Low RAM, ~90 % recall.
``opq_ivfpq``   OPQ pre-transform + IVF_PQ.  Better accuracy, same size.
``ivfpq_disk``  IVF_PQ with on-disk inverted lists.  Minimal RAM.

Memory optimisations
--------------------
- Memory-mapped index loading (search without loading into RAM).
- Streaming IVF training (sample-based, not full dataset).
- On-disk inverted lists for IVF indexes (``ivfpq_disk``).
- SQLite docstore — documents stay on disk, not in RAM.
- Configurable queue sizes and GC frequency in the indexing pipeline.
- Index sharding (optional, for parallel indexing).

Unified interface
-----------------
Implements the full ``RagService`` contract:
``index_from_dataframe``, ``index_from_parquet``,
``index_from_parquet_batches``, ``load_index``, ``save_index``,
``delete_index``, ``retrieve_documents``,
``retrieve_documents_with_scores``, ``batch_retrieve``,
``batch_retrieve_with_scores``, ``get_doc_count``,
``get_all_documents``, ``get_index_stats``,
``embed_prompt``, ``embed_passage``.
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
from typing import Any, Iterator, Literal, Sequence

import faiss
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from langchain.schema import Document
from langchain_community.vectorstores import FAISS, DistanceStrategy
from tqdm import tqdm

from .base import IndexResult, RagService, documents_from_dataframe
from src.storage.sqlite_docstore import SqliteDocstore
from .utils import IndexingConfig, build_embeddings

logger = logging.getLogger(__name__)

# FAISS + OpenMP can trigger "Cannot fork a new thread" on macOS.
# Setting this allows it to proceed with a warning rather than crashing.
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
faiss.omp_set_num_threads(1)

_FAISS_METRIC_MAP: dict[str, DistanceStrategy] = {
    "cosine":      DistanceStrategy.COSINE,
    "dot_product": DistanceStrategy.MAX_INNER_PRODUCT,
    "euclidean":   DistanceStrategy.EUCLIDEAN,
}


# ── Memory configuration ──────────────────────────────────────────────────────

@dataclass
class MemoryConfig:
    """Tuning knobs for memory-efficient FAISS operations.

    Attributes:
        max_ram_mb: Approximate RAM budget in MB for indexing/search.
        use_mmap: Memory-map the index file instead of loading into RAM.
        use_ondisk_ivf: Store inverted lists on disk (IVF indexes only).
        training_sample_size: Max vectors sampled for IVF training.
        queue_maxsize: Max items held in producer/consumer queues.
        gc_every_n_batches: Force GC every N indexing batches.
        prefetch_batches: Parquet batches to read ahead.
        shard_size: Max vectors per shard (0 = disabled).
    """

    max_ram_mb: int = 4_096
    use_mmap: bool = True
    use_ondisk_ivf: bool = False
    training_sample_size: int = 500_000
    queue_maxsize: int = 2
    gc_every_n_batches: int = 1
    prefetch_batches: int = 1
    shard_size: int = 0


DEFAULT_MEMORY_CONFIG = MemoryConfig()


# ── FAISS RAG service ─────────────────────────────────────────────────────────

class FaissRagService(RagService):
    """Memory-efficient FAISS vector retrieval service.

    Args:
        config: ``IndexingConfig`` specifying the embedding provider and model.
            Required for all vector strategies.
        strategy: Index strategy — see module docstring for options.
        distance_strategy: Vector distance metric (``"cosine"``,
            ``"dot_product"``, or ``"euclidean"``).
        memory_config: Memory tuning.  Defaults to ``MemoryConfig()``.
        hnsw_m: HNSW graph connectivity parameter.
        hnsw_ef_construction: HNSW build-time search width.
        hnsw_ef_search: HNSW query-time search width.
        normalize_l2: Apply L2 normalisation inside the FAISS store.
        ivfpq_nlist: IVF number of Voronoi cells.
        ivfpq_m: Number of sub-quantisers (must divide embedding dim).
        ivfpq_nbits: Bits per sub-quantiser code (usually 8).
        ivfpq_nprobe: Cells visited per query (higher = better recall).
        opq_m: OPQ rotation matrix sub-dimension.

    Example::

        config = IndexingConfig(
            embedding_provider="modal",
            embedding_model="Lajavaness/bilingual-embedding-small",
            gpu_batch_size=512,
            request_batch_size=2048,
            normalise_embeddings=True,
        )
        service = FaissRagService(config=config, strategy="ivfpq")
        service.index_from_parquet_batches(
            Path("data/wiki_full_bil/wiki_corpus.parquet"),
            text_field="text",
            metadata_fields=["wikipedia_id", "wikipedia_title"],
            collection_name="data/faiss_wiki",
        )
        docs = service.retrieve_documents("Who is Reza Pahlavi?", top_k=10)
    """

    # ── Construction ──────────────────────────────────────────────────────

    def __init__(
        self,
        config: IndexingConfig,
        *,
        strategy: Literal[
            "vector", "hnsw", "ivfpq", "ivfpq_disk", "opq_ivfpq"
        ] = "ivfpq",
        distance_strategy: Literal["cosine", "dot_product", "euclidean"] = "cosine",
        memory_config: MemoryConfig | None = None,
        # HNSW
        hnsw_m: int = 32,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 128,
        normalize_l2: bool = False,
        # IVF_PQ
        ivfpq_nlist: int = 4_096,
        ivfpq_m: int = 48,
        ivfpq_nbits: int = 8,
        ivfpq_nprobe: int = 64,
        # OPQ
        opq_m: int = 48,
    ) -> None:
        if strategy not in {*_FAISS_METRIC_MAP, "hnsw", "ivfpq", "ivfpq_disk", "opq_ivfpq", "vector"}:
            raise ValueError(f"Unsupported strategy: {strategy!r}")
        if distance_strategy not in _FAISS_METRIC_MAP:
            raise ValueError(
                f"Unsupported distance_strategy: {distance_strategy!r}. "
                f"Choose from {list(_FAISS_METRIC_MAP)}"
            )

        self.config = config
        self.strategy = strategy
        self.distance_strategy = _FAISS_METRIC_MAP[distance_strategy]
        self.memory_config = memory_config or DEFAULT_MEMORY_CONFIG
        self._normalize_l2 = normalize_l2

        # HNSW
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef_search = hnsw_ef_search

        # IVF_PQ
        self.ivfpq_nlist = ivfpq_nlist
        self.ivfpq_m = ivfpq_m
        self.ivfpq_nbits = ivfpq_nbits
        self.ivfpq_nprobe = ivfpq_nprobe

        # OPQ
        self.opq_m = opq_m

        # State
        self._faiss_store: FAISS | None = None
        self._current_index_path: Path | None = None
        self._is_mmap_loaded: bool = False

        # Streaming training state (reset between indexing runs)
        self._training_vectors: list[np.ndarray] = []
        self._training_vectors_count: int = 0
        self._is_trained: bool = False

        # Build embeddings
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

        # Text splitter (reused across batches)
        if config.chunk_size:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            self._text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
            )
        else:
            self._text_splitter = None

        # Prompt templates
        self._passage_prompt, self._query_prompt = self._load_prompts(
            getattr(config, "passage_prompt_file", None),
            getattr(config, "query_prompt_file", None),
        )

        logger.info(
            f"FAISS {strategy} strategy ready "
            f"(mmap={self.memory_config.use_mmap}, "
            f"max_ram={self.memory_config.max_ram_mb} MB)"
        )

    # ── Prompt helpers ────────────────────────────────────────────────────

    @staticmethod
    def _load_prompts(
        passage_file: str | None,
        query_file: str | None,
    ) -> tuple[str, str]:
        """Load passage and query prompt templates from disk.

        Args:
            passage_file: Override path for the passage prompt file.
            query_file: Override path for the query prompt file.

        Returns:
            Tuple of ``(passage_template, query_template)``.
        """
        from config import DATA_DIR
        prompts_dir = Path(DATA_DIR) / "prompts"

        def _read(override: str | None, name: str, default: str) -> str:
            if override:
                p = Path(override)
                if p.exists():
                    return p.read_text().strip()
                logger.warning(f"Custom prompt {p} not found, falling back to default")
            p = prompts_dir / name
            if p.exists():
                return p.read_text().strip()
            logger.warning(f"Prompt {p} not found, using built-in default")
            return default

        return (
            _read(passage_file, "embeding_promt.txt", "passage: {passage}"),
            _read(query_file,   "query_promt.txt",    "query: {query}"),
        )

    def embed_prompt(self, text: str) -> str:
        """Apply the query prompt template to *text*.

        Args:
            text: Raw query string.

        Returns:
            Prompt-wrapped query string.
        """
        return self._query_prompt.format(query=text)

    def embed_passage(self, text: str) -> str:
        """Apply the passage prompt template to *text*.

        Args:
            text: Raw passage string.

        Returns:
            Prompt-wrapped passage string.
        """
        return self._passage_prompt.format(passage=text)

    def _prepare_query(self, query: str) -> str:
        return self.embed_prompt(query)

    def _prepare_documents(
        self, documents: list[Document], *, log_details: bool = False
    ) -> list[Document]:
        """Optionally chunk and apply the passage prompt to *documents*.

        Args:
            documents: Input document list.
            log_details: Log first-document preview.

        Returns:
            Processed document list (chunked + prompt-wrapped).
        """
        if log_details and documents:
            preview = documents[0].page_content[:100].replace("\n", " ")
            logger.info(f"[Prepare] {len(documents):,} docs, first: '{preview}…'")

        if self._text_splitter is not None:
            documents = self._text_splitter.split_documents(documents)
            if log_details:
                logger.info(f"[Prepare] Split into {len(documents):,} chunks")

        return [
            Document(
                page_content=self._passage_prompt.format(passage=d.page_content),
                metadata=d.metadata,
            )
            for d in documents
        ]

    # ── FAISS index construction ──────────────────────────────────────────

    def _build_raw_index(self, dim: int) -> faiss.Index:
        """Create an (untrained) FAISS index for the configured strategy.

        IVF-family indexes are returned untrained.  Call
        ``_finalize_training`` before adding vectors.

        Args:
            dim: Embedding dimension.

        Returns:
            A FAISS index object.

        Raises:
            ValueError: For unsupported strategy or invalid IVF_PQ params.
        """
        if self.strategy == "vector":
            idx = faiss.IndexFlatL2(dim)
            logger.info(f"[Index] IndexFlatL2 (dim={dim})")

        elif self.strategy == "hnsw":
            idx = faiss.IndexHNSWFlat(dim, self.hnsw_m, faiss.METRIC_L2)
            idx.hnsw.efConstruction = self.hnsw_ef_construction
            idx.hnsw.efSearch = self.hnsw_ef_search
            logger.info(
                f"[Index] IndexHNSWFlat "
                f"(dim={dim}, M={self.hnsw_m}, efC={self.hnsw_ef_construction})"
            )

        elif self.strategy in ("ivfpq", "ivfpq_disk"):
            self._validate_ivfpq(dim)
            quantizer = faiss.IndexFlatL2(dim)
            idx = faiss.IndexIVFPQ(
                quantizer, dim,
                self.ivfpq_nlist, self.ivfpq_m, self.ivfpq_nbits,
            )
            idx.nprobe = self.ivfpq_nprobe
            logger.info(
                f"[Index] IndexIVFPQ "
                f"(dim={dim}, nlist={self.ivfpq_nlist}, "
                f"m={self.ivfpq_m}, nbits={self.ivfpq_nbits})"
            )

        elif self.strategy == "opq_ivfpq":
            self._validate_ivfpq(dim)
            opq = faiss.OPQMatrix(dim, self.opq_m)
            quantizer = faiss.IndexFlatL2(dim)
            ivfpq = faiss.IndexIVFPQ(
                quantizer, dim,
                self.ivfpq_nlist, self.ivfpq_m, self.ivfpq_nbits,
            )
            ivfpq.nprobe = self.ivfpq_nprobe
            idx = faiss.IndexPreTransform(opq, ivfpq)
            logger.info(
                f"[Index] OPQ({self.opq_m}) + IndexIVFPQ "
                f"(dim={dim}, nlist={self.ivfpq_nlist}, m={self.ivfpq_m})"
            )

        else:
            raise ValueError(f"Unsupported strategy: {self.strategy!r}")

        return idx

    def _validate_ivfpq(self, dim: int) -> None:
        """Raise if IVF_PQ sub-quantiser count does not divide dimension.

        Args:
            dim: Embedding dimension.

        Raises:
            ValueError: If ``dim % ivfpq_m != 0``.
        """
        if dim % self.ivfpq_m != 0:
            suggestion = next(
                (m for m in (48, 32, 64, 24, 16, 8) if dim % m == 0), dim
            )
            raise ValueError(
                f"IVF_PQ: dim ({dim}) must be divisible by ivfpq_m ({self.ivfpq_m}). "
                f"Try ivfpq_m={suggestion}."
            )

    # ── Streaming IVF training ────────────────────────────────────────────

    def _accumulate_training_vectors(self, embeddings: list[list[float]]) -> bool:
        """Buffer embedding vectors for streaming IVF training.

        Args:
            embeddings: Batch of embedding vectors.

        Returns:
            ``True`` if enough vectors have been collected for training.
        """
        max_samples = self.memory_config.training_sample_size
        if self._training_vectors_count >= max_samples:
            return True

        n = len(embeddings)
        remaining = max_samples - self._training_vectors_count
        if n <= remaining:
            self._training_vectors.append(np.array(embeddings, dtype=np.float32))
            self._training_vectors_count += n
        else:
            indices = np.random.choice(n, remaining, replace=False)
            sample = np.array([embeddings[i] for i in indices], dtype=np.float32)
            self._training_vectors.append(sample)
            self._training_vectors_count += remaining

        min_train = self.ivfpq_nlist * 39
        return self._training_vectors_count >= min_train

    def _finalize_training(self, index: faiss.Index) -> None:
        """Train *index* on accumulated vectors and clear the buffer.

        Args:
            index: The FAISS index to train (no-op if already trained).
        """
        if self._is_trained or not self._training_vectors:
            return
        all_vecs = np.vstack(self._training_vectors)
        logger.info(f"[Train] Training on {len(all_vecs):,} vectors…")
        index.train(all_vecs)
        self._is_trained = True
        del all_vecs
        self._training_vectors.clear()
        self._training_vectors_count = 0
        gc.collect()
        logger.info("[Train] ✓ Index trained")

    def _reset_training_state(self) -> None:
        """Clear streaming training buffers."""
        self._training_vectors.clear()
        self._training_vectors_count = 0
        self._is_trained = False

    def _setup_ondisk_ivf(self, index: faiss.Index, save_path: Path) -> faiss.Index:
        """Convert trained IVF index to use on-disk inverted lists.

        Args:
            index: Trained IVF index.
            save_path: Root directory for the index files.

        Returns:
            Modified index (inverted lists now on disk).

        Raises:
            ValueError: If the index is not yet trained.
        """
        if not index.is_trained:
            raise ValueError("Index must be trained before setting up on-disk IVF.")
        ivlists_path = save_path / "faiss" / "ivlists.bin"
        ivlists_path.parent.mkdir(parents=True, exist_ok=True)
        ivf_index = faiss.extract_index_ivf(index)
        invlists = faiss.OnDiskInvertedLists(
            ivf_index.nlist, ivf_index.code_size, str(ivlists_path)
        )
        ivf_index.replace_invlists(invlists, True)
        logger.info(f"[Index] On-disk inverted lists → {ivlists_path}")
        return index

    # ── Indexing ──────────────────────────────────────────────────────────

    def index_from_dataframe(
        self,
        df: pd.DataFrame,
        text_field: str,
        *,
        metadata_fields: Sequence[str] | None = None,
        collection_name: str = "faiss_index",
        output_dir: Path | None = None,
        **kwargs: Any,
    ) -> IndexResult:
        """Build a FAISS index from a pandas DataFrame.

        This is a convenience wrapper — for large DataFrames prefer
        ``index_from_parquet_batches``.

        Args:
            df: Source DataFrame.
            text_field: Column containing document text.
            metadata_fields: Extra columns to store as metadata.
            collection_name: Output path for the index when ``output_dir``
                is provided.
            output_dir: Directory to persist the index.  If ``None`` the
                index is kept in memory only.
            **kwargs: Passed to the LangChain ``FAISS.from_documents`` call.

        Returns:
            ``IndexResult`` with the ``FAISS`` store and chunk count.
        """
        documents = documents_from_dataframe(df, text_field, metadata_fields)
        prepared = self._prepare_documents(documents)
        texts = [d.page_content for d in prepared]
        metadatas = [d.metadata for d in prepared]

        logger.info(f"[Index] Embedding {len(texts):,} documents…")
        embeddings = self._embeddings.embed_documents(texts)

        save_path = (
            Path(output_dir) / collection_name if output_dir else Path(collection_name)
        )
        db_dir = save_path / "faiss"
        db_dir.mkdir(parents=True, exist_ok=True)

        dim = len(embeddings[0])
        raw_index = self._build_raw_index(dim)
        if not raw_index.is_trained:
            self._reset_training_state()
            self._accumulate_training_vectors(embeddings)
            self._finalize_training(raw_index)

        docstore = SqliteDocstore(db_dir / "docstore.sqlite")
        self._faiss_store = FAISS(
            embedding_function=self._embeddings,
            index=raw_index,
            docstore=docstore,
            index_to_docstore_id={},
            normalize_L2=self._normalize_l2,
        )
        self._faiss_store.add_embeddings(
            text_embeddings=list(zip(texts, embeddings)),
            metadatas=metadatas,
        )
        self._current_index_path = save_path
        self.save_index(save_path)

        n = len(texts)
        logger.info(f"[Index] ✓ {n:,} chunks indexed → {save_path}")
        return IndexResult(self._faiss_store, n)

    def index_from_parquet(
        self,
        parquet_path: Path,
        *,
        text_field: str = "text",
        metadata_fields: Sequence[str] | None = None,
        collection_name: str = "faiss_index",
        output_dir: Path | None = None,
        **kwargs: Any,
    ) -> IndexResult:
        """Build a FAISS index from a Parquet file (loads the whole file).

        For large files (> a few GB) use ``index_from_parquet_batches``.

        Args:
            parquet_path: Path to the ``.parquet`` file.
            text_field: Column containing document text.
            metadata_fields: Extra columns to store as metadata.
            collection_name: Output path / logical name.
            output_dir: Optional parent directory for the index.
            **kwargs: Ignored.

        Returns:
            ``IndexResult`` with the ``FAISS`` store and chunk count.

        Raises:
            FileNotFoundError: If ``parquet_path`` does not exist.
        """
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
        columns = [text_field] + list(metadata_fields or [])
        df = pq.read_table(parquet_path, columns=columns).to_pandas()
        return self.index_from_dataframe(
            df,
            text_field,
            metadata_fields=metadata_fields,
            collection_name=collection_name,
            output_dir=output_dir,
        )

    def index_from_parquet_batches(
        self,
        parquet_path: Path,
        *,
        text_field: str = "text",
        metadata_fields: Sequence[str] | None = None,
        collection_name: str = "faiss_index",
        batch_size: int = 5_000,
        skip_rows: int = 0,
        checkpoint: bool = True,
        progress_bar: bool = True,
        memory_config: MemoryConfig | None = None,
        **kwargs: Any,
    ) -> IndexResult:
        """Memory-efficient three-stage indexing pipeline from a Parquet file.

        Pipeline stages:
            Producer thread  →  Main thread (embed)  →  Insert thread
            read + chunk         GPU/API-bound            FAISS add

        Memory optimisations:
            - Streaming IVF training (samples, no full dataset in RAM).
            - Configurable queue sizes (default: 2).
            - Aggressive GC between batches.
            - SQLite docstore (documents on disk).
            - Optional on-disk inverted lists (``ivfpq_disk`` strategy).

        Args:
            parquet_path: Path to the ``.parquet`` file.
            text_field: Column containing document text.
            metadata_fields: Extra columns to store as metadata.
            collection_name: Output directory name for the index.
            batch_size: Parquet rows per batch.  Smaller = less RAM.
            skip_rows: Resume from this row offset.
            checkpoint: Save the index after every batch.
            progress_bar: Show tqdm progress bar.
            memory_config: Override service-level memory settings.
            **kwargs: Ignored.

        Returns:
            ``IndexResult`` with the ``FAISS`` store and total chunks indexed.

        Raises:
            FileNotFoundError: If ``parquet_path`` does not exist.
            ValueError: If embeddings are not configured.
        """
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        self._reset_training_state()
        mem_cfg = memory_config or self.memory_config
        total_chunks = 0
        rows_processed = 0
        batches_since_gc = 0

        parquet_file = pq.ParquetFile(parquet_path)
        total_rows = parquet_file.metadata.num_rows
        columns = [text_field] + list(metadata_fields or [])
        save_path = Path(collection_name)

        logger.info(
            f"Indexing {total_rows - skip_rows:,} rows "
            f"(of {total_rows:,} total) | "
            f"batch={batch_size:,} | strategy={self.strategy}"
        )

        # ── Shared state ──────────────────────────────────────────────────
        prepare_q: queue.Queue = queue.Queue(maxsize=mem_cfg.queue_maxsize)
        insert_q:  queue.Queue = queue.Queue(maxsize=mem_cfg.queue_maxsize)
        errors: list[Exception] = []
        cancel = threading.Event()
        store_lock = threading.Lock()
        training_done = threading.Event()
        pending_index: list[faiss.Index] = []   # holds index before training finishes
        pending_batches: list[tuple] = []        # batches buffered while IVF trains

        def _safe_put(q: queue.Queue, item: Any) -> bool:
            while not cancel.is_set():
                try:
                    q.put(item, timeout=2)
                    return True
                except queue.Full:
                    continue
            return False

        def _safe_get(q: queue.Queue) -> tuple[Any, bool]:
            while not cancel.is_set():
                try:
                    return q.get(timeout=2), True
                except queue.Empty:
                    continue
            return None, False

        # ── Stage 1: Producer ─────────────────────────────────────────────
        def producer() -> None:
            try:
                rows_seen = 0
                for batch in parquet_file.iter_batches(
                    batch_size=batch_size, columns=columns
                ):
                    if cancel.is_set():
                        break
                    batch_len = batch.num_rows
                    if rows_seen + batch_len <= skip_rows:
                        rows_seen += batch_len
                        continue
                    df = batch.to_pandas()
                    if rows_seen < skip_rows:
                        df = df.iloc[skip_rows - rows_seen:]
                    rows_seen += batch_len

                    docs = documents_from_dataframe(df, text_field, metadata_fields)
                    chunks = self._prepare_documents(docs)
                    del docs, df
                    if not _safe_put(prepare_q, (chunks, batch_len)):
                        break
                    del chunks
                    gc.collect()
            except Exception as e:
                errors.append(e)
                cancel.set()
            finally:
                _safe_put(prepare_q, None)

        # ── Stage 3: Inserter ─────────────────────────────────────────────
        def inserter() -> None:
            nonlocal total_chunks, rows_processed, batches_since_gc
            try:
                while True:
                    item, ok = _safe_get(insert_q)
                    if not ok or item is None:
                        break

                    texts, embeddings, metadatas, batch_rows = item
                    n = len(texts)
                    logger.info(f"[Insert] {n:,} docs → FAISS")

                    with store_lock:
                        if self._faiss_store is None:
                            dim = len(embeddings[0])
                            raw_index = self._build_raw_index(dim)

                            if not raw_index.is_trained:
                                has_enough = self._accumulate_training_vectors(embeddings)
                                if has_enough:
                                    self._finalize_training(raw_index)
                                    training_done.set()
                                    if self.strategy == "ivfpq_disk":
                                        raw_index = self._setup_ondisk_ivf(raw_index, save_path)
                                else:
                                    pending_index.append(raw_index)
                                    pending_batches.append(
                                        (texts, embeddings, metadatas, batch_rows)
                                    )
                                    logger.info(
                                        f"[Insert] Accumulating training vectors… "
                                        f"({self._training_vectors_count:,} so far)"
                                    )
                                    continue  # skip — not trained yet
                            else:
                                training_done.set()

                            db_dir = save_path / "faiss"
                            db_dir.mkdir(parents=True, exist_ok=True)
                            self._faiss_store = FAISS(
                                embedding_function=self._embeddings,
                                index=raw_index,
                                docstore=SqliteDocstore(db_dir / "docstore.sqlite"),
                                index_to_docstore_id={},
                                normalize_L2=self._normalize_l2,
                            )

                        # Handle deferred training completion
                        elif pending_index and not training_done.is_set():
                            has_enough = self._accumulate_training_vectors(embeddings)
                            if not has_enough:
                                pending_batches.append(
                                    (texts, embeddings, metadatas, batch_rows)
                                )
                                continue  # still collecting
                            self._finalize_training(pending_index[0])
                            training_done.set()
                            if self.strategy == "ivfpq_disk":
                                pending_index[0] = self._setup_ondisk_ivf(
                                    pending_index[0], save_path
                                )
                            db_dir = save_path / "faiss"
                            db_dir.mkdir(parents=True, exist_ok=True)
                            self._faiss_store = FAISS(
                                embedding_function=self._embeddings,
                                index=pending_index[0],
                                docstore=SqliteDocstore(db_dir / "docstore.sqlite"),
                                index_to_docstore_id={},
                                normalize_L2=self._normalize_l2,
                            )
                            # Flush batches that were buffered during training
                            for p_texts, p_embs, p_metas, p_rows in pending_batches:
                                self._faiss_store.add_embeddings(
                                    text_embeddings=list(zip(p_texts, p_embs)),
                                    metadatas=p_metas,
                                )
                                total_chunks += len(p_texts)
                                rows_processed += p_rows
                                pbar.update(p_rows)
                            pending_batches.clear()

                        self._faiss_store.add_embeddings(
                            text_embeddings=list(zip(texts, embeddings)),
                            metadatas=metadatas,
                        )

                    total_chunks += n
                    rows_processed += batch_rows
                    batches_since_gc += 1
                    pbar.update(batch_rows)

                    if checkpoint:
                        try:
                            self.save_index(save_path)
                            logger.info(
                                f"[Checkpoint] {rows_processed:,} rows / "
                                f"{total_chunks:,} chunks → {save_path}"
                            )
                        except Exception as ckpt_err:
                            logger.warning(f"[Checkpoint] Save failed: {ckpt_err}")

                    del texts, embeddings, metadatas
                    if batches_since_gc >= mem_cfg.gc_every_n_batches:
                        gc.collect()
                        batches_since_gc = 0

            except Exception as e:
                errors.append(e)
                cancel.set()

        # ── Start threads ─────────────────────────────────────────────────
        prod_t   = threading.Thread(target=producer,  name="FAISSProd",   daemon=True)
        insert_t = threading.Thread(target=inserter,  name="FAISSInsert", daemon=True)
        pbar = tqdm(
            total=total_rows,
            initial=skip_rows,
            desc="Indexing",
            unit="row",
            disable=not progress_bar,
        )

        prod_t.start()
        insert_t.start()

        # ── Stage 2: Embedder (main thread) ───────────────────────────────
        try:
            while True:
                item, ok = _safe_get(prepare_q)
                if not ok or item is None:
                    break
                chunks, batch_rows = item
                texts     = [d.page_content for d in chunks]
                metadatas = [d.metadata     for d in chunks]
                del chunks
                embeddings = self._embeddings.embed_documents(texts)
                if not _safe_put(insert_q, (texts, embeddings, metadatas, batch_rows)):
                    break
                del texts, embeddings, metadatas
                gc.collect()
        except Exception as e:
            errors.append(e)
            cancel.set()
        finally:
            _safe_put(insert_q, None)

        prod_t.join()
        insert_t.join()
        pbar.close()

        # Final save
        if self._faiss_store is not None:
            try:
                self.save_index(save_path)
                logger.info(f"[Save] Final save → {save_path}")
            except Exception as save_err:
                logger.error(f"[Save] Final save failed: {save_err}")
        self._current_index_path = save_path

        if errors:
            raise errors[0]

        logger.info(
            f"=== Indexing complete — "
            f"{total_chunks:,} chunks from {total_rows:,} rows ==="
        )
        return IndexResult(self._faiss_store, total_chunks)

    # ── Index lifecycle ───────────────────────────────────────────────────

    def load_index(
        self,
        path_or_name: str | Path,
        use_mmap: bool | None = None,
        docstore_path: str | Path | None = None,
        **kwargs: Any,
    ) -> FAISS:
        """Load a saved FAISS index from disk.

         Expects a directory with the layout::

            <path>/faiss/index.faiss
            <path>/faiss/docstore.sqlite

        Args:
            path_or_name: Root directory of the saved index.
            use_mmap: If ``True``, memory-map the index file.  Defaults to
                ``self.memory_config.use_mmap``.
            docstore_path: Optional override for the SQLite docstore file.
                Useful when the docstore lives in a different directory (e.g.
                ``ivfpq_low`` sharing ``faiss_high``'s docstore after the
                low-index copy was deleted).  When ``None``, defaults to
                ``<path>/faiss/docstore.sqlite``.
            **kwargs: Ignored.

        Returns:
            The loaded ``FAISS`` store.

        Raises:
            FileNotFoundError: If the index directory or files are missing.
            ValueError: If the id_map in the docstore is empty.
        """
        path = Path(path_or_name)
        faiss_dir = path / "faiss"
        faiss_file = faiss_dir / "index.faiss"
        db_file    = Path(docstore_path) if docstore_path is not None else faiss_dir / "docstore.sqlite"

        if not faiss_dir.exists():
            raise FileNotFoundError(f"FAISS index directory not found: {faiss_dir}")
        if not faiss_file.exists():
            raise FileNotFoundError(f"index.faiss missing in {faiss_dir}")

        mmap = use_mmap if use_mmap is not None else self.memory_config.use_mmap
        if mmap:
            logger.info("Loading FAISS index with memory-mapping…")
            raw = faiss.read_index(str(faiss_file), faiss.IO_FLAG_MMAP)
            self._is_mmap_loaded = True
        else:
            logger.info("Loading FAISS index into RAM…")
            raw = faiss.read_index(str(faiss_file))
            self._is_mmap_loaded = False

        if hasattr(raw, "nprobe"):
            raw.nprobe = self.ivfpq_nprobe
        if hasattr(raw, "hnsw"):
            raw.hnsw.efSearch = self.hnsw_ef_search

        logger.info(f"✓ {raw.ntotal:,} vectors ({'mmap' if mmap else 'RAM'})")

        docstore = SqliteDocstore(db_file)
        id_map = docstore.load_id_map()
        if not id_map:
            raise ValueError(f"id_map is empty in {db_file} — index may be corrupt")

        self._faiss_store = FAISS(
            embedding_function=self._embeddings,
            index=raw,
            docstore=docstore,
            index_to_docstore_id=id_map,
            normalize_L2=self._normalize_l2,
        )
        self._current_index_path = path
        return self._faiss_store

    def save_index(self, path: str | Path, **kwargs: Any) -> None:
        """Persist the current FAISS index to *path*.

        Writes two files under ``<path>/faiss/``::

            index.faiss      – raw FAISS binary
            docstore.sqlite  – text, metadata, and id_map

        The FAISS file is written atomically via a tmp-rename.

        Args:
            path: Destination root directory.
            **kwargs: Ignored.

        Raises:
            ValueError: If no index is currently loaded.
        """
        if self._faiss_store is None:
            raise ValueError("No FAISS index to save.")

        faiss_dir = Path(path) / "faiss"
        faiss_dir.mkdir(parents=True, exist_ok=True)

        tmp = faiss_dir / "index.faiss.tmp"
        faiss.write_index(self._faiss_store.index, str(tmp))
        tmp.rename(faiss_dir / "index.faiss")

        self._faiss_store.docstore.save_id_map(
            self._faiss_store.index_to_docstore_id
        )
        n = len(self._faiss_store.index_to_docstore_id)
        logger.info(f"FAISS index saved → {faiss_dir} ({n:,} vectors)")

    def delete_index(self, *, delete_files: bool = False, **kwargs: Any) -> None:
        """Drop the current index from memory and optionally from disk.

        Args:
            delete_files: If ``True`` and the index was loaded from / saved
                to disk, delete the entire ``<path>/faiss/`` directory.
            **kwargs: Ignored.
        """
        if delete_files and self._current_index_path:
            faiss_dir = self._current_index_path / "faiss"
            if faiss_dir.exists():
                shutil.rmtree(faiss_dir)
                logger.info(f"Deleted FAISS index files at {faiss_dir}")
        self._faiss_store = None
        self._current_index_path = None
        self._is_mmap_loaded = False
        logger.info("FAISS index cleared from memory")

    # ── Retrieval ─────────────────────────────────────────────────────────

    def _require_store(self) -> FAISS:
        """Return the current FAISS store or raise if not loaded.

        Returns:
            The active ``FAISS`` store.

        Raises:
            ValueError: If no index is loaded.
        """
        if self._faiss_store is None:
            raise ValueError(
                "No FAISS index loaded. "
                "Call load_index() or index_from_parquet_batches() first."
            )
        return self._faiss_store

    def retrieve_documents(
        self,
        query: str,
        *,
        top_k: int = 5,
        **kwargs: Any,
    ) -> list[Document]:
        """Return the top-k documents most relevant to *query*.

        Args:
            query: Free-text query string.
            top_k: Number of documents to return.
            **kwargs: Ignored.

        Returns:
            Ranked list of ``Document`` objects.

        Raises:
            ValueError: If no index is loaded.
        """
        return [doc for doc, _ in self.retrieve_documents_with_scores(query, top_k=top_k)]

    def retrieve_documents_with_scores(
        self,
        query: str,
        *,
        top_k: int = 5,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return the top-k documents with cosine similarity scores.

        Args:
            query: Free-text query string.
            top_k: Number of results to return.
            **kwargs: Ignored.

        Returns:
            List of ``(Document, score)`` tuples, best-first.

        Raises:
            ValueError: If no index is loaded.
        """
        store = self._require_store()
        prepared = self._prepare_query(query)

        # Over-fetch so we still return top_k real docs even if some FAISS
        # positions land on training-only vectors (e.g. IVF train_ slots)
        # that have no entry in index_to_docstore_id or in the docstore.
        fetch_k = top_k * 2 + 256
        embedding = store.embedding_function.embed_query(prepared)
        vec = np.array([embedding], dtype=np.float32)
        if store._normalize_L2:
            faiss.normalize_L2(vec)
        scores, indices = store.index.search(vec, fetch_k)

        results: list[tuple[Document, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            uid = store.index_to_docstore_id.get(int(idx))
            if uid is None:
                continue
            doc = store.docstore.search(uid)
            if not isinstance(doc, Document):
                # uid exists in id_map but not in docs table (e.g. train_ vectors)
                continue
            results.append((doc, float(score)))
            if len(results) >= top_k:
                break

        return results

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
            **kwargs: Ignored.

        Returns:
            One list of Documents per query.

        Raises:
            ValueError: If no index is loaded.
        """
        self._require_store()
        return [
            self.retrieve_documents(q, top_k=top_k)
            for q in tqdm(
                queries, desc=f"Retrieving ({self.strategy})", disable=not progress_bar
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

        Args:
            queries: List of query strings.
            top_k: Results per query.
            progress_bar: Show tqdm progress bar.
            **kwargs: Ignored.

        Returns:
            One list of ``(Document, score)`` tuples per query.

        Raises:
            ValueError: If no index is loaded.
        """
        self._require_store()
        return [
            self.retrieve_documents_with_scores(q, top_k=top_k)
            for q in tqdm(
                queries, desc=f"Retrieving ({self.strategy})", disable=not progress_bar
            )
        ]

    # ── Inspection ────────────────────────────────────────────────────────

    def get_doc_count(self) -> int:
        """Return the number of vectors currently indexed.

        Returns:
            Vector count, or 0 if no index is loaded.
        """
        if self._faiss_store is None:
            return 0
        return len(self._faiss_store.index_to_docstore_id)

    def get_all_documents(
        self,
        *,
        batch_size: int = 1_000,
        progress_bar: bool = True,
    ) -> list[Document]:
        """Return every document from the SQLite docstore.

        Iterates the ``index_to_docstore_id`` map and fetches each
        document from ``SqliteDocstore`` in batches.

        Args:
            batch_size: Number of docstore IDs to fetch per SQLite query.
            progress_bar: Show tqdm progress bar.

        Returns:
            List of all stored ``Document`` objects.

        Raises:
            ValueError: If no index is loaded.
        """
        store = self._require_store()
        id_map = store.index_to_docstore_id
        docstore: SqliteDocstore = store.docstore
        all_uids = list(id_map.values())
        results: list[Document] = []

        for start in tqdm(
            range(0, len(all_uids), batch_size),
            desc="Fetching documents",
            unit="batch",
            disable=not progress_bar,
        ):
            uid_batch = all_uids[start : start + batch_size]
            for uid in uid_batch:
                doc = docstore.search(uid)
                if doc is not None and not isinstance(doc, str):
                    results.append(doc)

        return results

    def get_index_stats(self) -> dict[str, Any]:
        """Return statistics about the current FAISS index.

        Returns:
            Dict with keys: ``loaded``, ``is_mmap``, ``n_vectors``,
            ``strategy``, and optionally ``nlist``, ``nprobe``,
            ``hnsw_m``, ``hnsw_ef_search``.
        """
        if self._faiss_store is None:
            return {"loaded": False}

        raw = self._faiss_store.index
        stats: dict[str, Any] = {
            "loaded":    True,
            "is_mmap":   self._is_mmap_loaded,
            "n_vectors": raw.ntotal,
            "strategy":  self.strategy,
        }
        if hasattr(raw, "nlist"):
            stats["nlist"]  = raw.nlist
            stats["nprobe"] = raw.nprobe
        if hasattr(raw, "hnsw"):
            stats["hnsw_m"]         = self.hnsw_m
            stats["hnsw_ef_search"] = raw.hnsw.efSearch
        return stats

    # ── Memory estimation helpers ─────────────────────────────────────────

    def estimate_index_memory(
        self, n_vectors: int, *, dim: int = 1_024
    ) -> dict[str, float]:
        """Estimate RAM usage (MB) for each index strategy.

        Args:
            n_vectors: Number of vectors to index.
            dim: Embedding dimension.

        Returns:
            Dict mapping strategy name to estimated RAM in MB.
        """
        f32 = 4
        flat_bytes = n_vectors * dim * f32
        centroids  = self.ivfpq_nlist * dim * f32
        return {
            "vector":     flat_bytes / 1e6,
            "hnsw":       flat_bytes * 2.2 / 1e6,
            "ivfpq":      (n_vectors * self.ivfpq_m + centroids) / 1e6,
            "opq_ivfpq":  (n_vectors * self.ivfpq_m + centroids + dim * dim * f32) / 1e6,
            "ivfpq_disk": (centroids + n_vectors * 8) / 1e6,
        }

    def recommend_strategy(
        self, n_vectors: int, available_ram_mb: int, *, dim: int = 1_024
    ) -> str:
        """Recommend the best strategy within the RAM budget.

        Args:
            n_vectors: Number of vectors to index.
            available_ram_mb: Available RAM in MB.
            dim: Embedding dimension.

        Returns:
            Recommended strategy name.
        """
        budget = available_ram_mb * 0.8
        estimates = self.estimate_index_memory(n_vectors, dim=dim)
        for strategy in ("vector", "hnsw", "ivfpq", "opq_ivfpq", "ivfpq_disk"):
            if estimates.get(strategy, float("inf")) < budget:
                return strategy
        return "ivfpq_disk"
