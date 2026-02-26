"""FAISS RAG service — vector (FAISS), BM25, and hybrid retrieval."""

from __future__ import annotations

import gc
import logging
import pickle
import queue
import shutil
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Literal, Optional, Sequence

import faiss
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS, DistanceStrategy
from tqdm import tqdm
import os

from .base import RagService, VectorStoreLike
from .document_utils import documents_from_dataframe
from .utils import IndexingConfig, build_embeddings
from .SqliteDocstore import SqliteDocstore

logger = logging.getLogger(__name__)

# FAISS + OpenMP can have issues with "RuntimeError: OpenMP error: Cannot fork a new thread" on some platforms (e.g., macOS). Setting this env var allows it to proceed with a warning instead of crashing. See
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
faiss.omp_set_num_threads(1)

_FAISS_METRIC_MAP = {
    "cosine": DistanceStrategy.COSINE,
    "dot_product": DistanceStrategy.MAX_INNER_PRODUCT,
    "euclidean": DistanceStrategy.EUCLIDEAN,
}
            
class FaissRagService(RagService):
    """Unified FAISS / BM25 / hybrid retrieval service.

    Strategies:
        * ``vector``  – exact brute-force search (``IndexFlatL2`` /
          ``IndexFlatIP``).
        * ``approximation`` – approximate kNN via HNSW
          (``IndexHNSWFlat``).
        * ``bm25`` – lexical BM25 retrieval (``rank_bm25``).
    """

    # ── Construction ─────────────────────────────────────────────────────

    def __init__(
        self,
        config: IndexingConfig | None = None,
        *,
        strategy: Literal["vector", "approximation", "bm25", "ivfpq"] = "vector",
        distance_strategy: Literal["cosine", "dot_product", "euclidean"] | None = None,
        connected_graph_notes: int = 32,
        normalize_l2: bool = False,
        # HNSW tuning (only used when strategy="approximation")
        # IVF_PQ tuning (only used when strategy="ivfpq")
        ivfpq_nlist: int = 4096,    # Voronoi cells — tune to ~sqrt(total_vectors)
        ivfpq_m: int = 16,           # sub-quantizers; dim must be divisible by m
        ivfpq_nbits: int = 8,        # bits/sub-quantizer  (8 → 256 centroids, 16 bytes/vec)
        ivfpq_nprobe: int = 64,      # cells searched per query (higher = better recall)
    ):
        self.strategy = strategy
        self._normalize_l2 = normalize_l2

        # Current in-memory stores
        self._faiss_store: FAISS | None = None
        self._bm25_retriever: BM25Retriever | None = None
        self._current_index_path: Path | None = None

        # HNSW
        self.connected_graph_notes = connected_graph_notes

        # IVF_PQ
        self.ivfpq_nlist = ivfpq_nlist
        self.ivfpq_m = ivfpq_m
        self.ivfpq_nbits = ivfpq_nbits
        self.ivfpq_nprobe = ivfpq_nprobe

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
            )

            if distance_strategy not in _FAISS_METRIC_MAP.keys():
                raise ValueError(f"Unsupported distance function: {distance_strategy}")

            self.distance_strategy = _FAISS_METRIC_MAP.get(
                distance_strategy,
                DistanceStrategy.COSINE
            )

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

        logger.info(f"FAISS {strategy} strategy ready")

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

    def _build_faiss_index(self, dim: int, connected_graph_notes: int) -> faiss.Index:
        """Create a raw FAISS index matching the chosen strategy/distance.

        IVF_PQ is returned **untrained** — call ``_train_faiss_index`` before
        adding any vectors.
        """
        if self.strategy == "vector":
            index = faiss.IndexFlat(dim)
        elif self.strategy == "hnsw":
            index = faiss.IndexHNSWFlat(dim, connected_graph_notes, faiss.METRIC_L2)
        elif self.strategy == "ivfpq":
            if dim % self.ivfpq_m != 0:
                raise ValueError(
                    f"IVF_PQ: dim ({dim}) must be divisible by ivfpq_m ({self.ivfpq_m})"
                )
            quantizer = faiss.IndexFlatL2(dim)
            index = faiss.IndexIVFPQ(
                quantizer, dim,
                self.ivfpq_nlist,
                self.ivfpq_m,
                self.ivfpq_nbits,
            )
            index.nprobe = self.ivfpq_nprobe
        else:
            raise ValueError(f"Unsupported strategy: {self.strategy}")

        return index

    def _train_faiss_index(self, index: faiss.Index, embeddings: list[list[float]]) -> None:
        """Train an IVF_PQ index on *embeddings* (no-op for flat/HNSW)."""
        if not hasattr(index, 'is_trained') or index.is_trained:
            return
        min_train = self.ivfpq_nlist * 39  # FAISS recommendation
        n = len(embeddings)
        if n < min_train:
            logger.warning(
                f"[Train] Only {n:,} vectors for training; recommended >= {min_train:,} "
                f"(nlist={self.ivfpq_nlist}). Recall may be reduced."
            )
        vectors = np.array(embeddings, dtype=np.float32)
        logger.info(f"[Train] Training IVF_PQ on {n:,} vectors...")
        index.train(vectors)
        logger.info(f"[Train] ✓ IVF_PQ trained")

    def load_faiss_store(self, path: str) -> Optional[FAISS]:
        """Load a previously-saved FAISS index from *path* (no pickle)."""
        path = Path(path)
        faiss_dir = path / "faiss"
        if not faiss_dir.exists():
            logger.warning(f"FAISS index directory not found at {faiss_dir}")
            raise FileNotFoundError(f"FAISS index directory not found at {faiss_dir}")


        faiss_file = faiss_dir / "index.faiss"
        db_file = faiss_dir / "docstore.sqlite"

        if not faiss_file.exists():
            raise FileNotFoundError(f"index.faiss missing in {faiss_dir}")

        print(f"  Loading FAISS binary index...")
        raw_index = faiss.read_index(str(faiss_file))
        # Restore IVF_PQ query-time probe count if applicable
        if hasattr(raw_index, 'nprobe'):
            raw_index.nprobe = self.ivfpq_nprobe
        print(f"  ✓ {raw_index.ntotal:,} vectors loaded")

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
    ) -> tuple[FAISS, int]:
        """
        Three-stage pipeline with bounded queues:

          Producer thread  →  Main thread (embed)  →  Insert thread
          read + chunk         GPU/API-bound            FAISS add

        Mirrors ``ElasticsearchRagService.index_from_parquet_batches``.
        """
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
        if self._embeddings is None:
            raise ValueError("Embeddings required for parquet batch indexing")

        total_chunks = 0
        rows_processed = 0

        parquet_file = pq.ParquetFile(parquet_path)
        total_rows = parquet_file.metadata.num_rows
        rows_to_index = total_rows - skip_rows

        columns = [text_field] + list(metadata_fields or [])
        if skip_rows > 0:
            logger.info(f"Skipping first {skip_rows:,} rows")
        logger.info(
            f"Indexing {rows_to_index:,} rows (of {total_rows:,} total) | parquet_batch={batch_size:,}"
        )

        # ── Shared state ─────────────────────────────────────────────────
        prepare_queue: queue.Queue = queue.Queue(maxsize=2)
        insert_queue: queue.Queue = queue.Queue(maxsize=2)
        exception_holder: list[Exception] = []
        cancel = threading.Event()
        store_lock = threading.Lock()

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
            nonlocal total_chunks, rows_processed
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
                            raw_index = self._build_faiss_index(dim, self.connected_graph_notes)

                            # IVF_PQ: must train before any vectors can be added
                            if not raw_index.is_trained:
                                self._train_faiss_index(raw_index, embeddings)

                            db_dir = Path(save_path) / "faiss"
                            db_dir.mkdir(parents=True, exist_ok=True)
                            self._faiss_store = FAISS(
                                embedding_function=self._embeddings,
                                index=raw_index,
                                docstore=SqliteDocstore(db_path=db_dir / "docstore.sqlite"),
                                index_to_docstore_id={},
                                normalize_L2=False,
                            )

                        self._faiss_store.add_embeddings(
                            text_embeddings=text_embedding_pairs,
                            metadatas=metadatas,
                        )

                    total_chunks += n
                    rows_processed += batch_rows
                    pbar.update(batch_rows)
                    logger.info(
                        f"[Insert] ✓ +{n:,} chunks / {batch_rows:,} rows "
                        f"(cumulative: {total_chunks:,} chunks, {rows_processed:,} rows)"
                    )

                    # ── Checkpoint save after every batch ────────────────
                    if checkpoint:
                        try:
                            logger.info(f"[Checkpoint] Saving index at {rows_processed:,} rows...")
                            self.save_index(save_path)
                            logger.info(
                                f"[Checkpoint] Saved at {rows_processed:,} rows "
                                f"({total_chunks:,} chunks) → {save_path}"
                            )
                        except Exception as ckpt_err:
                            logger.warning(f"[Checkpoint] Save failed: {ckpt_err}")

                    del texts, embeddings, metadatas
                    gc.collect()
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