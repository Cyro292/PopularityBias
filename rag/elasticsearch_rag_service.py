"""Elasticsearch RAG service — vector, BM25, and hybrid retrieval."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import gc
import logging
import time
import queue
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Literal, Optional, Sequence, Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from langchain.schema import Document
from langchain_elasticsearch import ElasticsearchStore
from langchain_elasticsearch.vectorstores import BM25Strategy, DenseVectorStrategy, DenseVectorScriptScoreStrategy
from tqdm import tqdm

from .base import RagService, VectorStoreLike
from .document_utils import documents_from_dataframe
from .utils import IndexingConfig, build_embeddings, split_documents

logger = logging.getLogger(__name__)

CustomQueryFn = Optional[Callable[[dict, Optional[str]], dict]]
_VALID_DISTANCES = {"COSINE", "DOT_PRODUCT", "EUCLIDEAN_DISTANCE"}


class ElasticsearchRagService(RagService):
    """Unified Elasticsearch retrieval: dense vector, BM25, or hybrid."""

    def __init__(
        self,
        config: IndexingConfig | None = None,
        *,
        es_url: str = "http://localhost:9200",
        es_user: str | None = None,
        es_password: str | None = None,
        strategy: Literal["vector", "approximation", "bm25", "hybrid"] = "vector",
        distance_function: Literal["COSINE", "DOT_PRODUCT", "EUCLIDEAN_DISTANCE"] | None = None,
    ):
        self.es_url = es_url
        self.es_user = es_user
        self.es_password = es_password
        self.strategy = strategy
        self._current_index_name: str | None = None

        # Load prompt templates
        self._passage_prompt, self._query_prompt = self._load_prompts()

        if strategy == "bm25":
            self.config = config or IndexingConfig()
            self._embeddings = None
            self._retrieval_strategy = BM25RetrievalStrategy()
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
            dist = distance_function or config.distance_function
            if dist and dist not in _VALID_DISTANCES:
                raise ValueError(f"Invalid distance: {dist}. Use one of {_VALID_DISTANCES}")
            self.distance_strategy = dist

            if strategy == "vector":
                # Exact brute-force vector search (script_score over all docs)
                self._retrieval_strategy = DenseVectorScriptScoreStrategy()
            elif strategy == "approximation":
                # Approximate kNN via HNSW graph (fast, use num_candidates to tune recall)
                self._retrieval_strategy = DenseVectorStrategy()
            elif strategy == "hybrid":
                # Approximate kNN + BM25 combined scoring
                self._retrieval_strategy = DenseVectorStrategy(hybrid=True)

        logger.info(f"Elasticsearch {strategy} strategy ready")

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
        if self.strategy != "bm25" and self._embeddings is not None:
            return self._query_prompt.format(query=query)
        return query

    def _prepare_documents(self, documents: list[Document], log_details: bool = False) -> list[Document]:
        """Chunk documents and apply embedding prompt."""
        original_count = len(documents)
        
        if log_details and documents:
            first_doc_preview = documents[0].page_content[:100].replace('\n', ' ')
            logger.info(f"Processing {original_count} documents, first doc start: '{first_doc_preview}...'")
            second_doc_preview = documents[1].page_content[:100].replace('\n', ' ') if len(documents) > 1 else "N/A"
            logger.info(f"Second doc start: '{second_doc_preview}...'")
        
        if self.config.chunk_size:
            documents = split_documents(
                documents, self.config.chunk_size, self.config.chunk_overlap
            )
            if log_details:
                logger.info(f"Split {original_count} documents into {len(documents)} chunks (chunk_size={self.config.chunk_size}, overlap={self.config.chunk_overlap})")
        
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
            first_prepared_preview = prepared[0].page_content[:100].replace('\n', ' ')
            logger.info(f"First prepared doc start: '{first_prepared_preview}...'")
            second_prepared_preview = prepared[1].page_content[:100].replace('\n', ' ') if len(prepared) > 1 else "N/A"
            logger.info(f"Second prepared doc start: '{second_prepared_preview}...'")
        
        return prepared

    def _create_store(self, index_name: str) -> ElasticsearchStore:
        """Create an ElasticsearchStore instance."""

        if index_name is None:
            raise ValueError("No index loaded")

        kwargs: dict = {
            "es_url": self.es_url,
            "index_name": index_name,
            "strategy": self._retrieval_strategy,
            "es_user": self.es_user,
            "es_password": self.es_password
        }
        if self._embeddings is not None:
            kwargs["embedding"] = self._embeddings
        if getattr(self, "distance_strategy", None):
            kwargs["distance_strategy"] = self.distance_strategy

        return ElasticsearchStore(**kwargs)
    
    def load_index(self, index_name: str) -> ElasticsearchStore:
        """Load an existing index by name."""
        self._current_index_name = index_name
        return self._create_store(index_name)

    def _bulk_insert(
        self,
        store: ElasticsearchStore,
        documents: list[Document],
        batch_size: int,
        show_progress: bool,
    ) -> int:
        """BM25 bulk insert — no embeddings needed."""
        start = time.time()
        indexed = 0
        logger.info(f"Starting bulk insert of {len(documents):,} documents with batch_size={batch_size}")
        
        pbar = tqdm(total=len(documents), desc="ES insert (BM25)", unit="doc") if show_progress else None

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            store.add_documents(batch)
            indexed += len(batch)
            if pbar:
                pbar.update(len(batch))

        if pbar:
            pbar.close()
        logger.info(f"Successfully indexed {indexed:,} documents in {time.time() - start:.1f}s")
        return indexed
    
    def index_from_parquet(self, parquet_path, output_dir, *, text_field = None, html_field = None, metadata_fields = None, collection_name = "rag"):
        return super().index_from_parquet(parquet_path, output_dir, text_field=text_field, html_field=html_field, metadata_fields=metadata_fields, collection_name=collection_name)

    def index_from_parquet_batches(
        self,
        parquet_path: Path,
        *,
        text_field: str = "text",
        metadata_fields: Sequence[str] | None = None,
        collection_name: str = None,
        progress_bar: bool | None = None,
        batch_size: int,
        skip_rows: int = 0,
    ) -> tuple[ElasticsearchStore, int]:
        """
        Three-stage pipeline with sub-batching and bounded queues:

          Producer thread  →  Main thread (embed)  →  Upload thread
          read+chunk+sub     GPU-bound (Modal)        I/O to ES

        Embedding is the bottleneck (remote GPU), so upload runs on a
        separate thread to overlap with the next embed call.  Sub-batching
        (≤ embed_sub_batch per queue item) keeps RAM bounded.
        """

        if collection_name is None:
            raise ValueError("collection_name is required for indexing")
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
        if self._embeddings is None:
            raise ValueError("Embeddings required for parquet batch indexing")

        store = self._create_store(collection_name)
        total_chunks = 0
        rows_processed = 0

        parquet_file = pq.ParquetFile(parquet_path)
        total_rows = parquet_file.metadata.num_rows
        rows_to_index = total_rows - skip_rows

        # Pipeline tuning ─────────────────────────────────────────────────
        columns = [text_field] + list(metadata_fields or [])
        if skip_rows > 0:
            logger.info(f"Skipping first {skip_rows:,} rows")
        logger.info(
            f"Indexing {rows_to_index:,} rows (of {total_rows:,} total) | parquet_batch={batch_size:,}"
        )

        # ── Shared state ─────────────────────────────────────────────────
        prepare_queue: queue.Queue = queue.Queue(maxsize=2)   # producer → embed
        upload_queue: queue.Queue  = queue.Queue(maxsize=2)   # embed → upload
        exception_holder: list[Exception] = []
        cancel = threading.Event()

        def _safe_put(q, item):
            while not cancel.is_set():
                try:
                    q.put(item, timeout=2)
                    return True
                except queue.Full:
                    continue
            return False

        def _safe_get(q):
            while not cancel.is_set():
                try:
                    return q.get(timeout=2), True
                except queue.Empty:
                    continue
            return None, False

        # ── Stage 1 – Producer thread: read → chunk → sub-batch ──────────
        def producer():
            try:
                rows_seen = 0
                for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
                    if cancel.is_set():
                        break
                    batch_len = batch.num_rows
                    # ── Skip logic ────────────────────────────────────────
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
                    # ──────────────────────────────────────────────────────
                    logger.info(f"[Prepare] {len(df):,} rows from parquet")

                    n_rows_in_batch = len(df)
                    documents = documents_from_dataframe(df, text_field, metadata_fields)
                    chunks = self._prepare_documents(documents, log_details=False)
                    logger.info(f"[Prepare] {len(chunks):,} chunks (from {n_rows_in_batch:,} rows)")
                    del documents, df

                    if not _safe_put(prepare_queue, (chunks, n_rows_in_batch)):
                        break

                    del chunks
                    gc.collect()
            except Exception as e:
                exception_holder.append(e)
                cancel.set()
            finally:
                _safe_put(prepare_queue, None)  # sentinel

        # ── Stage 3 – Upload thread: ES bulk insert ──────────────────────
        def uploader():
            nonlocal total_chunks, rows_processed
            try:
                while True:
                    item, ok = _safe_get(upload_queue)
                    if not ok or item is None:
                        break

                    texts, embeddings, metadatas, batch_rows = item
                    n = len(texts)
                    logger.info(f"[Upload] {n:,} docs → ES")

                    ES_CHUNK_SIZE = 5000

                    for i in range(0, n, ES_CHUNK_SIZE):
                        sub_texts = texts[i:i+ES_CHUNK_SIZE]
                        sub_embs  = embeddings[i:i+ES_CHUNK_SIZE]
                        sub_meta  = metadatas[i:i+ES_CHUNK_SIZE]

                        store.add_embeddings(
                            text_embeddings=list(zip(sub_texts, sub_embs)),
                            metadatas=sub_meta,
                            refresh_indices=False,
                        )

                    total_chunks += n
                    rows_processed += batch_rows
                    pbar.update(batch_rows)
                    logger.info(f"[Upload] ✓ +{n:,} chunks / {batch_rows:,} rows (cumulative: {total_chunks:,} chunks, {rows_processed:,} rows)")

                    del texts, embeddings, metadatas
                    gc.collect()

            except Exception as e:
                exception_holder.append(e)
                cancel.set()

        # ── Start threads ────────────────────────────────────────────────
        producer_thread = threading.Thread(target=producer, name="Prepare")
        upload_thread   = threading.Thread(target=uploader, name="Upload")

        pbar = tqdm(
            total=rows_to_index,
            desc="Indexing",
            unit="row",
            disable=not progress_bar,
        )

        producer_thread.start()
        upload_thread.start()

        # ── Stage 2 – Main thread: embed (GPU-bound) ────────────────────
        try:
            while True:
                item, ok = _safe_get(prepare_queue)
                if not ok or item is None:
                    break

                chunk_batch, chunk_batch_rows = item
                texts     = [doc.page_content for doc in chunk_batch]
                metadatas = [doc.metadata     for doc in chunk_batch]
                del chunk_batch

                n_rows = chunk_batch_rows
                logger.info(f"[Embed] {len(texts):,} chunks...")
                embeddings = self._embeddings.embed_documents(texts)

                # Hand off to upload thread (non-blocking unless queue full)
                if not _safe_put(upload_queue, (texts, embeddings, metadatas, n_rows)):
                    break

        except Exception as e:
            exception_holder.append(e)
            cancel.set()
        finally:
            _safe_put(upload_queue, None)  # sentinel for uploader

        # ── Wait for all stages ──────────────────────────────────────────
        producer_thread.join()
        upload_thread.join()
        pbar.close()

        # Final refresh
        try:
            store.client.indices.refresh(index=collection_name)
        except Exception:
            logger.warning("Could not refresh index — may take a moment to be searchable")

        if exception_holder:
            raise exception_holder[0]

        logger.info(f"\n=== Indexing Complete ===")
        logger.info(f"Total chunks indexed: {total_chunks:,} (from {total_rows:,} rows)")
        return store, total_chunks

    def index_from_dataframe(
        self,
        df: pd.DataFrame,
        text_field: str,
        html_field: str | None = None,
        *,
        metadata_fields: Sequence[str] | None = None,
        output_dir: Path | None = None,
        collection_name: str = None,
        progress_bar: bool | None = None,
    ) -> tuple[ElasticsearchStore]:
        raise NotImplementedError()

    def _validate_store(self, index) -> ElasticsearchStore:
        if not isinstance(index, ElasticsearchStore):
            raise ValueError("Valid ElasticsearchStore required")
        return index

    def delete_index(self, index: ElasticsearchStore) -> None:
        """Delete an Elasticsearch index."""
        if not self._current_index_name:
            raise ValueError("No index loaded")
        index.client.indices.delete(index=self._current_index_name)
        logger.info(f"Deleted '{self._current_index_name}'")

    def add_documents(
        self, index: ElasticsearchStore, documents: Sequence[Document]
    ) -> ElasticsearchStore:
        """Add documents to an existing index."""
        index.add_documents(list(documents))
        return index
    
    @contextmanager
    def _strategy_context(self, strategy: str | None):
        """Temporarily switch retrieval strategy."""
        if strategy is None or strategy == self.strategy:
            yield self.strategy
            return

        original_strategy = self.strategy
        original_retrieval_obj = self._retrieval_strategy
        original_embeddings = self._embeddings

        try:
            self.strategy = strategy
            if strategy == "bm25":
                self._retrieval_strategy = BM25Strategy()
                self._embeddings = None
            elif strategy in ["vector", "approximation", "hybrid"]:
                if self._embeddings is None:
                    raise ValueError(f"Embeddings required for '{strategy}' strategy")

                if strategy == "vector":
                    self._retrieval_strategy = DenseVectorScriptScoreStrategy()
                elif strategy == "approximation":
                    self._retrieval_strategy = DenseVectorStrategy()
                elif strategy == "hybrid":
                    self._retrieval_strategy = DenseVectorStrategy(hybrid=True)
            yield strategy
        finally:
            self.strategy = original_strategy
            self._retrieval_strategy = original_retrieval_obj
            self._embeddings = original_embeddings

    @staticmethod
    def _make_num_candidates_query(num_candidates: int):
        """Return a custom_query callback that overrides kNN num_candidates."""
        def _custom_query(query_body: dict, query_str: str | None) -> dict:
            if "knn" in query_body:
                query_body["knn"]["num_candidates"] = num_candidates
            return query_body
        return _custom_query

    def retrieve_documents(
        self,
        text: str,
        top_k: int = 5,
        strategy: str | None = None,
        num_candidates: int | None = None,
    ) -> list[Document]:
        """Retrieve documents using the configured strategy."""

        with self._strategy_context(strategy):
            index = self._create_store(self._current_index_name)
            custom_query = self._make_num_candidates_query(num_candidates) if num_candidates else None

            return index.similarity_search(self._prepare_query(text), top_k, custom_query=custom_query)

    def retrieve_documents_with_scores(
        self,
        text: str,
        top_k: int = 5,
        strategy: str | None = None,
        num_candidates: int | None = None,
    ) -> list[tuple[Document, float]]:
        """Retrieve documents with relevance scores."""
        with self._strategy_context(strategy):
            index = self._create_store(self._current_index_name)
            custom_query = self._make_num_candidates_query(num_candidates) if num_candidates else None

            return index.similarity_search_with_score(self._prepare_query(text), top_k, custom_query=custom_query)

    def batch_retrieve(
        self,
        questions: list[str],
        *,
        top_k: int = 5,
        strategy: str | None = None,
        num_candidates: int | None = None,
        progress_bar: bool = True,
    ) -> list[list[tuple[Document, float]]]:
        """Retrieve for multiple queries, optionally switching strategy."""

        with self._strategy_context(strategy) as strat:
            store = self._create_store(self._current_index_name)
            custom_query = self._make_num_candidates_query(num_candidates) if num_candidates else None
            results = []
            pdbar = tqdm(total=len(questions), desc=f"Retrieving ({strat})", unit="q", disable=not progress_bar)
            
            for question in questions:
                prepared = self._prepare_query(question)
                answer = store.similarity_search_with_score(prepared, top_k=top_k, custom_query=custom_query)
                results.append(answer)
                pdbar.update(1)

            pdbar.close()
            return results

    def get_store(self) -> ElasticsearchStore:
        """
        Return the underlying ElasticsearchStore for direct access.
        """
        if not self._current_index_name:
            raise ValueError("No index loaded. Use load_index() first.")
        return self._create_store(self._current_index_name)

    def retrieve_all_documents(
        self,
        batch_size: int = 1000,
        progress_bar: bool = True,
    ) -> list[Document]:
        """
        Retrieve all documents from the current index.
        
        Args:
            batch_size: Number of documents to fetch per scroll request
            progress_bar: Whether to show progress bar
            
        Returns:
            List of all documents in the index
        """
        if not self._current_index_name:
            raise ValueError("No index loaded. Use load_index() first.")
        
        store = self._create_store(self._current_index_name)
        es_client = store.client
        
        # Get total document count
        count_response = es_client.count(index=self._current_index_name)
        total_docs = count_response['count']
        
        logger.info(f"Retrieving {total_docs:,} documents from '{self._current_index_name}'")
        
        all_documents = []
        
        # Initialize scroll
        response = es_client.search(
            index=self._current_index_name,
            body={"query": {"match_all": {}}, "size": batch_size},
            scroll='5m'
        )
        
        scroll_id = response['_scroll_id']
        hits = response['hits']['hits']
        
        pbar = tqdm(total=total_docs, desc="Fetching documents", unit="doc") if progress_bar else None
        
        while hits:
            for hit in hits:
                doc = Document(
                    page_content=hit['_source'].get('text', ''),
                    metadata={
                        '_id': hit['_id'],
                        **{k: v for k, v in hit['_source'].items() if k != 'text'}
                    }
                )
                all_documents.append(doc)
            
            if pbar:
                pbar.update(len(hits))
            
            # Get next batch
            response = es_client.scroll(scroll_id=scroll_id, scroll='5m')
            scroll_id = response['_scroll_id']
            hits = response['hits']['hits']
        
        # Clear scroll
        es_client.clear_scroll(scroll_id=scroll_id)
        
        if pbar:
            pbar.close()
        
        logger.info(f"Retrieved {len(all_documents):,} documents")
        return all_documents
