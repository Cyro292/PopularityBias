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
from langchain_elasticsearch.vectorstores import DenseVectorStrategy, DenseVectorScriptScoreStrategy

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
        request_timeout: int = 600,
    ):
        self.es_url = es_url
        self.es_user = es_user
        self.es_password = es_password
        self.strategy = strategy
        self._current_index_name: str | None = None
        self._request_timeout: int = request_timeout

        # Load prompt templates (use config overrides if provided)
        passage_file = getattr(config, "passage_prompt_file", None) if config else None
        query_file = getattr(config, "query_prompt_file", None) if config else None
        self._passage_prompt, self._query_prompt = self._load_prompts(
            passage_prompt_file=passage_file,
            query_prompt_file=query_file,
        )

        if strategy == "bm25":
            self.config = config or IndexingConfig()
            self._embeddings = None
            self._retrieval_strategy = ElasticsearchStore.BM25RetrievalStrategy()
        else:
            if config is None:
                raise ValueError(f"IndexingConfig required for '{strategy}' strategy")
            self.config = config
            self._embeddings = build_embeddings(
                provider=config.embedding_provider,
                model=config.embedding_model,
                request_batch_size=config.request_batch_size,
                gpu_batch_size=config.gpu_batch_size,
                normalise_embeddings=config.normalise_embeddings,
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
    def _load_prompts(
        passage_prompt_file: str | None = None,
        query_prompt_file: str | None = None,
    ) -> tuple[str, str]:
        """Load passage and query prompt templates from disk.

        If explicit file paths are given they take priority, otherwise
        the default files under ``data/prompts/`` are used.
        """
        from config import DATA_DIR
        prompts_dir = Path(DATA_DIR) / "prompts"

        def _read(override_path: str | None, fallback_name: str, default: str) -> str:
            if override_path:
                p = Path(override_path)
                if p.exists():
                    logger.info(f"Using custom prompt: {p}")
                    return p.read_text().strip()
                logger.warning(f"Custom prompt {p} not found, falling back to default")
            path = prompts_dir / fallback_name
            if path.exists():
                return path.read_text().strip()
            logger.warning(f"Prompt {path} not found, using built-in default")
            return default

        return (
            _read(passage_prompt_file, "embeding_promt.txt", "passage: {passage}"),
            _read(query_prompt_file, "query_promt.txt", "query: {query}"),
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

    def _create_store(self, index_name: str, request_timeout: int | None = None) -> ElasticsearchStore:
        """Create an ElasticsearchStore instance."""

        if index_name is None:
            raise ValueError("No index loaded")

        kwargs: dict = {
            "es_url": self.es_url,
            "index_name": index_name,
            "strategy": self._retrieval_strategy,
            "es_user": self.es_user,
            "es_password": self.es_password,
            "es_params": {"request_timeout": request_timeout or self._request_timeout},
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
        end_row: Optional[int] = None,
        es_upload_batch: int = 2000,
        num_uploaders: int = 4,
        use_send_mode: bool = False,
    ) -> tuple[ElasticsearchStore, int]:
        ES_UPLOAD_BATCH = es_upload_batch
        NUM_UPLOADERS = num_uploaders

        if collection_name is None:
            raise ValueError("collection_name is required")
        if not parquet_path.exists():
            raise FileNotFoundError(parquet_path)
        if self._embeddings is None:
            raise ValueError("Embeddings required for vector indexing")

        # ── Pre-create the index ONCE before any thread starts ───────────
        # This prevents the race condition where multiple uploader threads
        # all call _create_index_if_not_exists() simultaneously and crash
        # with resource_already_exists_exception.
        bootstrap_store = self._create_store(collection_name)
        try:
            bootstrap_store._store._create_index_if_not_exists()
            logger.info(f"[Init] Index '{collection_name}' ready")
        except Exception as e:
            # Already exists from a previous partial run — that's fine
            if "resource_already_exists" in str(e).lower():
                logger.info(f"[Init] Index '{collection_name}' already exists, resuming")
            else:
                raise

        state = {"chunks": 0}          # mutable dict — no nonlocal needed
        chunks_lock = threading.Lock()

        parquet_file = pq.ParquetFile(parquet_path)
        total_rows = parquet_file.metadata.num_rows
        columns = [text_field] + list(metadata_fields or [])

        logger.info(f"Indexing {total_rows - skip_rows:,} rows (skip {skip_rows:,}) | "
                     f"batch={batch_size:,} | "
                     f"{'embed+send (Modal→ES direct)' if use_send_mode else 'embed→upload'}")

        # Stage sizes — tune these to keep each stage always busy
        prepare_queue: queue.Queue = queue.Queue(maxsize=3)   # Producer runs up to 4 batches ahead
        upload_queue:  queue.Queue = queue.Queue(maxsize=3)   # Used only in standard mode
        progress_queue: queue.Queue = queue.Queue()
        errors: list[Exception] = []
        cancel = threading.Event()

        # ── Stage 1: Producer ────────────────────────────────────────────
        # Reads parquet → chunks → puts (texts, metadatas) on prepare_queue
        def producer():
            try:
                rows_seen = 0
                for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
                    if cancel.is_set():
                        break
                    batch_len = batch.num_rows
                    if rows_seen + batch_len <= skip_rows:
                        rows_seen += batch_len
                        continue
                    df = batch.to_pandas()
                    if rows_seen < skip_rows:
                        df = df.iloc[skip_rows - rows_seen:]
                    if end_row is not None and rows_seen >= end_row:
                        break
                    rows_seen += batch_len

                    n_rows = len(df)
                    documents = documents_from_dataframe(df, text_field, metadata_fields)
                    chunks = self._prepare_documents(documents, log_details=False)
                    del documents, df

                    texts     = [doc.page_content for doc in chunks]
                    metadatas = [doc.metadata     for doc in chunks]
                    del chunks

                    prepare_queue.put((texts, metadatas, n_rows))
                    logger.info(f"[Produce] +{n_rows:,} rows → {len(texts):,} chunks queued")
                    gc.collect()
            except Exception as e:
                errors.append(e)
                cancel.set()
            finally:
                prepare_queue.put(None)  # sentinel

        # ── Stage 2a: Embedder — send mode (Modal → ES direct) ───────────
        # Embeds on GPU and pushes to ES without returning vectors to the client.
        # Progress is updated immediately after each batch completes.
        def embedder_send():
            try:
                while True:
                    item = prepare_queue.get()
                    if item is None:
                        break
                    texts, metadatas, n_rows = item

                    logger.info(f"[Embed+Send] {len(texts):,} chunks → Modal (direct to ES) …")
                    for attempt in range(6):
                        try:
                            indexed = self._embeddings.embed_and_send_documents(
                                texts, metadatas,
                                es_url=self.es_url,
                                index_name=collection_name,
                                strategy=self.strategy,
                                distance_strategy=getattr(self, "distance_strategy", None),
                                es_user=self.es_user,
                                es_password=self.es_password,
                                request_timeout=self._request_timeout,
                            )
                            break
                        except Exception as err:
                            if attempt >= 5:
                                raise
                            delay = [5, 15, 30, 60, 120][min(attempt, 4)]
                            logger.warning(f"[Embed+Send] ⚠ {type(err).__name__}: {err}, "
                                           f"retry {attempt+1}/5 in {delay}s")
                            time.sleep(delay)
                    logger.info(f"[Embed+Send] ✓ {indexed:,} docs indexed")

                    with chunks_lock:
                        state["chunks"] += indexed
                    progress_queue.put(n_rows)
                    gc.collect()

            except Exception as e:
                errors.append(e)
                cancel.set()

        # ── Stage 2b: Embedder — standard mode (embed → upload_queue) ────
        def embedder_standard():
            try:
                while True:
                    item = prepare_queue.get()
                    if item is None:
                        break
                    texts, metadatas, n_rows = item

                    logger.info(f"[Embed] {len(texts):,} chunks → Modal …")
                    for attempt in range(6):
                        try:
                            embeddings = self._embeddings.embed_documents(texts)
                            break
                        except Exception as err:
                            if attempt >= 5:
                                raise
                            delay = [5, 15, 30, 60, 120][min(attempt, 4)]
                            logger.warning(f"[Embed] ⚠ {type(err).__name__}: {err}, "
                                           f"retry {attempt+1}/5 in {delay}s")
                            time.sleep(delay)
                    logger.info(f"[Embed] ✓ {len(texts):,} done → upload_queue")

                    while True:
                        try:
                            upload_queue.put((texts, embeddings, metadatas, n_rows), timeout=1)
                            break
                        except queue.Full:
                            continue

                    del embeddings
                    gc.collect()

            except Exception as e:
                errors.append(e)
                cancel.set()
            finally:
                upload_queue.put(None)  # sentinel for uploader

        # ── Stage 3: Uploader (standard mode only) ────────────────────────
        def uploader():
            stores = [self._create_store(collection_name) for _ in range(NUM_UPLOADERS)]

            def _send(worker, pairs, metas):
                """Send one sub-batch with retry."""
                for attempt in range(6):
                    try:
                        stores[worker].add_embeddings(
                            text_embeddings=pairs, metadatas=metas, refresh_indices=False,
                        )
                        return len(pairs)
                    except Exception as err:
                        if attempt >= 5:
                            raise
                        delay = [5, 10, 30, 60, 120][min(attempt, 4)]
                        logger.warning(f"[Upload][w{worker}] ⚠ {type(err).__name__}: {err}, "
                                       f"retry {attempt+1}/5 in {delay}s")
                        time.sleep(delay)
                        stores[worker] = self._create_store(collection_name)

            try:
                pool = ThreadPoolExecutor(max_workers=NUM_UPLOADERS, thread_name_prefix="ES")
                while True:
                    try:
                        item = upload_queue.get(timeout=2)
                    except queue.Empty:
                        continue
                    if item is None:
                        break

                    texts, embeddings, metadatas, n_rows = item
                    n = len(texts)
                    logger.info(f"[Upload] {n:,} chunks received → uploading with {NUM_UPLOADERS} workers …")

                    futs = []
                    for idx, i in enumerate(range(0, n, ES_UPLOAD_BATCH)):
                        end = min(i + ES_UPLOAD_BATCH, n)
                        pairs = list(zip(texts[i:end], embeddings[i:end]))
                        metas = metadatas[i:end]
                        worker = idx % NUM_UPLOADERS
                        futs.append((end - i, n_rows, n, pool.submit(_send, worker, pairs, metas)))

                    for batch_n, nr, ntot, fut in futs:
                        fut.result()
                        with chunks_lock:
                            state["chunks"] += batch_n
                        progress_queue.put(round(batch_n * nr / ntot))

                    logger.info(f"[Upload] Batch of {n:,} chunks fully uploaded")
                    del texts, embeddings, metadatas
                    gc.collect()

                pool.shutdown(wait=True)
            except Exception as e:
                logger.error(f"[Upload] FAILED: {e}", exc_info=True)
                errors.append(e)

        # ── Start pipeline ───────────────────────────────────────────────
        threads = [threading.Thread(target=producer, name="Producer", daemon=True)]
        if use_send_mode:
            # embed+send: no uploader thread needed
            threads.append(threading.Thread(target=embedder_send, name="Embedder", daemon=True))
        else:
            threads.append(threading.Thread(target=embedder_standard, name="Embedder", daemon=True))
            threads.append(threading.Thread(target=uploader, name="Uploader", daemon=True))
        for t in threads:
            t.start()

        pbar = tqdm(total=total_rows, initial=skip_rows, desc="Indexing", unit="row", disable=not progress_bar)
        while any(t.is_alive() for t in threads):
            try:
                pbar.update(progress_queue.get(timeout=0.5))
            except queue.Empty:
                pass
        while not progress_queue.empty():
            pbar.update(progress_queue.get())
        for t in threads:
            t.join()
        pbar.close()

        try:
            bootstrap_store.client.indices.refresh(index=collection_name)
        except Exception as e:
            logger.error(f"Refresh failed: {e}")

        if errors:
            raise errors[0]

        logger.info(f"=== Done: {state['chunks']:,} chunks indexed ===")
        return bootstrap_store, state["chunks"]

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
                self._retrieval_strategy = ElasticsearchStore.BM25RetrievalStrategy()
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
        with self._strategy_context(strategy) as strat:
            index = self._create_store(self._current_index_name)
            custom_query = self._make_num_candidates_query(num_candidates) if num_candidates and strat != "bm25" else None
            return index.similarity_search(self._prepare_query(text), top_k, custom_query=custom_query)

    def retrieve_documents_with_scores(
        self,
        text: str,
        top_k: int = 5,
        strategy: str | None = None,
        num_candidates: int | None = None,
    ) -> list[tuple[Document, float]]:
        """Retrieve documents with relevance scores."""
        with self._strategy_context(strategy) as strat:
            index = self._create_store(self._current_index_name)
            custom_query = self._make_num_candidates_query(num_candidates) if num_candidates and strat != "bm25" else None
            return index.similarity_search_with_score(self._prepare_query(text), top_k, custom_query=custom_query)

    def batch_retrieve(
        self,
        questions: list[str],
        *,
        top_k: int = 5,
        strategy: str | None = None,
        num_candidates: int | None = None,
        progress_bar: bool = True,
        embed_batch_size: int = 512,
        search_workers: int = 16,
        request_timeout: int | None = None,
        msearch_batch_size: int = 100,
    ) -> list[list[tuple[Document, float]]]:
        """Retrieve for multiple queries using batched _msearch (vector) or threaded BM25.

        BM25: queries run in threads via similarity_search_with_score.
        Vector/approximation: queries are batch-embedded first, then sent to ES
        via _msearch in batches of `msearch_batch_size` — this sends many kNN
        queries in a single HTTP request, dramatically cutting network overhead
        compared to one-request-per-query threading.
        Set msearch_batch_size=0 to fall back to the old threaded approach.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with self._strategy_context(strategy) as strat:
            is_bm25 = strat == "bm25" or self._embeddings is None
            prepared = [self._prepare_query(q) for q in questions]
            n = len(prepared)
            timeout = request_timeout or self._request_timeout

            # ── Stage 1: batch-embed (vector only) ────────────────────────────
            if not is_bm25:
                logger.info(f"[Embed] Embedding {n:,} queries (batch_size={embed_batch_size})...")
                vectors: list[list[float]] = []
                embeddings = self._embeddings.embed_documents(prepared)
                vectors.extend(embeddings)
                logger.info(f"[Embed] ✓ {len(vectors):,} vectors ready")

            store = self._create_store(self._current_index_name, request_timeout=timeout)

            # ── Stage 2a: vector — _msearch batching ─────────────────────────
            if not is_bm25 and msearch_batch_size > 0:
                nc = num_candidates or (top_k * 10)
                results: list[list[tuple[Document, float]] | None] = [None] * n

                def _hits_to_docs(hits: list[dict]) -> list[tuple[Document, float]]:
                    out = []
                    for hit in hits:
                        src = hit.get("_source", {})
                        doc = Document(
                            page_content=src.get("text", ""),
                            metadata=src.get("metadata", {}),
                        )
                        out.append((doc, float(hit.get("_score") or 0.0)))
                    return out

                def _run_msearch_batch(batch_indices: list[int]) -> list[tuple[int, list[tuple[Document, float]]]]:
                    body: list[dict] = []
                    for i in batch_indices:
                        body.append({"index": self._current_index_name})
                        body.append({
                            "knn": {
                                "field": "vector",
                                "query_vector": vectors[i],
                                "k": top_k,
                                "num_candidates": nc,
                            },
                            "size": top_k,
                            "_source": True,
                        })
                    resp = store.client.msearch(body=body, request_timeout=timeout)
                    out = []
                    for idx, response in zip(batch_indices, resp["responses"]):
                        hits = response.get("hits", {}).get("hits", [])
                        out.append((idx, _hits_to_docs(hits)))
                    return out

                # Split into batches and process concurrently
                batches = [
                    list(range(start, min(start + msearch_batch_size, n)))
                    for start in range(0, n, msearch_batch_size)
                ]
                n_workers = max(1, min(search_workers, len(batches)))

                with ThreadPoolExecutor(max_workers=n_workers) as pool:
                    futures = {pool.submit(_run_msearch_batch, b): b for b in batches}
                    with tqdm(total=n, desc=f"Retrieving ({strat})", unit="q", disable=not progress_bar) as pbar:
                        for fut in as_completed(futures):
                            for idx, hits in fut.result():
                                results[idx] = hits
                            pbar.update(len(futures[fut]))

                return results  # type: ignore[return-value]

            # ── Stage 2b: BM25 (or msearch disabled) — threaded ──────────────
            custom_query = self._make_num_candidates_query(num_candidates) if num_candidates and not is_bm25 else None
            results = [None] * n

            def _search(idx: int):
                if is_bm25:
                    return idx, store.similarity_search_with_score(prepared[idx], k=top_k)
                return idx, store.similarity_search_by_vector_with_relevance_scores(
                    vectors[idx], k=top_k, custom_query=custom_query
                )

            with ThreadPoolExecutor(max_workers=search_workers) as pool:
                futures_map = {pool.submit(_search, i): i for i in range(n)}
                for fut in tqdm(as_completed(futures_map), total=n, desc=f"Retrieving ({strat})", unit="q", disable=not progress_bar):
                    idx, hits = fut.result()
                    results[idx] = hits

            return results  # type: ignore[return-value]

    def get_store(self) -> ElasticsearchStore:
        """
        Return the underlying ElasticsearchStore for direct access.
        """
        if not self._current_index_name:
            raise ValueError("No index loaded. Use load_index() first.")
        return self._create_store(self._current_index_name)
