"""Elasticsearch RAG service — vector, BM25, and hybrid retrieval."""

from __future__ import annotations

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
from langchain_elasticsearch.vectorstores import BM25Strategy, DenseVectorStrategy
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
        strategy: Literal["vector", "bm25", "hybrid"] = "vector",
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
            self._retrieval_strategy = BM25Strategy()
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
            self._retrieval_strategy = DenseVectorStrategy(hybrid=(strategy == "hybrid"))

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

    def _prepare_documents(self, documents: list[Document]) -> list[Document]:
        """Chunk documents and apply embedding prompt."""
        if self.config.chunk_size:
            documents = split_documents(
                documents, self.config.chunk_size, self.config.chunk_overlap
            )
        if self._embeddings is None:
            return documents
        return [
            Document(
                page_content=self._passage_prompt.format(passage=d.page_content),
                metadata=d.metadata,
            )
            for d in documents
        ]

    def _create_store(self, index_name: str) -> ElasticsearchStore:
        """Create an ElasticsearchStore instance."""
        kwargs: dict = {
            "es_url": self.es_url,
            "index_name": index_name,
            "strategy": self._retrieval_strategy,
        }
        if self._embeddings:
            kwargs["embedding"] = self._embeddings
            if getattr(self, "distance_strategy", None):
                kwargs["distance_strategy"] = self.distance_strategy
        if self.es_user:
            kwargs["es_user"] = self.es_user
        if self.es_password:
            kwargs["es_password"] = self.es_password
        return ElasticsearchStore(**kwargs)

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
        pbar = tqdm(total=len(documents), desc="ES insert (BM25)", unit="doc") if show_progress else None

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            store.add_documents(batch)
            indexed += len(batch)
            if pbar:
                pbar.update(len(batch))

        if pbar:
            pbar.close()
        logger.info(f"Indexed {indexed:,} in {time.time() - start:.1f}s")
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
    ) -> tuple[ElasticsearchStore, int]:
        
        if collection_name is None:
            raise ValueError("collection_name is required for indexing")
        
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
        
        store = self._create_store(collection_name)
        total_chunks = 0
        
        pbar = tqdm(total=None, desc="Streaming parquet", unit="batch", disable=not progress_bar)
        parquet_file = pq.ParquetFile(parquet_path)
        
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=[text_field] + list(metadata_fields or [])):
            df = batch.to_pandas()
            documents = documents_from_dataframe(df, text_field, metadata_fields)

            cleaned_documents = self._prepare_documents(documents)
            
            store.add_documents(cleaned_documents)

            total_chunks += len(documents)
            pbar.update(1)
        
        pbar.close()
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
        pass


    def load_index(
        self, output_dir: Path, collection_name: str = "rag"
    ) -> Optional[ElasticsearchStore]:
        """Connect to an existing Elasticsearch index."""
        store = self._create_store(collection_name)
        self._current_index_name = collection_name
        try:
            store.client.indices.get(index=collection_name)
            logger.info(f"Connected to '{collection_name}'")
            return store
        except Exception as e:
            logger.warning(f"Index '{collection_name}' not found: {e}")
            return None

    def _validate_store(self, index) -> ElasticsearchStore:
        if not isinstance(index, ElasticsearchStore):
            raise ValueError("Valid ElasticsearchStore required")
        return index

    def retrieve_documents(
        self,
        index: VectorStoreLike | None,
        text: str,
        *,
        top_k: int = 5,
        custom_query: CustomQueryFn = None,
    ) -> list[Document]:
        """Retrieve documents using the configured strategy."""
        return self._validate_store(index).similarity_search(
            self._prepare_query(text), k=top_k, custom_query=custom_query,
        )

    def retrieve_documents_with_scores(
        self,
        index: VectorStoreLike | None,
        text: str,
        *,
        top_k: int = 5,
        custom_query: CustomQueryFn = None,
    ) -> list[tuple[Document, float]]:
        """Retrieve documents with relevance scores."""
        return self._validate_store(index).similarity_search_with_score(
            self._prepare_query(text), k=top_k, custom_query=custom_query,
        )

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

    def batch_retrieve(
        self,
        index: ElasticsearchStore,
        questions: list[str],
        *,
        top_k: int = 5,
        strategy: str | None = None,
        progress_bar: bool = True,
        batch_size: int = 32,
    ) -> list[list[tuple[Document, float]]]:
        """Retrieve for multiple queries, optionally switching strategy."""
        if not self._current_index_name:
            raise ValueError("No index loaded")

        with self._strategy_context(strategy) as strat:
            store = self._create_store(self._current_index_name)
            results = []
            it = tqdm(questions, desc="Retrieving", unit="q") if progress_bar else questions
            for q in it:
                if strat == "hybrid":
                    results.append(
                        [(d, 0.0) for d in self.retrieve_documents(store, q, top_k=top_k)]
                    )
                else:
                    results.append(
                        self.retrieve_documents_with_scores(store, q, top_k=top_k)
                    )
            return results