"""Elasticsearch-based RAG service supporting both vector search and BM25.

This service provides a unified interface for:
- Dense vector search using embeddings (semantic similarity)
- BM25 keyword-based search (exact term matching)
- Hybrid retrieval combining both approaches

Elasticsearch offers significant advantages over separate Chroma + BM25 implementations:
- Single index for both search modes (no data duplication)
- Production-grade scalability and performance
- Native support for hybrid search strategies
- Advanced filtering and aggregation capabilities
- Can run locally via Docker or use Elastic Cloud

Installation:
    pip install -qU langchain-elasticsearch

Local Setup:
    # Quick-start Elasticsearch in Docker:
    curl -fsSL https://elastic.co/start-local | sh

    # Or manually with Docker:
    docker run -d --name elasticsearch \\
        -p 9200:9200 -p 9300:9300 \\
        -e "discovery.type=single-node" \\
        -e "xpack.security.enabled=false" \\
        docker.elastic.co/elasticsearch/elasticsearch:8.12.0

Reference: https://docs.langchain.com/oss/python/integrations/vectorstores/elasticsearch
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Literal, Optional, Sequence

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
from langchain.schema import Document
from langchain_elasticsearch import ElasticsearchStore
from langchain_elasticsearch.vectorstores import (
    ApproxRetrievalStrategy,
    BM25Strategy,
    DenseVectorStrategy,
)

from .base import RagService, VectorStoreLike
from .document_utils import documents_from_html_dataframe, documents_from_text_dataframe
from .utils import IndexingConfig, build_embeddings, split_documents

logger = logging.getLogger(__name__)

# Type alias for custom query function
CustomQueryFn = Optional[Callable[[dict, Optional[str]], dict]]


class ElasticsearchRagService(RagService):
    """RAG Service using Elasticsearch for both vector and BM25 search.

    This service provides a unified interface for multiple retrieval strategies:

    1. **Dense Vector Search** (default): Semantic similarity using embeddings
       - Best for: Natural language queries, concept matching, multi-lingual search
       - Requires: Embedding model (OpenAI, Google, HuggingFace)

    2. **BM25 Keyword Search**: Traditional full-text search
       - Best for: Exact term matching, technical terms, proper nouns
       - Requires: No embeddings needed

    3. **Hybrid Search**: Combines vector + BM25 for best of both worlds
       - Best for: Production applications needing robust retrieval
       - Requires: Embedding model

    Example - Vector Search:
        >>> config = IndexingConfig(embedding_provider="openai")
        >>> service = ElasticsearchRagService(
        ...     config=config,
        ...     es_url="http://localhost:9200",
        ...     strategy="vector"
        ... )
        >>> index, count = service.index_from_parquet(
        ...     parquet_path=Path("data.parquet"),
        ...     text_field="content",
        ...     output_dir=Path("./es_index")
        ... )
        >>> results = service.retrieve_documents(index, "your query", top_k=5)

    Example - BM25 Search (no embeddings):
        >>> service = ElasticsearchRagService(
        ...     es_url="http://localhost:9200",
        ...     strategy="bm25"
        ... )
        >>> index, count = service.index_from_parquet(
        ...     parquet_path=Path("data.parquet"),
        ...     text_field="content",
        ...     output_dir=Path("./es_index")
        ... )
        >>> results = service.retrieve_documents(index, "keyword query", top_k=5)

    Example - Hybrid Search:
        >>> config = IndexingConfig(embedding_provider="openai")
        >>> service = ElasticsearchRagService(
        ...     config=config,
        ...     es_url="http://localhost:9200",
        ...     strategy="hybrid"
        ... )
        >>> results = service.retrieve_documents(index, "query", top_k=5)
    """

    def __init__(
        self,
        config: IndexingConfig | None = None,
        *,
        es_url: str = "http://localhost:9200",
        es_user: str | None = None,
        es_password: str | None = None,
        strategy: Literal["vector", "bm25", "hybrid"] = "vector",
        distance_function: Literal["cosine", "dot_product", "euclidean"] | None = None,
    ):
        """Initialize Elasticsearch RAG service.

        Args:
            config: IndexingConfig for chunking and embeddings (required for vector/hybrid strategies).
            es_url: Elasticsearch connection URL.
            es_user: Optional Elasticsearch username.
            es_password: Optional Elasticsearch password.
            strategy: Retrieval strategy to use:
                - "vector": Dense vector search (requires embeddings)
                - "bm25": BM25 keyword search (no embeddings needed)
                - "hybrid": Combines vector + BM25
            distance_function: Distance metric for vector search:
                - "cosine": Cosine similarity (default, recommended)
                - "dot_product": Inner product (for normalized embeddings)
                - "euclidean": Euclidean distance

        Raises:
            ValueError: If vector/hybrid strategy is used without embeddings config.
        """
        self.es_url = es_url
        self.es_user = es_user
        self.es_password = es_password
        self.strategy = strategy
        self._current_index_name = None  # Store current index name

        # Load embedding prompt template
        self._load_embedding_prompt()

        # BM25 doesn't need embeddings or config
        if strategy == "bm25":
            self.config = config or IndexingConfig()
            self._embeddings = None
            self._retrieval_strategy = BM25Strategy()
            logger.info("Initialized Elasticsearch with BM25 strategy (no embeddings)")
        else:
            # Vector and hybrid strategies require embeddings
            if config is None:
                raise ValueError(f"IndexingConfig required for '{strategy}' strategy")

            self.config = config
            self._embeddings = build_embeddings(
                provider=self.config.embedding_provider,
                model=self.config.embedding_model,
                trust_remote_code=self.config.trust_remote_code,
                rate_limiter=self.config.rate_limiter,
                requests_per_second=self.config.requests_per_second,
                check_interval=self.config.rate_limit_check_interval,
                bucket_size=self.config.rate_limit_bucket_size,
            )

            # Map distance functions to Elasticsearch distance_strategy format
            distance = distance_function or self.config.distance_function
            
            if distance not in (None, "COSINE", "DOT_PRODUCT", "EUCLIDEAN_DISTANCE"):
                raise ValueError(
                    f"Invalid distance_function: {distance}. "
                    "Must be 'COSINE', 'DOT_PRODUCT', or 'EUCLIDEAN_DISTANCE'."
                )
            
            self.distance_strategy = distance

            # Create retrieval strategy (distance is set on ElasticsearchStore, not strategy)
            use_hybrid = (strategy == "hybrid")
            self._retrieval_strategy = DenseVectorStrategy(hybrid=use_hybrid)

            strategy_name = "hybrid vector+BM25" if use_hybrid else "dense vector"
            logger.info(
                f"Initialized Elasticsearch with {strategy_name} strategy "
                f"({self.distance_strategy} distance)"
            )

    def _load_embedding_prompt(self):
        """Load embedding prompt templates from files.

        The prompts are applied to document/query content before embedding to improve
        retrieval quality. Some embedding models (like E5, BGE) benefit from
        instructional prefixes.
        """
        from config import DATA_DIR

        # Load passage prompt (for documents)
        passage_prompt_file = Path(DATA_DIR) / "prompts" / "embeding_promt.txt"
        if passage_prompt_file.exists():
            self.embedding_prompt = passage_prompt_file.read_text().strip()
            logger.info(f"Loaded embedding prompt: {self.embedding_prompt[:50]}...")
        else:
            # Default prompt if file doesn't exist
            self.embedding_prompt = "passage: {passage}"
            logger.warning(f"Embedding prompt file not found at {passage_prompt_file}, using default")

        # Load query prompt (for retrieval)
        query_prompt_file = Path(DATA_DIR) / "prompts" / "query_promt.txt"
        if query_prompt_file.exists():
            self.query_prompt = query_prompt_file.read_text().strip()
            logger.info(f"Loaded query prompt: {self.query_prompt[:50]}...")
        else:
            # Default prompt if file doesn't exist
            self.query_prompt = "query: {query}"
            logger.warning(f"Query prompt file not found at {query_prompt_file}, using default")

    def _apply_embedding_prompt(self, documents: list[Document]) -> list[Document]:
        """Apply embedding prompt template to documents.

        Wraps each document's content with the embedding prompt template.
        This improves retrieval quality for models trained with instruction prefixes.

        Args:
            documents: Documents to apply prompt to.

        Returns:
            Documents with prompted content.
        """
        # Only apply if we're using embeddings (not BM25-only)
        if self._embeddings is None:
            return documents

        prompted_docs = []
        for doc in documents:
            # Apply prompt template
            prompted_content = self.embedding_prompt.format(passage=doc.page_content)

            # Create new document with prompted content
            prompted_docs.append(Document(
                page_content=prompted_content,
                metadata=doc.metadata.copy()
            ))

        return prompted_docs

    def _should_apply_query_prompt(self) -> bool:
        """Check if query prompt should be applied based on current strategy."""
        return self.strategy != "bm25" and self._embeddings is not None
    
    def _prepare_query(self, query: str) -> str:
        """Prepare query text for retrieval based on current strategy."""
        if self._should_apply_query_prompt():
            return self.query_prompt.format(query=query)
        return query

    def _prepare_documents(self, documents: list[Document]) -> list[Document]:
        """Apply chunking and embedding prompt if configured."""
        if self.config.chunk_size:
            documents = split_documents(
                documents,
                self.config.chunk_size,
                self.config.chunk_overlap
            )
        return self._apply_embedding_prompt(documents)

    @contextmanager
    def _strategy_context(self, strategy: str | None, distance_metric: str | None = None):
        """Context manager for temporarily switching retrieval strategy.
        
        Args:
            strategy: Strategy to switch to ("vector", "bm25", "hybrid"), or None to keep current.
            distance_metric: Optional distance metric for vector strategies.
            
        Yields:
            The target strategy name.
        """
        if strategy is None or strategy == self.strategy:
            yield self.strategy
            return
            
        # Save original state
        original = (self.strategy, self._retrieval_strategy, getattr(self, 'distance_strategy', None))
        
        try:
            logger.info(f"Switching from {self.strategy} to {strategy} strategy")
            self.strategy = strategy
            
            if strategy == 'bm25':
                self._retrieval_strategy = BM25Strategy()
            elif strategy in ('vector', 'hybrid'):
                if distance_metric:
                    distance_map = {"cosine": "COSINE", "dot_product": "DOT_PRODUCT", 
                                    "euclidean": "EUCLIDEAN_DISTANCE", "COSINE": "COSINE",
                                    "DOT_PRODUCT": "DOT_PRODUCT", "EUCLIDEAN_DISTANCE": "EUCLIDEAN_DISTANCE"}
                    self.distance_strategy = distance_map.get(distance_metric, distance_metric)
                self._retrieval_strategy = DenseVectorStrategy(hybrid=(strategy == 'hybrid'))
            else:
                raise ValueError(f"Invalid strategy: {strategy}. Must be 'vector', 'bm25', or 'hybrid'")
            
            yield strategy
        finally:
            # Restore original state
            self.strategy, self._retrieval_strategy, dist = original
            if dist is not None:
                self.distance_strategy = dist

    def _batch_index_documents(
        self,
        store: ElasticsearchStore,
        documents: list[Document],
        batch_size: int,
        show_progress: bool,
    ) -> int:
        """Index documents in batches with optional progress bar.
        
        Args:
            store: ElasticsearchStore instance.
            documents: Documents to index.
            batch_size: Size of each batch.
            show_progress: Whether to show progress bar.
            
        Returns:
            Number of documents indexed.
        """
        if not documents:
            return 0
            
        if not show_progress or len(documents) <= batch_size:
            store.add_documents(documents)
            return len(documents)
        
        indexed = 0
        with tqdm(total=len(documents), desc="Indexing documents", unit="docs") as pbar:
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                try:
                    store.add_documents(batch)
                    indexed += len(batch)
                    pbar.update(len(batch))
                except Exception as e:
                    logger.error(f"Error indexing batch {i}-{i+len(batch)}: {e}")
                    # Fallback: try one by one
                    for idx, doc in enumerate(batch):
                        try:
                            store.add_documents([doc])
                            indexed += 1
                            pbar.update(1)
                        except Exception as doc_error:
                            logger.error(f"Failed doc at {i+idx}: {doc_error}")
                            logger.error(f"Metadata: {doc.metadata}, Content: {doc.page_content[:200]}...")
                            raise
        return indexed

    def _create_store(
        self,
        index_name: str,
    ) -> ElasticsearchStore:
        """Create ElasticsearchStore instance.

        Args:
            index_name: Name for the Elasticsearch index.

        Returns:
            Configured ElasticsearchStore instance.
        """
        kwargs = {
            "es_url": self.es_url,
            "index_name": index_name,
            "strategy": self._retrieval_strategy,
        }

        # Add embeddings and distance_strategy for vector strategies
        if self._embeddings is not None:
            kwargs["embedding"] = self._embeddings
            # Add distance_strategy for vector-based retrieval
            if hasattr(self, 'distance_strategy'):
                kwargs["distance_strategy"] = self.distance_strategy

        # Add credentials if provided
        if self.es_user:
            kwargs["es_user"] = self.es_user
        if self.es_password:
            kwargs["es_password"] = self.es_password

        return ElasticsearchStore(**kwargs)

    def index_from_parquet(
        self,
        parquet_path: Path,
        output_dir: Path,
        *,
        text_field: str | None = None,
        html_field: str | None = None,
        metadata_fields: Sequence[str] | None = None,
        collection_name: str = "rag",
        progress_bar: bool | None = None,
        batch_size: int = 500,
    ) -> tuple[ElasticsearchStore, int]:
        """Load text or HTML from Parquet and index in Elasticsearch."""
        if (text_field is None) == (html_field is None):
            raise ValueError("Exactly one of text_field or html_field must be provided")

        field = html_field or text_field
        meta_fields = tuple(metadata_fields or ())

        logger.info(f"Reading {parquet_path}")
        df = pq.read_table(parquet_path, columns=[field, *meta_fields]).to_pandas()
        
        return self.index_from_dataframe(
            df, 
            text_field=text_field or field,
            html_field=html_field,
            metadata_fields=metadata_fields,
            collection_name=collection_name,
            progress_bar=progress_bar,
            batch_size=batch_size,
        )

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
        batch_size: int = 500,
    ) -> tuple[ElasticsearchStore, int]:
        """Create an Elasticsearch index from a pandas DataFrame."""
        if html_field is not None:
            raise NotImplementedError("HTML field indexing from DataFrame is not implemented")

        metadata_fields_tuple = tuple(metadata_fields or ())
        logger.info(f"Indexing DataFrame with {len(df)} rows")

        # Create and prepare documents
        documents = documents_from_text_dataframe(
            df, text_field, metadata_fields_tuple, source="dataframe", row_offset=0,
        )
        documents = self._prepare_documents(documents)
        logger.info(f"Prepared {len(documents)} document chunks")

        # Create store and index
        store = self._create_store(collection_name)
        self._current_index_name = collection_name

        show_progress = self.config.use_progress if progress_bar is None else progress_bar
        num_indexed = self._batch_index_documents(store, documents, batch_size, show_progress)
        
        logger.info(f"Successfully indexed {num_indexed} chunks to '{collection_name}'")
        return store, num_indexed

    def load_index(
        self,
        output_dir: Path,
        collection_name: str = "rag"
    ) -> Optional[ElasticsearchStore]:
        """Load existing Elasticsearch index.

        Note: Elasticsearch indices are persistent by nature, so this simply
        reconnects to an existing index. The output_dir parameter is kept
        for API compatibility but not used.

        Args:
            output_dir: Directory path (not used for Elasticsearch).
            collection_name: Name of the Elasticsearch index to connect to.

        Returns:
            ElasticsearchStore instance connected to the existing index.
        """
        logger.info(f"Connecting to existing Elasticsearch index '{collection_name}'")
        store = self._create_store(collection_name)
        self._current_index_name = collection_name

        # Check if index exists by attempting a simple operation
        try:
            # This will raise an exception if the index doesn't exist
            store.client.indices.get(index=collection_name)
            logger.info(f"Successfully connected to index '{collection_name}'")
            return store
        except Exception as e:
            logger.warning(f"Could not connect to index '{collection_name}': {e}")
            return None

    def retrieve_documents(
        self,
        index: VectorStoreLike | None,
        text: str,
        *,
        top_k: int = 5,
        custom_query: CustomQueryFn = None,
    ) -> list[Document]:
        """Retrieve documents using the configured strategy."""
        if not index or not isinstance(index, ElasticsearchStore):
            raise ValueError("Valid ElasticsearchStore index is required")

        return index.similarity_search(
            self._prepare_query(text), 
            k=top_k,
            custom_query=custom_query,
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
        if not index or not isinstance(index, ElasticsearchStore):
            raise ValueError("Valid ElasticsearchStore index is required")

        return index.similarity_search_with_score(
            self._prepare_query(text),
            k=top_k,
            custom_query=custom_query,
        )

    def delete_index(self, index: ElasticsearchStore) -> None:
        """Delete an Elasticsearch index.

        Args:
            index: ElasticsearchStore instance to delete.
        """
        if not self._current_index_name:
            raise ValueError("No index name stored. Load or create an index first.")
        
        try:
            index.client.indices.delete(index=self._current_index_name)
            logger.info(f"Deleted index '{self._current_index_name}'")
        except Exception as e:
            logger.error(f"Failed to delete index '{self._current_index_name}': {e}")
            raise

    def add_documents(
        self,
        index: ElasticsearchStore,
        documents: Sequence[Document]
    ) -> ElasticsearchStore:
        """Add documents to an existing Elasticsearch index.

        Args:
            index: ElasticsearchStore instance.
            documents: Documents to add.

        Returns:
            Updated index.
        """
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
        """Retrieve documents for multiple queries in batch.

        Args:
            index: ElasticsearchStore instance.
            questions: List of query strings.
            top_k: Number of documents to return per query.
            strategy: Override retrieval strategy ("vector", "bm25", "hybrid").
            distance_metric: Distance metric for vector search.
            progress_bar: Whether to show progress bar.
            batch_size: Number of queries to process in each batch.

        Returns:
            List of retrieval results, one per query.
        """
        if not self._current_index_name:
            raise ValueError("No index name stored. Load or create an index first.")
        
        with self._strategy_context(strategy) as current_strategy:
            # Recreate store with correct strategy
            store = self._create_store(self._current_index_name)
            
            results = []
            iterator = tqdm(questions, desc="Retrieving", unit="query") if progress_bar else questions
            
            for question in iterator:
                if current_strategy == "hybrid":
                    docs = self.retrieve_documents(store, question, top_k=top_k)
                    result = [(doc, 0.0) for doc in docs]
                else:
                    result = self.retrieve_documents_with_scores(store, question, top_k=top_k)
                results.append(result)
            
            return results
