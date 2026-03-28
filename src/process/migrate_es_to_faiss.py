#!/usr/bin/env python3
"""Migrate documents from Elasticsearch to FAISS with memory-efficient streaming.

This script exports all documents from an Elasticsearch index and creates a
memory-efficient FAISS index. It supports:

  - Streaming export via Point-In-Time (PIT) pagination
  - Memory-efficient IVF_PQ indexing with streaming training
  - Checkpointing for resumable migrations
  - Optional re-embedding (if ES vectors are incompatible)
  - Deduplication during export

Usage:
    # Basic migration (re-embed documents)
    python scripts/migrate_es_to_faiss.py \
        --es-index wiki_full_l \
        --output-dir data/faiss_wiki \
        --strategy ivfpq

    # Migration with existing vectors (no re-embedding)
    python scripts/migrate_es_to_faiss.py \
        --es-index wiki_full_l \
        --output-dir data/faiss_wiki \
        --use-existing-vectors

    # Resume interrupted migration
    python scripts/migrate_es_to_faiss.py \
        --es-index wiki_full_l \
        --output-dir data/faiss_wiki \
        --resume

    # Low-RAM mode (for machines with < 8GB RAM)
    python scripts/migrate_es_to_faiss.py \
        --es-index wiki_full_l \
        --output-dir data/faiss_wiki \
        --low-ram
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterator

import dotenv
import numpy as np
from elasticsearch import Elasticsearch
from langchain.schema import Document
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import DATA_DIR
from src.rag.faiss_rag_service import FaissRagService, MemoryConfig
from src.rag.SqliteDocstore import SqliteDocstore
from src.rag.utils import IndexingConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment
dotenv.load_dotenv()


# === Configuration ===========================================================

@dataclass
class MigrationConfig:
    """Configuration for ES to FAISS migration."""

    # Elasticsearch source
    es_url: str = os.getenv("ELASTICSEARCH_ENDPOINT", "http://localhost:9200")
    es_user: str | None = os.getenv("ELASTICSEARCH_USERNAME")
    es_password: str | None = os.getenv("ELASTICSEARCH_PASSWORD")
    es_index: str = "wiki_full_l"

    # FAISS destination
    output_dir: Path = DATA_DIR / "faiss_migrated"
    strategy: str = "ivfpq"

    # Memory settings
    low_ram: bool = False
    max_ram_mb: int = 4096

    # Processing settings
    batch_size: int = 1000
    page_size: int = 1000
    pit_keep_alive: str = "10m"

    # Embedding settings
    use_existing_vectors: bool = False
    embedding_provider: str = "modal"
    embedding_model: str = "intfloat/multilingual-e5-large"
    gpu_batch_size: int = 64
    request_batch_size: int = 100
    normalise_embeddings: bool = True

    # Migration control
    resume: bool = False
    checkpoint_every: int = 10_000
    deduplicate: bool = True
    skip_docs: int = 0
    max_docs: int | None = None

    # IVF_PQ settings (tuned for ~4M docs with 1024-dim vectors)
    ivfpq_nlist: int = 4096
    ivfpq_m: int = 32  # Must divide embedding dim
    ivfpq_nbits: int = 8
    ivfpq_nprobe: int = 64


# === Elasticsearch Export ====================================================

def get_es_client(config: MigrationConfig) -> Elasticsearch:
    """Create an Elasticsearch client."""
    kwargs = {"request_timeout": 300}
    if config.es_user and config.es_password:
        kwargs["basic_auth"] = (config.es_user, config.es_password)
    return Elasticsearch(config.es_url, **kwargs)


def get_index_doc_count(client: Elasticsearch, index: str) -> int:
    """Get the total document count for an index."""
    try:
        stats = client.indices.stats(index=index)
        return stats["indices"][index]["primaries"]["docs"]["count"]
    except Exception:
        try:
            return client.count(index=index)["count"]
        except Exception:
            return 0


def stream_es_documents(
    client: Elasticsearch,
    config: MigrationConfig,
) -> Generator[dict, None, None]:
    """Stream all documents from ES using PIT + search_after.

    Yields dicts with keys: _id, text, metadata, vector (if present).
    """
    seen_hashes: set[str] = set() if config.deduplicate else None
    skipped = 0
    yielded = 0

    # Open Point-In-Time
    pit_resp = client.open_point_in_time(
        index=config.es_index,
        keep_alive=config.pit_keep_alive,
    )
    pit_id = pit_resp["id"]
    logger.info(f"Opened PIT: {pit_id[:40]}...")

    search_after: list | None = None
    page = 0

    try:
        while True:
            if config.max_docs and yielded >= config.max_docs:
                logger.info(f"Reached max_docs limit ({config.max_docs:,})")
                break

            body = {
                "size": config.page_size,
                "query": {"match_all": {}},
                "sort": [{"_shard_doc": "asc"}],
                "pit": {"id": pit_id, "keep_alive": config.pit_keep_alive},
                "fields": ["vector"],
            }
            if search_after is not None:
                body["search_after"] = search_after

            resp = client.search(body=body, _source=True)
            hits = resp["hits"]["hits"]

            if not hits:
                break

            # ES may rotate PIT id
            pit_id = resp["pit_id"]
            page += 1

            for doc in hits:
                if config.max_docs and yielded >= config.max_docs:
                    break

                source = doc["_source"]

                # Skip documents if resuming
                total_seen = (page - 1) * config.page_size + hits.index(doc)
                if total_seen < config.skip_docs:
                    continue

                # Deduplication
                if seen_hashes is not None:
                    raw_text = source.get("text", "")
                    wiki_id = str(source.get("metadata", {}).get("wikipedia_id", ""))
                    dedup_key = hashlib.md5(
                        f"{wiki_id}\x00{raw_text}".encode("utf-8", errors="replace")
                    ).hexdigest()

                    if dedup_key in seen_hashes:
                        skipped += 1
                        continue
                    seen_hashes.add(dedup_key)

                # Prefer vector from _source; fall back to fields response
                # (dense_vector may not be in _source depending on ES version
                # and index settings, but is always available via fields).
                vector = source.get("vector")
                if vector is None:
                    fields = doc.get("fields", {})
                    vector = fields.get("vector")

                yielded += 1
                yield {
                    "_id": doc["_id"],
                    "text": source.get("text", ""),
                    "metadata": source.get("metadata", {}),
                    "vector": vector,
                }

            search_after = hits[-1]["sort"]

            if page % 50 == 0:
                logger.info(
                    f"  Page {page:,} — exported {yielded:,} docs, "
                    f"skipped {skipped:,} duplicates"
                )

    finally:
        try:
            client.close_point_in_time(body={"id": pit_id})
            logger.info(f"Closed PIT. Total: {yielded:,} docs, {skipped:,} duplicates skipped")
        except Exception:
            pass


# === Migration State =========================================================

def load_migration_state(output_dir: Path) -> dict:
    """Load saved migration state for resuming."""
    state_file = output_dir / "migration_state.json"
    if state_file.exists():
        with open(state_file) as f:
            return json.load(f)
    return {"docs_processed": 0, "chunks_indexed": 0, "completed": False}


def save_migration_state(output_dir: Path, state: dict) -> None:
    """Save migration state for resuming."""
    output_dir.mkdir(parents=True, exist_ok=True)
    state_file = output_dir / "migration_state.json"
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)


# === Main Migration ==========================================================

def create_faiss_service(config: MigrationConfig) -> FaissRagService:
    """Create a configured FAISS service for migration."""
    # Memory configuration based on mode
    if config.low_ram:
        mem_config = MemoryConfig(
            max_ram_mb=2048,
            use_mmap=True,
            queue_maxsize=1,
            gc_every_n_batches=1,
            training_sample_size=100_000,
        )
        batch_size = 500
    else:
        mem_config = MemoryConfig(
            max_ram_mb=config.max_ram_mb,
            use_mmap=True,
            queue_maxsize=2,
            gc_every_n_batches=2,
            training_sample_size=500_000,
        )
        batch_size = config.batch_size

    # Indexing configuration
    indexing_config = IndexingConfig(
        embedding_provider=config.embedding_provider,
        embedding_model=config.embedding_model,
        chunk_size=None,  # Don't re-chunk, ES docs are already chunked
        chunk_overlap=0,
        gpu_batch_size=config.gpu_batch_size,
        request_batch_size=config.request_batch_size,
        normalise_embeddings=config.normalise_embeddings,
    )

    return FaissRagService(
        config=indexing_config,
        strategy=config.strategy,
        distance_strategy="cosine",
        memory_config=mem_config,
        ivfpq_nlist=config.ivfpq_nlist,
        ivfpq_m=config.ivfpq_m,
        ivfpq_nbits=config.ivfpq_nbits,
        ivfpq_nprobe=config.ivfpq_nprobe,
    )


def migrate_with_existing_vectors(
    config: MigrationConfig,
    es_client: Elasticsearch,
    total_docs: int,
) -> None:
    """Migrate using existing vectors from ES (no re-embedding).

    This is faster but requires that ES has stored vectors and they're
    compatible with the FAISS index configuration.
    """
    import faiss
    from src.rag.SqliteDocstore import SqliteDocstore

    logger.info("Migration mode: Using existing vectors from Elasticsearch")

    # Load state for resuming
    state = load_migration_state(config.output_dir) if config.resume else {}
    skip_docs = state.get("docs_processed", 0)
    if skip_docs > 0:
        logger.info(f"Resuming from document {skip_docs:,}")

    # Setup output directory
    faiss_dir = config.output_dir / "faiss"
    faiss_dir.mkdir(parents=True, exist_ok=True)

    # Initialize or load docstore
    docstore = SqliteDocstore(faiss_dir / "docstore.sqlite")

    # Accumulators for batch processing
    vectors_buffer: list[np.ndarray] = []
    texts_buffer: list[str] = []
    metadatas_buffer: list[dict] = []
    id_map: dict[int, str] = {}

    # FAISS index (created after we know dimension)
    faiss_index: faiss.Index | None = None
    dim: int | None = None
    total_indexed = skip_docs
    training_vectors: list[np.ndarray] = []

    # Stream documents
    config_copy = MigrationConfig(**vars(config))
    config_copy.skip_docs = skip_docs

    pbar = tqdm(
        total=total_docs,
        initial=skip_docs,
        desc="Migrating",
        unit="docs",
    )

    try:
        for doc in stream_es_documents(es_client, config_copy):
            vector = doc.get("vector")
            if vector is None:
                logger.warning(f"Document {doc['_id']} has no vector, skipping")
                continue

            # Convert to numpy
            vec = np.array(vector, dtype=np.float32)

            # Initialize index on first vector
            if faiss_index is None:
                dim = len(vec)
                logger.info(f"Detected embedding dimension: {dim}")

                # Validate IVF_PQ params
                if config.strategy == "ivfpq" and dim % config.ivfpq_m != 0:
                    # Auto-adjust m
                    for m in [48, 32, 64, 24, 16, 8]:
                        if dim % m == 0:
                            config.ivfpq_m = m
                            logger.info(f"Auto-adjusted ivfpq_m to {m} for dim={dim}")
                            break

                # Create index (will train later)
                if config.strategy == "ivfpq":
                    quantizer = faiss.IndexFlatL2(dim)
                    faiss_index = faiss.IndexIVFPQ(
                        quantizer, dim,
                        config.ivfpq_nlist,
                        config.ivfpq_m,
                        config.ivfpq_nbits,
                    )
                    faiss_index.nprobe = config.ivfpq_nprobe
                elif config.strategy == "hnsw":
                    faiss_index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_L2)
                else:
                    faiss_index = faiss.IndexFlatL2(dim)

            # Accumulate for training (IVF indexes need training)
            if not faiss_index.is_trained:
                training_vectors.append(vec)

                # Train when we have enough vectors
                min_train = config.ivfpq_nlist * 39
                if len(training_vectors) >= min_train:
                    logger.info(f"Training index on {len(training_vectors):,} vectors...")
                    train_data = np.vstack(training_vectors)
                    faiss_index.train(train_data)
                    logger.info("Training complete")

                    # Add training vectors to index
                    faiss_index.add(train_data)
                    for i, v in enumerate(training_vectors):
                        uid = f"train_{i}"
                        id_map[len(id_map)] = uid

                    del train_data, training_vectors
                    training_vectors = []
                    gc.collect()

                continue  # Don't add to buffer yet

            # Buffer vectors
            vectors_buffer.append(vec)
            texts_buffer.append(doc["text"])
            metadatas_buffer.append(doc["metadata"])

            # Process batch
            if len(vectors_buffer) >= config.batch_size:
                # Add to FAISS
                batch_vectors = np.vstack(vectors_buffer)
                start_idx = faiss_index.ntotal
                faiss_index.add(batch_vectors)

                # Add to docstore
                docs_to_add = {}
                for i, (text, meta) in enumerate(zip(texts_buffer, metadatas_buffer)):
                    uid = f"doc_{start_idx + i}"
                    id_map[start_idx + i] = uid
                    docs_to_add[uid] = Document(page_content=text, metadata=meta)
                docstore.add(docs_to_add)

                total_indexed += len(vectors_buffer)
                pbar.update(len(vectors_buffer))

                # Checkpoint
                if total_indexed % config.checkpoint_every == 0:
                    logger.info(f"Checkpoint at {total_indexed:,} docs")
                    faiss.write_index(faiss_index, str(faiss_dir / "index.faiss"))
                    docstore.save_id_map(id_map)
                    save_migration_state(config.output_dir, {
                        "docs_processed": total_indexed,
                        "completed": False,
                    })

                # Clear buffers
                vectors_buffer.clear()
                texts_buffer.clear()
                metadatas_buffer.clear()
                del batch_vectors
                gc.collect()

        # Process remaining vectors
        if vectors_buffer:
            batch_vectors = np.vstack(vectors_buffer)

            # Handle case where we never had enough to train
            if not faiss_index.is_trained:
                all_train = training_vectors + [v for v in vectors_buffer]
                train_data = np.vstack(all_train)
                logger.info(f"Training index on {len(train_data):,} vectors (final)...")
                faiss_index.train(train_data)
                faiss_index.add(train_data)
                # Update id_map for all
                for i in range(len(train_data)):
                    id_map[i] = f"doc_{i}"
                del train_data, all_train
            else:
                start_idx = faiss_index.ntotal
                faiss_index.add(batch_vectors)

                docs_to_add = {}
                for i, (text, meta) in enumerate(zip(texts_buffer, metadatas_buffer)):
                    uid = f"doc_{start_idx + i}"
                    id_map[start_idx + i] = uid
                    docs_to_add[uid] = Document(page_content=text, metadata=meta)
                docstore.add(docs_to_add)

            total_indexed += len(vectors_buffer)

    finally:
        pbar.close()

    # Final save
    logger.info(f"Saving final index with {faiss_index.ntotal:,} vectors...")
    faiss.write_index(faiss_index, str(faiss_dir / "index.faiss"))
    docstore.save_id_map(id_map)
    save_migration_state(config.output_dir, {
        "docs_processed": total_indexed,
        "completed": True,
    })

    logger.info(f"Migration complete! {total_indexed:,} documents indexed to {config.output_dir}")


def migrate_with_reembedding(
    config: MigrationConfig,
    es_client: Elasticsearch,
    total_docs: int,
) -> None:
    """Migrate by re-embedding documents (slower but more flexible).

    This uses the FaissRagService's indexing pipeline which handles
    batching, training, and checkpointing automatically.
    """
    logger.info("Migration mode: Re-embedding documents")

    # Create a temporary parquet file from ES documents
    # This allows us to use the existing index_from_parquet_batches method
    import pandas as pd

    temp_parquet = config.output_dir / "temp_es_export.parquet"
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Load state for resuming
    state = load_migration_state(config.output_dir) if config.resume else {}
    skip_docs = state.get("docs_processed", 0)

    if not temp_parquet.exists() or not config.resume:
        logger.info("Exporting ES documents to temporary parquet...")

        # Stream to parquet in chunks
        rows = []
        config_copy = MigrationConfig(**vars(config))
        config_copy.skip_docs = 0  # Export everything, skip during indexing

        for doc in tqdm(
            stream_es_documents(es_client, config_copy),
            total=total_docs,
            desc="Exporting",
        ):
            row = {"text": doc["text"], **doc["metadata"]}
            rows.append(row)

            # Write chunks to avoid memory issues
            if len(rows) >= 100_000:
                df = pd.DataFrame(rows)
                if not temp_parquet.exists():
                    df.to_parquet(temp_parquet, index=False)
                else:
                    # Append to existing parquet
                    existing = pd.read_parquet(temp_parquet)
                    pd.concat([existing, df], ignore_index=True).to_parquet(
                        temp_parquet, index=False
                    )
                    del existing
                rows.clear()
                gc.collect()

        # Write remaining
        if rows:
            df = pd.DataFrame(rows)
            if temp_parquet.exists():
                existing = pd.read_parquet(temp_parquet)
                pd.concat([existing, df], ignore_index=True).to_parquet(
                    temp_parquet, index=False
                )
            else:
                df.to_parquet(temp_parquet, index=False)

        logger.info(f"Exported to {temp_parquet}")

    # Now use FaissRagService to index
    service = create_faiss_service(config)

    # Determine metadata fields from parquet
    df_sample = pd.read_parquet(temp_parquet, columns=None).head(1)
    metadata_fields = [c for c in df_sample.columns if c != "text"]
    logger.info(f"Metadata fields: {metadata_fields}")

    # Run indexing
    service.index_from_parquet_batches(
        parquet_path=temp_parquet,
        text_field="text",
        metadata_fields=metadata_fields,
        collection_name=str(config.output_dir),
        progress_bar=True,
        batch_size=config.batch_size,
        skip_rows=skip_docs,
        checkpoint=True,
    )

    # Cleanup temp file
    if temp_parquet.exists():
        temp_parquet.unlink()
        logger.info("Cleaned up temporary parquet file")

    save_migration_state(config.output_dir, {
        "docs_processed": total_docs,
        "completed": True,
    })

    logger.info(f"Migration complete! Index saved to {config.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate Elasticsearch index to FAISS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Elasticsearch options
    parser.add_argument(
        "--es-url",
        default=os.getenv("ELASTICSEARCH_ENDPOINT", "http://localhost:9200"),
        help="Elasticsearch URL",
    )
    parser.add_argument(
        "--es-index",
        default="wiki_full_l",
        help="Source Elasticsearch index name",
    )
    parser.add_argument(
        "--es-user",
        default=os.getenv("ELASTICSEARCH_USERNAME"),
        help="Elasticsearch username",
    )
    parser.add_argument(
        "--es-password",
        default=os.getenv("ELASTICSEARCH_PASSWORD"),
        help="Elasticsearch password",
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR / "faiss_migrated",
        help="Output directory for FAISS index",
    )
    parser.add_argument(
        "--strategy",
        choices=["vector", "hnsw", "ivfpq", "opq_ivfpq", "ivfpq_disk"],
        default="ivfpq",
        help="FAISS index strategy",
    )

    # Migration mode
    parser.add_argument(
        "--use-existing-vectors",
        action="store_true",
        help="Use vectors from ES instead of re-embedding",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume interrupted migration",
    )

    # Memory options
    parser.add_argument(
        "--low-ram",
        action="store_true",
        help="Low RAM mode (< 8GB available)",
    )
    parser.add_argument(
        "--max-ram-mb",
        type=int,
        default=4096,
        help="Maximum RAM to use in MB",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for processing",
    )

    # Embedding options
    parser.add_argument(
        "--embedding-provider",
        default="modal",
        choices=["modal", "openai", "huggingface"],
        help="Embedding provider",
    )
    parser.add_argument(
        "--embedding-model",
        default="intfloat/multilingual-e5-large",
        help="Embedding model name",
    )
    parser.add_argument(
        "--gpu-batch-size",
        type=int,
        default=64,
        help="Forward-pass batch size on GPU (modal provider only). Default: 64",
    )
    parser.add_argument(
        "--request-batch-size",
        type=int,
        default=100,
        help="Number of texts per Modal request batch. Default: 100",
    )
    parser.add_argument(
        "--normalise-embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalise embedding vectors. Default: on",
    )

    # IVF_PQ options
    parser.add_argument("--ivfpq-nlist", type=int, default=4096)
    parser.add_argument("--ivfpq-m", type=int, default=32)
    parser.add_argument("--ivfpq-nbits", type=int, default=8)
    parser.add_argument("--ivfpq-nprobe", type=int, default=64)

    # Limit options
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Maximum documents to migrate (for testing)",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Disable deduplication",
    )

    args = parser.parse_args()

    # Build config
    config = MigrationConfig(
        es_url=args.es_url,
        es_user=args.es_user,
        es_password=args.es_password,
        es_index=args.es_index,
        output_dir=args.output_dir,
        strategy=args.strategy,
        low_ram=args.low_ram,
        max_ram_mb=args.max_ram_mb,
        batch_size=args.batch_size,
        use_existing_vectors=args.use_existing_vectors,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        gpu_batch_size=args.gpu_batch_size,
        request_batch_size=args.request_batch_size,
        normalise_embeddings=args.normalise_embeddings,
        resume=args.resume,
        deduplicate=not args.no_dedupe,
        max_docs=args.max_docs,
        ivfpq_nlist=args.ivfpq_nlist,
        ivfpq_m=args.ivfpq_m,
        ivfpq_nbits=args.ivfpq_nbits,
        ivfpq_nprobe=args.ivfpq_nprobe,
    )

    # Connect to Elasticsearch
    logger.info(f"Connecting to Elasticsearch at {config.es_url}")
    es_client = get_es_client(config)

    if not es_client.ping():
        logger.error("Could not connect to Elasticsearch!")
        sys.exit(1)

    # Get document count
    total_docs = get_index_doc_count(es_client, config.es_index)
    logger.info(f"Source index '{config.es_index}' has {total_docs:,} documents")

    if total_docs == 0:
        logger.error("No documents found in source index!")
        sys.exit(1)

    # Show memory estimates
    service = create_faiss_service(config)
    estimates = service.estimate_index_memory(total_docs, dim=1024)
    logger.info("Estimated RAM usage by strategy:")
    for strategy, mb in estimates.items():
        logger.info(f"  {strategy}: {mb:,.0f} MB")

    recommended = service.recommend_strategy(
        total_docs, config.max_ram_mb, dim=1024
    )
    if recommended != config.strategy:
        logger.warning(
            f"Recommended strategy for {config.max_ram_mb}MB RAM: {recommended} "
            f"(you selected: {config.strategy})"
        )

    # Run migration
    logger.info(f"Starting migration with strategy: {config.strategy}")
    start_time = time.time()

    if config.use_existing_vectors:
        migrate_with_existing_vectors(config, es_client, total_docs)
    else:
        migrate_with_reembedding(config, es_client, total_docs)

    elapsed = time.time() - start_time
    logger.info(f"Total migration time: {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
