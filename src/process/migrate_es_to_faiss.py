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
        --es-index wiki_full_bil \
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
from src.storage.sqlite_docstore import SqliteDocstore
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
    es_index: str = "wiki_full_bil"

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
    checkpoint_every: int = 50_000
    deduplicate: bool = True
    skip_docs: int = 0
    max_docs: int | None = None

    # IVF_PQ settings (tuned for ~4M docs with 1024-dim vectors)
    ivfpq_nlist: int = 4096
    ivfpq_m: int = 64  # Must divide embedding dim
    ivfpq_nbits: int = 8
    ivfpq_nprobe: int = 256


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
    *,
    resume_search_after: list | None = None,
) -> Generator[dict, None, None]:
    """Stream all documents from ES using PIT + search_after.

    Yields dicts with keys: _id, text, metadata, vector (if present).

    Args:
        client: Elasticsearch client.
        config: Migration configuration.
        resume_search_after: If provided, skip directly to this PIT cursor
            position instead of scanning from the beginning. Obtained from
            a previous run's checkpoint via ``migration_state["search_after"]``.
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

    # Jump directly to the stored cursor position if resuming
    search_after: list | None = resume_search_after
    if search_after:
        logger.info(f"Resuming from search_after cursor: {search_after}")

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
                    "_sort": doc["sort"],   # expose cursor for checkpointing
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


def resolve_resume_cursor(
    faiss_dir: Path,
    es_client: Elasticsearch,
    es_index: str,
) -> tuple[int, list | None]:
    """Derive the true resume position from on-disk data — never trusts the state file.

    Algorithm:
    1. Read ``faiss_index.ntotal`` from ``index.faiss`` — this is the authoritative
       count of vectors on disk.
    2. Find the id_map entry at position ``ntotal - 1`` in SQLite to get the UID of
       the last vector that was flushed to disk.
    3. Trim any SQLite rows that are *ahead* of the FAISS checkpoint (written to
       SQLite after the last ``faiss.write_index`` call but before the crash).
    4. Look up the ``wikipedia_id`` from the last doc, query ES for its
       ``_shard_doc`` sort value, and return that as the PIT ``search_after`` cursor.

    Returns:
        ``(total_indexed, search_after)`` where ``total_indexed`` is
        ``faiss_index.ntotal`` and ``search_after`` is the ES PIT cursor list.
    """
    import faiss
    import sqlite3 as _sqlite3

    index_path = faiss_dir / "index.faiss"
    db_path = faiss_dir / "docstore.sqlite"

    if not index_path.exists() or not db_path.exists():
        logger.info("No existing checkpoint found — starting fresh")
        return 0, None

    # ── Step 1: how many vectors are actually on disk ─────────────────────
    logger.info(f"Loading existing FAISS index to determine checkpoint position...")
    idx = faiss.read_index(str(index_path))
    ntotal = idx.ntotal
    del idx  # free RAM — will be re-loaded in the main function
    logger.info(f"FAISS on disk: {ntotal:,} vectors")

    # ── Step 2: trim SQLite to match FAISS ────────────────────────────────
    conn = _sqlite3.connect(str(db_path))
    id_map_count = conn.execute("SELECT COUNT(*) FROM id_map").fetchone()[0]
    doc_count = conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
    logger.info(f"SQLite before trim — id_map: {id_map_count:,}, docs: {doc_count:,}")

    if id_map_count > ntotal:
        excess = id_map_count - ntotal
        logger.info(f"Trimming {excess:,} id_map rows ahead of FAISS checkpoint...")
        conn.execute("DELETE FROM id_map WHERE pos >= ?", (ntotal,))
        conn.commit()

    # Trim docs table: any uid with numeric suffix >= ntotal that refers to a
    # regular doc (not a training vector) may have been written after the last
    # faiss.write_index call — remove them so docstore stays in sync.
    # Training vector uids (train_*) are always within the FAISS snapshot.
    conn.execute(
        "DELETE FROM docs WHERE CAST(SUBSTR(uid, 5) AS INTEGER) >= ?",
        (ntotal,)
    )
    conn.commit()

    id_map_after = conn.execute("SELECT COUNT(*) FROM id_map").fetchone()[0]
    doc_after = conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
    logger.info(f"SQLite after trim  — id_map: {id_map_after:,}, docs: {doc_after:,}")

    # ── Step 3: find the last real doc's wikipedia_id ─────────────────────
    # Walk backwards from ntotal-1 to find a non-training-vector entry
    last_wikipedia_id: int | None = None
    for offset in range(min(1000, ntotal)):
        pos = ntotal - 1 - offset
        row = conn.execute(
            "SELECT uid FROM id_map WHERE pos = ?", (pos,)
        ).fetchone()
        if row is None:
            continue
        uid = row[0]
        if not uid.startswith("doc_"):
            continue  # skip train_* entries
        doc_row = conn.execute(
            "SELECT metadata FROM docs WHERE uid = ?", (uid,)
        ).fetchone()
        if doc_row is None:
            continue
        import json as _json
        meta = _json.loads(doc_row[0])
        last_wikipedia_id = meta.get("wikipedia_id")
        if last_wikipedia_id is not None:
            logger.info(f"Last doc uid={uid}, wikipedia_id={last_wikipedia_id}")
            break

    conn.close()

    if last_wikipedia_id is None:
        logger.warning("Could not find a last wikipedia_id — resuming without cursor (full re-stream)")
        return ntotal, None

    # ── Step 4: query ES for the _shard_doc sort value ────────────────────
    resp = es_client.search(
        index=es_index,
        body={
            "size": 1,
            "query": {"term": {"metadata.wikipedia_id": last_wikipedia_id}},
            "sort": [{"_shard_doc": "asc"}],
            "_source": False,
        },
    )
    hits = resp["hits"]["hits"]
    if not hits:
        logger.warning(
            f"wikipedia_id {last_wikipedia_id} not found in ES — "
            "resuming without cursor (full re-stream)"
        )
        return ntotal, None

    search_after = hits[0]["sort"]
    logger.info(f"ES resume cursor (search_after): {search_after}")
    return ntotal, search_after


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
    from src.storage.sqlite_docstore import SqliteDocstore

    logger.info("Migration mode: Using existing vectors from Elasticsearch")

    # Setup output directory
    faiss_dir = config.output_dir / "faiss"
    faiss_dir.mkdir(parents=True, exist_ok=True)

    # Initialize or load docstore
    docstore = SqliteDocstore(faiss_dir / "docstore.sqlite")

    # ── Resume: derive ground truth from on-disk data ─────────────────────
    # Never trust migration_state.json — reconcile FAISS + SQLite directly.
    faiss_index = None
    total_indexed = 0
    resume_search_after: list | None = None

    if config.resume and (faiss_dir / "index.faiss").exists():
        total_indexed, resume_search_after = resolve_resume_cursor(
            faiss_dir, es_client, config.es_index
        )
        if total_indexed > 0:
            logger.info(f"Loading existing FAISS index ({total_indexed:,} vectors)...")
            faiss_index = faiss.read_index(str(faiss_dir / "index.faiss"))
            logger.info("FAISS index loaded — resuming from checkpoint")
    else:
        logger.info("Starting fresh migration")

    # Accumulators for batch processing
    vectors_buffer: list[np.ndarray] = []
    texts_buffer: list[str] = []
    metadatas_buffer: list[dict] = []
    # id_map is written incrementally to SQLite — no in-memory dict needed
    last_sort: list | None = None  # tracks ES PIT cursor for checkpoint resume
    training_vectors: list[np.ndarray] = []  # only used before IVF_PQ is trained

    # Stream documents
    pbar = tqdm(
        total=total_docs,
        initial=total_indexed,
        desc="Migrating",
        unit="docs",
    )

    try:
        for doc in stream_es_documents(es_client, config, resume_search_after=resume_search_after):
            vector = doc.get("vector")
            if vector is None:
                logger.warning(f"Document {doc['_id']} has no vector, skipping")
                continue

            last_sort = doc.get("_sort")  # save cursor before processing

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
                    id_map_batch = {i: f"train_{i}" for i in range(len(training_vectors))}
                    docstore.append_id_map(id_map_batch)

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
                id_map_batch = {}
                for i, (text, meta) in enumerate(zip(texts_buffer, metadatas_buffer)):
                    uid = f"doc_{start_idx + i}"
                    id_map_batch[start_idx + i] = uid
                    docs_to_add[uid] = Document(page_content=text, metadata=meta)
                docstore.add(docs_to_add)
                docstore.append_id_map(id_map_batch)

                total_indexed += len(vectors_buffer)
                pbar.update(len(vectors_buffer))

                # Checkpoint
                if total_indexed % config.checkpoint_every == 0:
                    logger.info(f"Checkpoint at {total_indexed:,} docs")
                    faiss.write_index(faiss_index, str(faiss_dir / "index.faiss"))
                    # id_map is already persisted incrementally — no full rewrite
                    save_migration_state(config.output_dir, {
                        "docs_processed": total_indexed,
                        "search_after": last_sort,
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
                id_map_batch = {i: f"doc_{i}" for i in range(len(train_data))}
                docstore.append_id_map(id_map_batch)
                del train_data, all_train
            else:
                start_idx = faiss_index.ntotal
                faiss_index.add(batch_vectors)

                docs_to_add = {}
                id_map_batch = {}
                for i, (text, meta) in enumerate(zip(texts_buffer, metadatas_buffer)):
                    uid = f"doc_{start_idx + i}"
                    id_map_batch[start_idx + i] = uid
                    docs_to_add[uid] = Document(page_content=text, metadata=meta)
                docstore.add(docs_to_add)
                docstore.append_id_map(id_map_batch)

            total_indexed += len(vectors_buffer)

    finally:
        pbar.close()

    # Final save
    logger.info(f"Saving final index with {faiss_index.ntotal:,} vectors...")
    faiss.write_index(faiss_index, str(faiss_dir / "index.faiss"))
    docstore.flush()  # ensure any pending doc inserts are committed
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
    # Use MigrationConfig dataclass defaults as the single source of truth
    _defaults = MigrationConfig()

    parser = argparse.ArgumentParser(
        description="Migrate Elasticsearch index to FAISS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Elasticsearch options
    parser.add_argument(
        "--es-url",
        default=_defaults.es_url,
        help="Elasticsearch URL",
    )
    parser.add_argument(
        "--es-index",
        default=_defaults.es_index,
        help="Source Elasticsearch index name",
    )
    parser.add_argument(
        "--es-user",
        default=_defaults.es_user,
        help="Elasticsearch username",
    )
    parser.add_argument(
        "--es-password",
        default=_defaults.es_password,
        help="Elasticsearch password",
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_defaults.output_dir,
        help="Output directory for FAISS index",
    )
    parser.add_argument(
        "--strategy",
        choices=["vector", "hnsw", "ivfpq", "opq_ivfpq", "ivfpq_disk"],
        default=_defaults.strategy,
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
        default=_defaults.max_ram_mb,
        help="Maximum RAM to use in MB",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=_defaults.batch_size,
        help="Batch size for processing",
    )

    # Embedding options
    parser.add_argument(
        "--embedding-provider",
        default=_defaults.embedding_provider,
        choices=["modal", "openai", "huggingface"],
        help="Embedding provider",
    )
    parser.add_argument(
        "--embedding-model",
        default=_defaults.embedding_model,
        help="Embedding model name",
    )
    parser.add_argument(
        "--gpu-batch-size",
        type=int,
        default=_defaults.gpu_batch_size,
        help="Forward-pass batch size on GPU (modal provider only). Default: 64",
    )
    parser.add_argument(
        "--request-batch-size",
        type=int,
        default=_defaults.request_batch_size,
        help="Number of texts per Modal request batch. Default: 100",
    )
    parser.add_argument(
        "--normalise-embeddings",
        action=argparse.BooleanOptionalAction,
        default=_defaults.normalise_embeddings,
        help="Normalise embedding vectors. Default: on",
    )

    # IVF_PQ options
    parser.add_argument("--ivfpq-nlist", type=int, default=_defaults.ivfpq_nlist)
    parser.add_argument("--ivfpq-m", type=int, default=_defaults.ivfpq_m)
    parser.add_argument("--ivfpq-nbits", type=int, default=_defaults.ivfpq_nbits)
    parser.add_argument("--ivfpq-nprobe", type=int, default=_defaults.ivfpq_nprobe)

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
    estimates = service.estimate_index_memory(total_docs, dim=384)
    logger.info("Estimated RAM usage by strategy:")
    for strategy, mb in estimates.items():
        logger.info(f"  {strategy}: {mb:,.0f} MB")

    recommended = service.recommend_strategy(
        total_docs, config.max_ram_mb, dim=384
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
