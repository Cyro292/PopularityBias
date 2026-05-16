"""FAISS indexing CLI.

Builds a FAISS index from a wiki corpus Parquet file using the batched
streaming pipeline in :class:`src.rag.faiss_rag_service.FaissRagService`.
Supports resume via ``--skip-rows``.

Usage
-----
::

    # Default (wiki_full_bil, ivfpq strategy)
    python -m src.process.indexing.run_faiss

    # Custom collection and output directory
    python -m src.process.indexing.run_faiss \\
        --collection wiki_full_bil \\
        --output-dir data/wiki_full_bil/faiss_high

    # Resume a partial build
    python -m src.process.indexing.run_faiss \\
        --parquet data/wiki_full_bil/wiki_corpus.parquet \\
        --output-dir data/wiki_full_bil/faiss_high \\
        --skip-rows 5000000

    # Use flat (exact) strategy
    python -m src.process.indexing.run_faiss --strategy flat

    python -m src.process.indexing.run_faiss --help
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import dotenv
dotenv.load_dotenv()

from config import DATA_DIR
from src.rag.faiss_rag_service import FaissRagService
from src.rag.utils import IndexingConfig

logger = logging.getLogger(__name__)


# ── Defaults ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _Defaults:
    """Default hyperparameters for FAISS indexing (single source of truth)."""

    collection:           str            = "wiki_full_bil"
    strategy:             str            = "ivfpq"
    distance:             str            = "cosine"
    embedding_model:      str            = "Lajavaness/bilingual-embedding-small"
    embedding_provider:   str            = "huggingface"
    chunk_size:           int            = 1_000
    chunk_overlap:        int            = 100
    batch_size:           int            = 5_000
    skip_rows:            int            = 0
    gpu_batch_size:       int            = 254
    request_batch_size:   int            = 254
    normalise_embeddings: bool           = True
    metadata_fields:      tuple[str, ...] = (
        "wikipedia_id",
        "wikipedia_title",
        "popularity_avg",
        "popularity_rank",
    )


DEFAULTS = _Defaults()


# ── Argument parsing ──────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a FAISS index from a wiki corpus Parquet file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--collection", "-c", default=DEFAULTS.collection,
                   help="Collection name — resolves default parquet and output paths.")
    p.add_argument("--parquet", "-p", type=Path, default=None,
                   help="Explicit corpus Parquet path (overrides --collection path).")
    p.add_argument("--output-dir", "-o", type=Path, default=None,
                   help="FAISS index output directory (default: data/<collection>/faiss/).")
    p.add_argument("--strategy", default=DEFAULTS.strategy,
                   choices=["flat", "ivfpq", "hnsw", "ivfpq_disk"],
                   help="FAISS index strategy.")
    p.add_argument("--distance", default=DEFAULTS.distance,
                   choices=["cosine", "l2", "inner_product"],
                   help="Distance metric.")
    p.add_argument("--embedding-model", "-m", default=DEFAULTS.embedding_model)
    p.add_argument("--embedding-provider", default=DEFAULTS.embedding_provider)
    p.add_argument("--chunk-size", type=int, default=DEFAULTS.chunk_size)
    p.add_argument("--chunk-overlap", type=int, default=DEFAULTS.chunk_overlap)
    p.add_argument("--batch-size", type=int, default=DEFAULTS.batch_size,
                   help="Parquet rows per batch (smaller = less RAM).")
    p.add_argument("--skip-rows", "-s", type=int, default=DEFAULTS.skip_rows,
                   help="Resume from this row offset.")
    p.add_argument("--gpu-batch-size", type=int, default=DEFAULTS.gpu_batch_size)
    p.add_argument("--request-batch-size", type=int, default=DEFAULTS.request_batch_size)
    p.add_argument("--no-normalise", action="store_true",
                   help="Disable embedding normalisation.")
    p.add_argument("--metadata-fields", nargs="+", default=list(DEFAULTS.metadata_fields))
    return p.parse_args(argv)


# ── Indexing job ──────────────────────────────────────────────────────────────

def run_indexing(
    parquet_path: Path,
    output_dir: Path,
    *,
    strategy: str,
    distance: str,
    embedding_model: str,
    embedding_provider: str,
    chunk_size: int,
    chunk_overlap: int,
    batch_size: int,
    skip_rows: int,
    gpu_batch_size: int,
    request_batch_size: int,
    normalise_embeddings: bool,
    metadata_fields: list[str],
) -> int:
    """Build and persist a FAISS index.

    Args:
        parquet_path: Path to the corpus Parquet file.
        output_dir: Directory to write the index into.
        strategy: FAISS index type (``"ivfpq"``, ``"flat"``, etc.).
        distance: Distance metric (``"cosine"``, ``"l2"``, ``"inner_product"``).
        embedding_model: Sentence-transformer model identifier.
        embedding_provider: Embedding backend (``"huggingface"``, ``"modal"``).
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.
        batch_size: Parquet rows per batch.
        skip_rows: Row offset to resume a partial build.
        gpu_batch_size: Embedding GPU batch size.
        request_batch_size: Embedding API request batch size.
        normalise_embeddings: Whether to L2-normalise embeddings.
        metadata_fields: Corpus columns stored as document metadata.

    Returns:
        Total number of chunks indexed.

    Raises:
        FileNotFoundError: If ``parquet_path`` does not exist.
    """
    if not parquet_path.exists():
        raise FileNotFoundError(f"Corpus not found: {parquet_path}")

    logger.info("=" * 60)
    logger.info("FAISS INDEXING")
    logger.info("  Parquet    : %s", parquet_path)
    logger.info("  Output dir : %s", output_dir)
    logger.info("  Strategy   : %s  distance=%s", strategy, distance)
    logger.info("  Model      : %s (%s)", embedding_model, embedding_provider)
    logger.info("  Chunk      : size=%d  overlap=%d", chunk_size, chunk_overlap)
    logger.info("  Batch      : %d rows  skip=%d", batch_size, skip_rows)
    logger.info("=" * 60)

    config = IndexingConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model,
        embedding_provider=embedding_provider,
        gpu_batch_size=gpu_batch_size,
        request_batch_size=request_batch_size,
        normalise_embeddings=normalise_embeddings,
        trust_remote_code=True,
    )
    service = FaissRagService(
        config=config,
        strategy=strategy,
        distance_strategy=distance,
    )

    t0 = time.time()
    _index, n_indexed = service.index_from_parquet_batches(
        parquet_path=parquet_path,
        text_field="text",
        metadata_fields=metadata_fields,
        collection_name=str(output_dir),
        batch_size=batch_size,
        skip_rows=skip_rows,
        checkpoint=True,
        progress_bar=True,
    )
    elapsed = time.time() - t0

    logger.info("=" * 60)
    logger.info("DONE — %d chunks indexed in %.1f min", n_indexed, elapsed / 60)
    logger.info("Index written to: %s", output_dir)
    logger.info("=" * 60)
    return n_indexed


# ── Entry point ───────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    """Parse arguments and run FAISS indexing."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-28s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    args = _parse_args(argv)

    collection_dir = DATA_DIR / args.collection
    parquet_path: Path = args.parquet or (collection_dir / "wiki_corpus.parquet")
    output_dir: Path   = args.output_dir or (collection_dir / "faiss")

    try:
        run_indexing(
            parquet_path=parquet_path,
            output_dir=output_dir,
            strategy=args.strategy,
            distance=args.distance,
            embedding_model=args.embedding_model,
            embedding_provider=args.embedding_provider,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            batch_size=args.batch_size,
            skip_rows=args.skip_rows,
            gpu_batch_size=args.gpu_batch_size,
            request_batch_size=args.request_batch_size,
            normalise_embeddings=not args.no_normalise,
            metadata_fields=args.metadata_fields,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Interrupted — index may be incomplete.")
        sys.exit(130)
    except Exception as exc:
        logger.error("Indexing failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
