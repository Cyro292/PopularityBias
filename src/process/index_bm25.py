"""BM25 indexing script for the wiki_full_bil corpus.

Builds a bm25s-backed BM25 index from a wiki corpus Parquet file and writes
it to disk.  The resulting index directory can be loaded by
:class:`src.rag.bm25_rag_service.BM25RagService` for offline keyword
retrieval without Elasticsearch.

On-disk layout written by this script
--------------------------------------
::

    <output_dir>/
        scores/         ← bm25s vocab + score arrays
        docstore.db     ← SQLite: doc_id → (text, metadata JSON)
        arrays.npz      ← compact numpy metadata (wikipedia_ids, popularities, deciles, titles)
        config.json     ← hyperparameters and corpus statistics

RAM requirements
----------------
Building the index requires tokenising the full corpus in memory.  For the
``wiki_full_bil`` corpus (~5.9 M articles chunked to ~30 M chunks at
chunk_size=1000) expect **12–16 GB** peak RAM during the tokenisation step.
Query-time RAM after loading is ~500 MB.

Usage
-----
::

    # Default (wiki_full_bil, chunk_size=1000, chunk_overlap=100)
    python -m src.process.index_bm25

    # Custom collection
    python -m src.process.index_bm25 --collection wiki_full_bil

    # Explicit parquet + output directory
    python -m src.process.index_bm25 \\
        --parquet data/wiki_full_bil/wiki_corpus.parquet \\
        --output-dir data/wiki_full_bil/bm25

    # Tune BM25 parameters
    python -m src.process.index_bm25 --k1 1.2 --b 0.75

    # Skip chunking (index whole articles as single documents)
    python -m src.process.index_bm25 --no-chunk

    # Full help
    python -m src.process.index_bm25 --help
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Ensure project root on sys.path when run as a script or via -m
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import dotenv
dotenv.load_dotenv()

from config import DATA_DIR
from src.rag.bm25_rag_service import BM25RagService

logger = logging.getLogger(__name__)

# ── Default values (single source of truth) ───────────────────────────────────

@dataclass(frozen=True)
class IndexingDefaults:
    """Default hyperparameters for BM25 indexing."""

    collection: str = "wiki_full_bil"
    chunk: bool = True
    chunk_size: int = 1_000
    chunk_overlap: int = 100
    k1: float = 1.5
    b: float = 0.75
    metadata_fields: tuple[str, ...] = (
        "wikipedia_id",
        "wikipedia_title",
        "popularity_avg",
        "popularity_rank",
        "decile",
    )


DEFAULTS = IndexingDefaults()


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Populated :class:`argparse.Namespace`.
    """
    p = argparse.ArgumentParser(
        description="Build a bm25s index from a wiki corpus Parquet file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Paths ──────────────────────────────────────────────────────────────
    p.add_argument(
        "--collection", "-c",
        default=DEFAULTS.collection,
        help="Collection name — used to resolve default parquet and output paths "
             "(data/<collection>/wiki_corpus.parquet → data/<collection>/bm25/).",
    )
    p.add_argument(
        "--parquet", "-p",
        type=Path,
        default=None,
        help="Explicit path to the corpus Parquet file.  "
             "Overrides the path derived from --collection.",
    )
    p.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=None,
        help="Directory to write the index into.  "
             "Defaults to data/<collection>/bm25/.",
    )

    # ── Chunking ───────────────────────────────────────────────────────────
    p.add_argument(
        "--no-chunk",
        action="store_true",
        help="Disable document chunking — index whole articles as single documents.",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULTS.chunk_size,
        help="Maximum characters per chunk.",
    )
    p.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULTS.chunk_overlap,
        help="Overlap in characters between consecutive chunks.",
    )

    # ── BM25 hyperparameters ───────────────────────────────────────────────
    p.add_argument(
        "--k1",
        type=float,
        default=DEFAULTS.k1,
        help="BM25 term-saturation parameter (typical range 1.2–2.0).",
    )
    p.add_argument(
        "--b",
        type=float,
        default=DEFAULTS.b,
        help="BM25 length-normalisation parameter (0 = off, 1 = full).",
    )

    # ── Metadata ───────────────────────────────────────────────────────────
    p.add_argument(
        "--metadata-fields",
        nargs="+",
        default=list(DEFAULTS.metadata_fields),
        help="Corpus columns stored as document metadata.",
    )

    return p.parse_args(argv)


# ── Indexing job ──────────────────────────────────────────────────────────────

def run_indexing(
    parquet_path: Path,
    output_dir: Path,
    *,
    chunk: bool,
    chunk_size: int,
    chunk_overlap: int,
    k1: float,
    b: float,
    metadata_fields: list[str],
) -> int:
    """Build and persist a BM25 index for one corpus.

    Args:
        parquet_path: Path to the corpus Parquet file.
        output_dir: Directory to write the index into.
        chunk: Whether to split articles into chunks.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.
        k1: BM25 term-saturation parameter.
        b: BM25 length-normalisation parameter.
        metadata_fields: Corpus columns stored as document metadata.

    Returns:
        Number of indexed chunks (or documents when ``chunk=False``).

    Raises:
        FileNotFoundError: If *parquet_path* does not exist.
    """
    if not parquet_path.exists():
        raise FileNotFoundError(f"Corpus not found: {parquet_path}")

    logger.info("=" * 60)
    logger.info("BM25 INDEXING")
    logger.info("=" * 60)
    logger.info("  Parquet      : %s", parquet_path)
    logger.info("  Output dir   : %s", output_dir)
    logger.info("  Chunk        : %s (size=%d, overlap=%d)", chunk, chunk_size, chunk_overlap)
    logger.info("  BM25 params  : k1=%.2f  b=%.2f", k1, b)
    logger.info("  Metadata     : %s", ", ".join(metadata_fields))
    logger.info("=" * 60)

    service = BM25RagService(
        chunk=chunk,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        k1=k1,
        b=b,
    )

    t0 = time.time()
    result = service.index_from_parquet(
        parquet_path,
        text_field="text",
        metadata_fields=metadata_fields,
        output_dir=output_dir,
    )
    elapsed = time.time() - t0

    _index, n_indexed = result
    logger.info("=" * 60)
    logger.info("DONE — %d chunks indexed in %.1f min", n_indexed, elapsed / 60)
    logger.info("Index written to: %s", output_dir)
    logger.info("=" * 60)

    return n_indexed


# ── Entry point ───────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    """Parse arguments and run the BM25 indexing pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-28s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    args = _parse_args(argv)

    # ── Resolve paths ──────────────────────────────────────────────────────
    collection_dir = DATA_DIR / args.collection
    parquet_path: Path = args.parquet or (collection_dir / "wiki_corpus.parquet")
    output_dir: Path = args.output_dir or (collection_dir / "bm25")

    # ── Guard: warn if output already exists ──────────────────────────────
    if output_dir.exists():
        logger.warning(
            "Output directory %s already exists — existing files will be overwritten.",
            output_dir,
        )

    # ── Run ────────────────────────────────────────────────────────────────
    try:
        run_indexing(
            parquet_path=parquet_path,
            output_dir=output_dir,
            chunk=not args.no_chunk,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            k1=args.k1,
            b=args.b,
            metadata_fields=args.metadata_fields,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user — index may be incomplete.")
        sys.exit(130)
    except Exception as exc:
        logger.error("Indexing failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
