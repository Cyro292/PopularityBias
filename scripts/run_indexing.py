#!/usr/bin/env python3
"""Standalone Elasticsearch indexing script.

Supports two modes:

1) **Single job** via CLI flags (unchanged from before):

    python scripts/run_indexing.py -c wiki_full_l -s 4025000

2) **Multiple sequential jobs** via a JSON config file:

    python scripts/run_indexing.py --jobs jobs.json

    Where jobs.json contains a list of job objects.  Each key maps to a
    CLI flag (use underscores instead of hyphens).  Only the fields you
    want to override need to be present — everything else uses defaults.

    Example jobs.json:
    [
      {
        "collection": "wiki_full_l",
        "embedding_model": "intfloat/multilingual-e5-large",
        "skip_rows": 4025000
      },
      {
        "collection": "wiki_full_s",
        "embedding_model": "intfloat/multilingual-e5-small",
        "skip_rows": 0
      }
    ]

Other examples (single-job mode):

    python scripts/run_indexing.py --skip-rows 4025000
    python scripts/run_indexing.py -m intfloat/multilingual-e5-small
    python scripts/run_indexing.py --collection wiki_full_v2 --batch-size 20000
    python scripts/run_indexing.py --no-send-mode
    python scripts/run_indexing.py --help
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dotenv
dotenv.load_dotenv()

from config import DATA_DIR
from rag.elasticsearch_rag_service import ElasticsearchRagService
from rag.utils import IndexingConfig

logger = logging.getLogger(__name__)

# ── Defaults (single source of truth) ────────────────────────────────────────
DEFAULTS: dict = {
    "collection":           "wiki_full_l",
    "skip_rows":            0,
    "embedding_model":      "intfloat/multilingual-e5-large",
    "embedding_provider":   "modal",
    "gpu_batch_size":       512,
    "request_batch_size":   4096,
    "normalise_embeddings": True,
    "chunk_size":           1000,
    "chunk_overlap":        100,
    "es_url":               "http://52.241.5.104:9200/",
    "strategy":             "approximation",
    "request_timeout":      300,
    "batch_size":           35_000,
    "send_mode":            True,
    "parquet":              None,
    "passage_prompt_file":  None,
    "query_prompt_file":    None,
    "metadata_fields":      ["wikipedia_id", "wikipedia_title", "popularity_avg", "popularity_rank"],
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create / resume Elasticsearch indices from wiki_corpus.parquet files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Multi-job mode ────────────────────────────────────────────────────
    p.add_argument(
        "--jobs", type=Path, default=None,
        help="Path to a JSON file with a list of job configs (runs sequentially). "
             "When provided, all other flags are ignored.",
    )

    # ── Single-job flags ──────────────────────────────────────────────────
    p.add_argument("--collection", "-c", default=DEFAULTS["collection"],
                   help="Index / collection name")
    p.add_argument("--skip-rows", "-s", type=int, default=DEFAULTS["skip_rows"],
                   help="Resume from this row offset")
    p.add_argument("--embedding-model", "-m", default=DEFAULTS["embedding_model"],
                   help="Sentence-transformer model (must match deployed Modal service)")
    p.add_argument("--embedding-provider", default=DEFAULTS["embedding_provider"])
    p.add_argument("--gpu-batch-size", type=int, default=DEFAULTS["gpu_batch_size"])
    p.add_argument("--request-batch-size", type=int, default=DEFAULTS["request_batch_size"])
    p.add_argument("--normalise-embeddings", type=bool, default=DEFAULTS["normalise_embeddings"])
    p.add_argument("--chunk-size", type=int, default=DEFAULTS["chunk_size"])
    p.add_argument("--chunk-overlap", type=int, default=DEFAULTS["chunk_overlap"])
    p.add_argument("--es-url", default=DEFAULTS["es_url"])
    p.add_argument("--strategy", default=DEFAULTS["strategy"],
                   choices=["vector", "bm25", "approximation", "hybrid"])
    p.add_argument("--request-timeout", type=int, default=DEFAULTS["request_timeout"])
    p.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--no-send-mode", action="store_true",
                   help="Use standard embed→upload instead of Modal→ES direct send")

    # ── Prompt templates ──────────────────────────────────────────────────
    p.add_argument("--passage-prompt-file", type=str, default=None,
                   help="Path to a custom passage/document prompt file (must contain {passage})")
    p.add_argument("--query-prompt-file", type=str, default=None,
                   help="Path to a custom query prompt file (must contain {query})")

    p.add_argument("--parquet", type=Path, default=None,
                   help="Explicit parquet path (default: data/<collection>/wiki_corpus.parquet)")
    p.add_argument(
        "--metadata-fields", nargs="+", default=DEFAULTS["metadata_fields"],
        help="Metadata columns to index alongside the text",
    )

    return p.parse_args(argv)


def _args_to_job(args: argparse.Namespace) -> dict:
    """Convert parsed CLI args to a job dict."""
    return {
        "collection":           args.collection,
        "skip_rows":            args.skip_rows,
        "embedding_model":      args.embedding_model,
        "embedding_provider":   args.embedding_provider,
        "gpu_batch_size":       args.gpu_batch_size,
        "request_batch_size":   args.request_batch_size,
        "normalise_embeddings": args.normalise_embeddings,
        "chunk_size":           args.chunk_size,
        "chunk_overlap":        args.chunk_overlap,
        "es_url":               args.es_url,
        "strategy":             args.strategy,
        "request_timeout":      args.request_timeout,
        "batch_size":           args.batch_size,
        "send_mode":            not args.no_send_mode,
        "parquet":              str(args.parquet) if args.parquet else None,
        "passage_prompt_file":  args.passage_prompt_file,
        "query_prompt_file":    args.query_prompt_file,
        "metadata_fields":      args.metadata_fields,
    }


def _load_jobs(path: Path) -> list[dict]:
    """Load and validate a jobs JSON file."""
    with open(path) as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        sys.exit(f"ERROR: {path} must contain a JSON array of job objects")
    jobs = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            sys.exit(f"ERROR: Job #{i + 1} in {path} is not an object")
        # Merge with defaults — user only needs to specify overrides
        job = {**DEFAULTS, **entry}
        jobs.append(job)
    return jobs


def run_job(job: dict, job_num: int, total_jobs: int) -> int:
    """Execute a single indexing job. Returns the number of chunks indexed."""
    parquet_path = Path(job["parquet"]) if job["parquet"] else (Path(DATA_DIR) / job["collection"] / "wiki_corpus.parquet")
    if not parquet_path.exists():
        logger.error(f"Parquet not found: {parquet_path} — skipping job #{job_num}")
        return 0

    es_user = os.getenv("ELASTICSEARCH_USERNAME")
    es_password = os.getenv("ELASTICSEARCH_PASSWORD")

    header = f"  Job {job_num}/{total_jobs}"
    print()
    print("=" * 60)
    print(header)
    print("=" * 60)
    print(f"  Collection : {job['collection']}")
    print(f"  Parquet    : {parquet_path}")
    print(f"  Skip rows  : {job['skip_rows']:,}")
    print(f"  Strategy   : {job['strategy']}")
    print(f"  Model      : {job['embedding_model']}")
    print(f"  Chunk      : {job['chunk_size']} chars (overlap {job['chunk_overlap']})")
    print(f"  Batch      : {job['batch_size']:,}")
    print(f"  Send mode  : {job['send_mode']}")
    print(f"  ES URL     : {job['es_url']}")
    print("=" * 60)

    config = IndexingConfig(
        chunk_size=job["chunk_size"],
        chunk_overlap=job["chunk_overlap"],
        embedding_provider=job["embedding_provider"],
        embedding_model=job["embedding_model"],
        gpu_batch_size=job["gpu_batch_size"],
        normalise_embeddings=job["normalise_embeddings"],
        request_batch_size=job["request_batch_size"],
        passage_prompt_file=job.get("passage_prompt_file"),
        query_prompt_file=job.get("query_prompt_file"),
        trust_remote_code=True,
        use_progress=False,
    )

    service = ElasticsearchRagService(
        config=config,
        es_url=job["es_url"],
        es_user=es_user,
        es_password=es_password,
        strategy=job["strategy"],
        request_timeout=job["request_timeout"],
    )

    _index, num_chunks = service.index_from_parquet_batches(
        parquet_path=parquet_path,
        text_field="text",
        metadata_fields=job["metadata_fields"],
        collection_name=job["collection"],
        progress_bar=True,
        batch_size=job["batch_size"],
        skip_rows=job["skip_rows"],
        use_send_mode=job["send_mode"],
    )

    print(f"\n  DONE job {job_num}/{total_jobs} — {num_chunks:,} chunks → '{job['collection']}'")
    return num_chunks


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # ── Logging ───────────────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-28s  %(levelname)-7s  %(message)s",
        force=True,
    )
    for noisy in ("httpx", "httpcore", "openai", "elastic_transport"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # ── Build job list ────────────────────────────────────────────────────
    if args.jobs:
        if not args.jobs.exists():
            sys.exit(f"ERROR: Jobs file not found: {args.jobs}")
        jobs = _load_jobs(args.jobs)
        logger.info(f"Loaded {len(jobs)} job(s) from {args.jobs}")
    else:
        jobs = [_args_to_job(args)]

    # ── Run all jobs sequentially ─────────────────────────────────────────
    total = len(jobs)
    results: list[tuple[int, str, int]] = []  # (job_num, collection, chunks)
    t0 = time.time()

    for i, job in enumerate(jobs, 1):
        try:
            chunks = run_job(job, i, total)
            results.append((i, job["collection"], chunks))
        except Exception as e:
            logger.error(f"Job {i}/{total} FAILED: {e}", exc_info=True)
            results.append((i, job["collection"], -1))

    elapsed = time.time() - t0

    # ── Final summary ─────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  ALL JOBS COMPLETE  ({elapsed / 60:.1f} min)")
    print("=" * 60)
    for num, coll, chunks in results:
        status = f"{chunks:,} chunks" if chunks >= 0 else "FAILED"
        print(f"  #{num}  {coll:<30s}  {status}")
    print("=" * 60)


if __name__ == "__main__":
    main()
