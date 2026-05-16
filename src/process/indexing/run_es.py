"""Elasticsearch indexing CLI.

Builds or resumes an Elasticsearch index from a wiki corpus Parquet file.
Supports single-job mode (CLI flags) and multi-job mode (JSON config file).

Single-job examples
-------------------
::

    python -m src.process.indexing.run_es
    python -m src.process.indexing.run_es -c wiki_full_bil -s 4025000
    python -m src.process.indexing.run_es -m intfloat/multilingual-e5-large
    python -m src.process.indexing.run_es --no-send-mode
    python -m src.process.indexing.run_es --help

Multi-job mode
--------------
::

    python -m src.process.indexing.run_es --jobs data/jobs/job1.json

    jobs.json example::

        [
          {"collection": "wiki_full_l", "skip_rows": 4025000},
          {"collection": "wiki_full_s", "embedding_model": "intfloat/multilingual-e5-small"}
        ]

    Only fields you want to override need to be present; everything else
    uses the defaults defined in ``DEFAULTS``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import dotenv
dotenv.load_dotenv()

from config import DATA_DIR
from src.rag.elasticsearch_rag_service import ElasticsearchRagService
from src.rag.utils import IndexingConfig

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _Defaults:
    """Default hyperparameters for ES indexing (single source of truth)."""

    collection:           str            = "wiki_full_l"
    skip_rows:            int            = 0
    embedding_model:      str            = "intfloat/multilingual-e5-large"
    embedding_provider:   str            = "modal"
    gpu_batch_size:       int            = 512
    request_batch_size:   int            = 4096
    normalise_embeddings: bool           = True
    chunk_size:           int            = 1000
    chunk_overlap:        int            = 100
    strategy:             str            = "approximation"
    request_timeout:      int            = 300
    batch_size:           int            = 35_000
    send_mode:            bool           = True
    passage_prompt_file:  str | None     = None
    query_prompt_file:    str | None     = None
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
        description="Create / resume Elasticsearch indices from wiki corpus Parquet files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--jobs", type=Path, default=None,
                   help="JSON file with a list of job configs (runs sequentially). "
                        "All other flags are ignored when this is provided.")
    p.add_argument("--collection", "-c", default=DEFAULTS.collection)
    p.add_argument("--skip-rows", "-s", type=int, default=DEFAULTS.skip_rows,
                   help="Resume from this row offset.")
    p.add_argument("--embedding-model", "-m", default=DEFAULTS.embedding_model)
    p.add_argument("--embedding-provider", default=DEFAULTS.embedding_provider)
    p.add_argument("--gpu-batch-size", type=int, default=DEFAULTS.gpu_batch_size)
    p.add_argument("--request-batch-size", type=int, default=DEFAULTS.request_batch_size)
    p.add_argument("--normalise-embeddings", type=bool, default=DEFAULTS.normalise_embeddings)
    p.add_argument("--chunk-size", type=int, default=DEFAULTS.chunk_size)
    p.add_argument("--chunk-overlap", type=int, default=DEFAULTS.chunk_overlap)
    p.add_argument("--es-url", default=os.getenv("ELASTICSEARCH_ENDPOINT", "http://localhost:9200/"))
    p.add_argument("--strategy", default=DEFAULTS.strategy,
                   choices=["vector", "bm25", "approximation", "hybrid"])
    p.add_argument("--request-timeout", type=int, default=DEFAULTS.request_timeout)
    p.add_argument("--batch-size", type=int, default=DEFAULTS.batch_size)
    p.add_argument("--no-send-mode", action="store_true",
                   help="Use standard embed→upload instead of Modal→ES direct send.")
    p.add_argument("--passage-prompt-file", type=str, default=DEFAULTS.passage_prompt_file)
    p.add_argument("--query-prompt-file", type=str, default=DEFAULTS.query_prompt_file)
    p.add_argument("--parquet", type=Path, default=None,
                   help="Explicit parquet path (default: data/<collection>/wiki_corpus.parquet).")
    p.add_argument("--metadata-fields", nargs="+", default=list(DEFAULTS.metadata_fields))
    return p.parse_args(argv)


def _args_to_job(args: argparse.Namespace) -> dict:
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
    with open(path) as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        sys.exit(f"ERROR: {path} must contain a JSON array of job objects.")
    # Merge each entry on top of dataclass defaults converted to a plain dict
    _base = {**asdict(DEFAULTS), "parquet": None, "es_url": os.getenv("ELASTICSEARCH_ENDPOINT", "http://localhost:9200/")}
    jobs = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            sys.exit(f"ERROR: Job #{i + 1} in {path} is not an object.")
        jobs.append({**_base, **entry})
    return jobs


# ── Job execution ─────────────────────────────────────────────────────────────

def run_job(job: dict, job_num: int, total_jobs: int) -> int:
    """Execute a single ES indexing job.

    Args:
        job: Job config dict (merged with DEFAULTS).
        job_num: 1-based job number (for display).
        total_jobs: Total number of jobs in the batch.

    Returns:
        Number of chunks indexed, or 0 on skip.

    Raises:
        Exception: Propagates any error from the indexing service.
    """
    parquet_path = (
        Path(job["parquet"]) if job["parquet"]
        else DATA_DIR / job["collection"] / "wiki_corpus.parquet"
    )
    if not parquet_path.exists():
        logger.error("Parquet not found: %s — skipping job #%d", parquet_path, job_num)
        return 0

    es_user     = os.getenv("ELASTICSEARCH_USERNAME")
    es_password = os.getenv("ELASTICSEARCH_PASSWORD")

    print()
    print("=" * 60)
    print(f"  Job {job_num}/{total_jobs}")
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


# ── Entry point ───────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    """Parse arguments and run all ES indexing jobs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-28s  %(levelname)-7s  %(message)s",
        force=True,
    )
    for noisy in ("httpx", "httpcore", "openai", "elastic_transport"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    args = _parse_args(argv)

    if args.jobs:
        if not args.jobs.exists():
            sys.exit(f"ERROR: Jobs file not found: {args.jobs}")
        jobs = _load_jobs(args.jobs)
        logger.info("Loaded %d job(s) from %s", len(jobs), args.jobs)
    else:
        jobs = [_args_to_job(args)]

    total = len(jobs)
    results: list[tuple[int, str, int]] = []
    t0 = time.time()

    for i, job in enumerate(jobs, 1):
        try:
            chunks = run_job(job, i, total)
            results.append((i, job["collection"], chunks))
        except Exception as e:
            logger.error("Job %d/%d FAILED: %s", i, total, e, exc_info=True)
            results.append((i, job["collection"], -1))

    elapsed = time.time() - t0
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
