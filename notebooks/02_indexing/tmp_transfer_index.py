"""Transfers wiki_full_l index from remote Elasticsearch to local via scroll + bulk.

Sorts by wikipedia_id ascending so progress is deterministic. Checkpoints the last
completed wikipedia_id to disk so the transfer can be safely interrupted and resumed.

On resume:
  - Deletes all local docs with the last checkpointed wikipedia_id (may be partial)
  - Restarts scrolling from that wikipedia_id

Speed improvements over v1:
  - Concurrent workers: each worker owns an independent sliced scroll context so
    fetch and bulk-index happen in parallel across N workers (default 6).
  - Larger batch size (2000 docs) — amortises the ~1.5 s remote round-trip cost.
  - Per-batch _refresh + _count removed — replaced with a background progress thread
    that logs every 15 s without blocking workers.
  - bulk uses refresh=false during transfer; single _refresh call at the end.
  - Failed bulk items are retried once before being counted as lost.

Safe resume strategy:
  - Each worker tracks its own minimum confirmed wikipedia_id (the lowest id whose
    batch has been fully bulk-indexed).
  - The checkpoint stores the MINIMUM of all per-worker confirmed ids so that on
    resume we are guaranteed every doc below that id was successfully indexed.
  - On resume: all local docs with wikipedia_id >= checkpoint are deleted before
    restarting, ensuring no partial or out-of-order batches survive.

Usage:
    python tmp_transfer_index.py                    # full transfer / resume
    python tmp_transfer_index.py --reset-checkpoint # ignore checkpoint, start fresh
    python tmp_transfer_index.py --fresh            # wipe local index and start over
    python tmp_transfer_index.py --workers 8        # override worker count
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# === Config ===
SRC_HOST        = os.environ["REMOTE_ELASTICSEARCH_ENDPOINT"]
DST_HOST        = os.environ.get("ELASTICSEARCH_ENDPOINT", "http://localhost:9200")
INDEX           = "wiki_full_l"
USER            = os.environ["REMOTE_ELASTICSEARCH_USERNAME"]
PASSWORD        = os.environ["REMOTE_ELASTICSEARCH_PASSWORD"]
BATCH_SIZE      = 2000      # docs per scroll page / bulk request
SCROLL_TTL      = "10m"     # generous TTL — each worker refreshes on every page
DEFAULT_WORKERS = 6         # concurrent scroll+bulk workers
MAX_WORKER_RETRIES = 3      # times a worker will re-open its scroll on transient failure
CHECKPOINT_FILE = Path("tmp_transfer_checkpoint.json")
LOG_FILE        = "tmp_transfer_index.log"

# === Logging ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

AUTH    = base64.b64encode(f"{USER}:{PASSWORD}".encode()).decode()
HEADERS = {"Content-Type": "application/json", "Authorization": f"Basic {AUTH}"}


# ── HTTP helper ──────────────────────────────────────────────────────────────

# Transient errors that are safe to retry
_RETRYABLE = (
    TimeoutError,
    ConnectionResetError,
    ConnectionRefusedError,
    urllib.error.URLError,   # wraps socket.timeout, [Errno 60], [Errno 61], [Errno 65]
)
# Backoff schedule in seconds: 5s, 15s, 30s, 60s, 120s
_BACKOFF = [5, 15, 30, 60, 120]


def request(
    url: str,
    *,
    method: str = "GET",
    body: dict | None = None,
    timeout: int = 60,
    max_retries: int = 6,
) -> dict:
    """Make an authenticated JSON request, retrying on transient network errors.

    Args:
        url: Full URL to request.
        method: HTTP method.
        body: Optional JSON body dict.
        timeout: Per-attempt socket timeout in seconds.
        max_retries: Maximum number of retry attempts after the first failure.

    Returns:
        Parsed JSON response dict.

    Raises:
        RuntimeError: On non-retryable HTTP errors or exhausted retries.
    """
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, headers=HEADERS, method=method)

    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as res:
                return json.loads(res.read())
        except urllib.error.HTTPError as e:
            # 429 / 503 are retryable; everything else is a hard error
            if e.code in (429, 503) and attempt < max_retries:
                delay = _BACKOFF[min(attempt, len(_BACKOFF) - 1)]
                logger.warning(
                    "HTTP %d %s %s — retry %d/%d in %ds",
                    e.code, method, url, attempt + 1, max_retries, delay,
                )
                time.sleep(delay)
                continue
            raise RuntimeError(f"HTTP {e.code} {method} {url}: {e.read().decode()}") from e
        except _RETRYABLE as e:
            if attempt >= max_retries:
                raise RuntimeError(
                    f"Network error after {max_retries} retries — {method} {url}: {e}"
                ) from e
            delay = _BACKOFF[min(attempt, len(_BACKOFF) - 1)]
            logger.warning(
                "Network error %s %s (%s) — retry %d/%d in %ds",
                method, url, e, attempt + 1, max_retries, delay,
            )
            time.sleep(delay)

    # unreachable, but satisfies type checkers
    raise RuntimeError(f"Exhausted retries for {method} {url}")


# ── Index management ─────────────────────────────────────────────────────────

MAPPING = {
    "mappings": {
        "properties": {
            "metadata": {
                "properties": {
                    "popularity_avg":  {"type": "float"},
                    "popularity_rank": {"type": "float"},
                    "wikipedia_id":    {"type": "long"},
                    "wikipedia_title": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                    },
                }
            },
            "text": {
                "type": "text",
                "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
            },
            # bbq_hnsw — quantised vectors, ~40% smaller than plain hnsw.
            "vector": {
                "type": "dense_vector",
                "dims": 1024,
                "index": True,
                "similarity": "cosine",
                "index_options": {
                    "type": "bbq_hnsw",
                    "m": 16,
                    "ef_construction": 100,
                    "rescore_vector": {"oversample": 3.0},
                },
            },
        }
    }
}


def ensure_index() -> None:
    """Create the destination index with the correct mapping if it doesn't exist."""
    try:
        request(f"{DST_HOST}/{INDEX}")
        logger.info("Index %s already exists — resuming.", INDEX)
    except RuntimeError:
        request(f"{DST_HOST}/{INDEX}", method="PUT", body=MAPPING)
        logger.info("Created index %s on destination.", INDEX)


def delete_index() -> None:
    """Delete the local index if it exists."""
    try:
        request(f"{DST_HOST}/{INDEX}", method="DELETE")
        logger.info("Deleted local index %s.", INDEX)
    except RuntimeError:
        logger.info("Local index %s did not exist.", INDEX)


def remote_count() -> int:
    res = request(f"{SRC_HOST}/{INDEX}/_count")
    return res["count"]


def local_count() -> int:
    """Return local doc count without forcing a refresh (cheap)."""
    try:
        res = request(f"{DST_HOST}/{INDEX}/_count")
        return res["count"]
    except RuntimeError:
        return 0


# ── Checkpoint ───────────────────────────────────────────────────────────────

_checkpoint_lock = threading.Lock()


def load_checkpoint() -> int | None:
    """Return the last safe resume wikipedia_id, or None if no checkpoint.

    The stored value is the MINIMUM confirmed id across all workers — every doc
    with wikipedia_id < this value has been fully indexed.  On resume we restart
    from this id (inclusive) after purging all local docs >= this id.
    """
    if CHECKPOINT_FILE.exists():
        data = json.loads(CHECKPOINT_FILE.read_text())
        wid = data.get("safe_resume_id")
        logger.info("Checkpoint found: safe_resume_id=%s", wid)
        return wid
    return None


def save_checkpoint(per_worker_min: dict[int, int | None]) -> None:
    """Persist the minimum confirmed wikipedia_id across all active workers.

    Args:
        per_worker_min: mapping of worker_id -> lowest wikipedia_id whose batch
                        has been fully bulk-indexed, or None if not yet started.
    """
    confirmed = [v for v in per_worker_min.values() if v is not None]
    if not confirmed:
        return
    # The safe boundary is the minimum across workers — everything below this
    # has been confirmed by every worker that covers that id range.
    safe_id = min(confirmed)
    with _checkpoint_lock:
        existing = None
        if CHECKPOINT_FILE.exists():
            existing = json.loads(CHECKPOINT_FILE.read_text()).get("safe_resume_id")
        # Only write if the new safe boundary is higher than the stored one
        if existing is None or safe_id > existing:
            CHECKPOINT_FILE.write_text(json.dumps({"safe_resume_id": safe_id}))


def clear_checkpoint() -> None:
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        logger.info("Checkpoint cleared.")


# ── Resume helpers ────────────────────────────────────────────────────────────

def delete_local_gte(wikipedia_id: int) -> int:
    """Delete all local docs with wikipedia_id >= the given value.

    This ensures no partial or out-of-order batches survive a resume.
    Returns count of deleted docs.
    """
    body = {"query": {"range": {"metadata.wikipedia_id": {"gte": wikipedia_id}}}}
    res = request(
        f"{DST_HOST}/{INDEX}/_delete_by_query?refresh=true&scroll_size=5000",
        method="POST",
        body=body,
        timeout=300,
    )
    deleted = res.get("deleted", 0)
    logger.info(
        "Deleted %s local docs with wikipedia_id >= %s (safe resume cleanup)",
        f"{deleted:,}",
        wikipedia_id,
    )
    return deleted


# ── Scroll helpers ────────────────────────────────────────────────────────────

def open_scroll(
    resume_from: int | None,
    *,
    worker_id: int,
    n_workers: int,
) -> tuple[str, list[dict]]:
    """Open a sliced scroll on the remote for this worker's partition.

    Uses ES scroll slicing so each worker fetches a disjoint subset of docs.
    resume_from applies a wikipedia_id >= filter on top of the slice.
    """
    query: dict = (
        {"range": {"metadata.wikipedia_id": {"gte": resume_from}}}
        if resume_from is not None
        else {"match_all": {}}
    )
    body: dict = {
        "query": query,
        "sort": [{"metadata.wikipedia_id": "asc"}],
        "_source": True,
        "fields": ["vector"],
    }
    if n_workers > 1:
        body["slice"] = {"id": worker_id, "max": n_workers}

    res = request(
        f"{SRC_HOST}/{INDEX}/_search?scroll={SCROLL_TTL}&size={BATCH_SIZE}",
        method="POST",
        body=body,
        timeout=180,
    )
    return res["_scroll_id"], res["hits"]["hits"]


def next_scroll(scroll_id: str) -> tuple[str, list[dict]]:
    res = request(
        f"{SRC_HOST}/_search/scroll",
        method="POST",
        body={"scroll": SCROLL_TTL, "scroll_id": scroll_id},
        timeout=180,
    )
    return res["_scroll_id"], res["hits"]["hits"]


def close_scroll(scroll_id: str) -> None:
    try:
        request(f"{SRC_HOST}/_search/scroll", method="DELETE", body={"scroll_id": scroll_id})
    except Exception:
        pass


# ── Bulk index ────────────────────────────────────────────────────────────────

def _build_bulk_body(hits: list[dict]) -> bytes:
    lines = []
    for hit in hits:
        doc = dict(hit["_source"])
        if "vector" in hit.get("fields", {}):
            doc["vector"] = hit["fields"]["vector"]
        lines.append(json.dumps({"index": {"_index": INDEX, "_id": hit["_id"]}}))
        lines.append(json.dumps(doc))
    return ("\n".join(lines) + "\n").encode()


def bulk_index(hits: list[dict], *, retry: bool = True) -> tuple[int, int]:
    """Bulk index hits into the local index.

    Returns (indexed, failed). Failed items are retried once if retry=True.
    Uses refresh=false for speed; caller does a single refresh at the end.
    """
    body = _build_bulk_body(hits)
    req = urllib.request.Request(
        f"{DST_HOST}/_bulk?refresh=false", data=body, headers=HEADERS, method="POST"
    )
    with urllib.request.urlopen(req, timeout=120) as res:
        result = json.loads(res.read())

    ok_hits   = [hits[i] for i, item in enumerate(result["items"]) if item["index"]["status"] in (200, 201)]
    fail_hits = [hits[i] for i, item in enumerate(result["items"]) if item["index"]["status"] not in (200, 201)]

    if fail_hits and retry:
        logger.warning("Retrying %d failed docs in this batch.", len(fail_hits))
        time.sleep(1)
        r_ok, r_fail = bulk_index(fail_hits, retry=False)
        return len(ok_hits) + r_ok, r_fail

    return len(ok_hits), len(fail_hits)


# ── Worker ────────────────────────────────────────────────────────────────────

def worker_transfer(
    *,
    worker_id: int,
    n_workers: int,
    resume_from: int | None,
    counters: dict,
    counters_lock: threading.Lock,
    stop_event: threading.Event,
    per_worker_min: dict[int, int | None],
) -> None:
    """Scroll + bulk worker. Runs one sliced scroll to completion.

    On transient failure the worker re-opens its scroll from the last confirmed
    wikipedia_id and retries up to MAX_WORKER_RETRIES times independently.
    It does NOT set stop_event on error — only a fatal unrecoverable failure
    (retries exhausted) propagates the exception to the executor, which then
    sets stop_event in transfer().

    When the slice is fully consumed the worker sets its per_worker_min entry
    to sys.maxsize so it no longer drags down the global checkpoint minimum.
    """
    # The point from which this worker will (re-)open its scroll.
    # Starts at the global resume_from; advances as batches are confirmed.
    worker_resume_from: int | None = resume_from

    for attempt in range(MAX_WORKER_RETRIES + 1):
        scroll_id: str | None = None
        try:
            scroll_id, hits = open_scroll(
                worker_resume_from, worker_id=worker_id, n_workers=n_workers
            )
            logger.info(
                "Worker %d scroll opened (slice %d/%d, attempt %d, resume_from=%s).",
                worker_id, worker_id, n_workers, attempt + 1, worker_resume_from,
            )

            while hits and not stop_event.is_set():
                indexed, failed = bulk_index(hits)

                batch_min_wid = int(hits[0]["_source"]["metadata"]["wikipedia_id"])
                batch_max_wid = int(hits[-1]["_source"]["metadata"]["wikipedia_id"])

                with counters_lock:
                    current = per_worker_min.get(worker_id)
                    if current is None or batch_min_wid > current:
                        per_worker_min[worker_id] = batch_min_wid
                    save_checkpoint(per_worker_min)

                    counters["indexed"] += indexed
                    counters["failed"]  += failed
                    counters["batches"] += 1

                # Advance the per-worker resume point so a retry starts past
                # the last confirmed batch rather than re-sending it.
                worker_resume_from = batch_max_wid + 1

                scroll_id, hits = next_scroll(scroll_id)

            # Slice fully consumed — signal that this worker is no longer the
            # bottleneck for the global checkpoint minimum.
            if not stop_event.is_set():
                with counters_lock:
                    per_worker_min[worker_id] = sys.maxsize
                    save_checkpoint(per_worker_min)
                logger.info("Worker %d done (slice exhausted).", worker_id)
                return  # success — exit retry loop

        except Exception as e:
            if scroll_id is not None:
                close_scroll(scroll_id)
                scroll_id = None

            if stop_event.is_set():
                # Another worker already triggered a hard stop; just bail.
                logger.info("Worker %d stopping due to global stop_event.", worker_id)
                return

            if attempt < MAX_WORKER_RETRIES:
                delay = _BACKOFF[min(attempt, len(_BACKOFF) - 1)]
                logger.warning(
                    "Worker %d error (attempt %d/%d), retrying in %ds from wikipedia_id=%s: %s",
                    worker_id, attempt + 1, MAX_WORKER_RETRIES, delay, worker_resume_from, e,
                )
                time.sleep(delay)
                continue
            else:
                logger.error(
                    "Worker %d failed after %d retries: %s",
                    worker_id, MAX_WORKER_RETRIES, e,
                )
                raise  # propagate to transfer() which will set stop_event
        finally:
            if scroll_id is not None:
                close_scroll(scroll_id)


# ── Progress reporter ─────────────────────────────────────────────────────────

def progress_reporter(
    *,
    total: int,
    counters: dict,
    counters_lock: threading.Lock,
    stop_event: threading.Event,
    start: float,
    interval: float = 15.0,
) -> None:
    """Logs progress every `interval` seconds until stop_event is set."""
    while not stop_event.is_set():
        time.sleep(interval)
        if stop_event.is_set():
            break
        with counters_lock:
            indexed = counters["indexed"]
            failed  = counters["failed"]
            batches = counters["batches"]
        elapsed   = time.time() - start
        rate      = indexed / elapsed if elapsed > 0 else 0
        pct       = 100 * indexed / total if total > 0 else 0
        remaining = (total - indexed) / rate if rate > 0 else 0
        eta       = str(timedelta(seconds=int(remaining)))
        logger.info(
            "Progress | batches=%d | indexed=%s/%s (%.1f%%) | failed=%s | rate=%.0f/s | ETA=%s",
            batches,
            f"{indexed:,}",
            f"{total:,}",
            pct,
            f"{failed:,}",
            rate,
            eta,
        )


# ── Transfer ──────────────────────────────────────────────────────────────────

def transfer(
    *,
    reset_checkpoint: bool = False,
    fresh: bool = False,
    n_workers: int = DEFAULT_WORKERS,
) -> None:
    """Run the full transfer with concurrent sliced scroll workers."""
    if fresh:
        clear_checkpoint()
        delete_index()

    ensure_index()

    if reset_checkpoint and not fresh:
        clear_checkpoint()

    total = remote_count()
    logger.info("Remote has %s docs. Starting transfer with %d workers.", f"{total:,}", n_workers)

    resume_from = load_checkpoint()
    if resume_from is not None:
        delete_local_gte(resume_from)

    counters: dict             = {"indexed": 0, "failed": 0, "batches": 0}
    counters_lock              = threading.Lock()
    stop_event                 = threading.Event()
    per_worker_min: dict[int, int | None] = {i: None for i in range(n_workers)}
    start                      = time.time()

    reporter = threading.Thread(
        target=progress_reporter,
        kwargs=dict(
            total=total,
            counters=counters,
            counters_lock=counters_lock,
            stop_event=stop_event,
            start=start,
            interval=15.0,
        ),
        daemon=True,
    )
    reporter.start()

    try:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(
                    worker_transfer,
                    worker_id=i,
                    n_workers=n_workers,
                    resume_from=resume_from,
                    counters=counters,
                    counters_lock=counters_lock,
                    stop_event=stop_event,
                    per_worker_min=per_worker_min,
                ): i
                for i in range(n_workers)
            }
            for future in as_completed(futures):
                wid = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error("Worker %d raised: %s", wid, e)
                    stop_event.set()

    except KeyboardInterrupt:
        stop_event.set()
        logger.info("Interrupted. Checkpoint saved. Re-run to resume.")
    finally:
        stop_event.set()

    # Final refresh + verification
    request(f"{DST_HOST}/{INDEX}/_refresh", method="POST")
    final   = local_count()
    elapsed = time.time() - start
    with counters_lock:
        total_failed = counters["failed"]

    logger.info(
        "Transfer complete. local=%s/%s | failed=%s | time=%s",
        f"{final:,}",
        f"{total:,}",
        f"{total_failed:,}",
        str(timedelta(seconds=int(elapsed))),
    )
    if final >= total:
        clear_checkpoint()
    else:
        logger.warning(
            "%d docs missing. Re-run to resume from checkpoint.",
            total - final,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer wiki_full_l from remote ES to local.")
    parser.add_argument(
        "--reset-checkpoint",
        action="store_true",
        help="Clear checkpoint and re-transfer from the beginning (keeps existing local docs).",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete local index + clear checkpoint and start completely from scratch.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of concurrent scroll+bulk workers (default: {DEFAULT_WORKERS}).",
    )
    args = parser.parse_args()
    transfer(reset_checkpoint=args.reset_checkpoint, fresh=args.fresh, n_workers=args.workers)
