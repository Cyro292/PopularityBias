"""latency_utils.py — Latency tracking helpers for the RAG evaluation pipeline.

Each pipeline stage writes a ``latency_<key>.json`` file alongside its CSV
checkpoint.  The file contains per-question timings and summary statistics so
that end-to-end latency (question → retrieval → LLM answer) can be analysed
across backends and popularity deciles.

Schema
------
::

    {
        "backend_key": "ivfpq_high",
        "stage": "retrieval",          # "retrieval" | "generation"
        "n_questions": 6968,
        "total_s": 42.1,
        "avg_ms": 6.04,
        "p50_ms": 5.8,
        "p95_ms": 11.2,
        "p99_ms": 18.4,
        "per_question": [              # one entry per question, in order
            {"question_id": "...", "latency_ms": 5.2},
            ...
        ]
    }
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable, TypeVar

import numpy as np

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ═══════════════════════════════════════════════════════════════════════════════
# Core helpers
# ═══════════════════════════════════════════════════════════════════════════════

def time_batch(
    fn: Callable[[], T],
    question_ids: list[str],
) -> tuple[T, list[float]]:
    """Run *fn* and split total wall-clock time evenly across questions.

    This is used when a backend processes all questions in one batched call
    (e.g. FAISS matrix search, ES msearch).  The per-question latency is the
    average — not individual measurements — because batched execution does not
    expose per-item timings.

    Args:
        fn: Zero-argument callable that executes the batch operation.
        question_ids: IDs of the questions being processed (used for length).

    Returns:
        Tuple of (fn's return value, list of per-question latency in ms).
    """
    t0     = time.perf_counter()
    result = fn()
    total_s = time.perf_counter() - t0
    avg_ms  = (total_s / max(len(question_ids), 1)) * 1000.0
    latencies_ms = [avg_ms] * len(question_ids)
    return result, latencies_ms


def save_latency(
    *,
    path: Path,
    backend_key: str,
    stage: str,
    question_ids: list[str],
    latencies_ms: list[float],
) -> None:
    """Persist latency measurements to *path* as JSON.

    Args:
        path: Destination file (created / overwritten).
        backend_key: Backend identifier (e.g. ``"ivfpq_high"``).
        stage: Pipeline stage label (``"retrieval"`` or ``"generation"``).
        question_ids: Question IDs in the same order as *latencies_ms*.
        latencies_ms: Per-question latency in milliseconds.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.array(latencies_ms, dtype=float)
    record = {
        "backend_key": backend_key,
        "stage":       stage,
        "n_questions": len(question_ids),
        "total_s":     round(float(arr.sum() / 1000.0), 4),
        "avg_ms":      round(float(arr.mean()), 4),
        "p50_ms":      round(float(np.percentile(arr, 50)), 4),
        "p95_ms":      round(float(np.percentile(arr, 95)), 4),
        "p99_ms":      round(float(np.percentile(arr, 99)), 4),
        "per_question": [
            {"question_id": qid, "latency_ms": round(float(ms), 4)}
            for qid, ms in zip(question_ids, latencies_ms)
        ],
    }
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    logger.info(
        "[%s] Latency (%s): avg=%.1f ms  p95=%.1f ms  total=%.1f s → %s",
        backend_key, stage, record["avg_ms"], record["p95_ms"], record["total_s"], path.name,
    )


def load_latency(path: Path) -> dict | None:
    """Load a latency JSON file.

    Args:
        path: Path written by :func:`save_latency`.

    Returns:
        Parsed dict, or ``None`` if the file does not exist.
    """
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def merge_latency(
    retrieval: dict | None,
    generation: dict | None,
) -> dict | None:
    """Combine retrieval and generation latency records into end-to-end stats.

    Per-question end-to-end latency = retrieval_ms + generation_ms.

    Args:
        retrieval: Record from :func:`load_latency` for the retrieval stage.
        generation: Record from :func:`load_latency` for the generation stage.

    Returns:
        Dict with combined stats, or ``None`` if both inputs are ``None``.
    """
    if retrieval is None and generation is None:
        return None

    def _ms_map(record: dict | None) -> dict[str, float]:
        if record is None:
            return {}
        return {e["question_id"]: e["latency_ms"] for e in record.get("per_question", [])}

    ret_map = _ms_map(retrieval)
    gen_map = _ms_map(generation)
    all_ids = sorted(set(ret_map) | set(gen_map))

    e2e_ms = [
        ret_map.get(qid, 0.0) + gen_map.get(qid, 0.0)
        for qid in all_ids
    ]
    arr = np.array(e2e_ms, dtype=float)

    return {
        "backend_key":    (retrieval or generation)["backend_key"],
        "stage":          "end_to_end",
        "n_questions":    len(all_ids),
        "total_s":        round(float(arr.sum() / 1000.0), 4),
        "avg_ms":         round(float(arr.mean()), 4),
        "p50_ms":         round(float(np.percentile(arr, 50)), 4),
        "p95_ms":         round(float(np.percentile(arr, 95)), 4),
        "p99_ms":         round(float(np.percentile(arr, 99)), 4),
        "retrieval_avg_ms":  round(float(np.mean(list(ret_map.values()))), 4) if ret_map else None,
        "generation_avg_ms": round(float(np.mean(list(gen_map.values()))), 4) if gen_map else None,
        "per_question": [
            {
                "question_id":   qid,
                "retrieval_ms":  round(ret_map.get(qid, 0.0), 4),
                "generation_ms": round(gen_map.get(qid, 0.0), 4),
                "total_ms":      round(ret_map.get(qid, 0.0) + gen_map.get(qid, 0.0), 4),
            }
            for qid in all_ids
        ],
    }
