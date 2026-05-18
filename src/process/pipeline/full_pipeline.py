"""End-to-end RAG evaluation pipeline.

Runs all three stages in sequence:
  Stage 1 — Retrieval   (retrieved_docs_<key>.csv)
  Stage 2 — Generation  (answer_checkpoint_<llm>_<key>_top<n>.csv)
  Stage 3 — Evaluation  (results_<llm>_<key>_top<n>.parquet)

Each stage respects its own checkpoints — already-done work is skipped unless
you pass a restart flag.

Usage
-----
::

    # Full run (skip stages whose outputs already exist)
    python scripts/run_pipeline.py

    # Wipe and redo everything
    python scripts/run_pipeline.py --restart

    # Only redo retrieval for specific backends, keep generation + eval
    python scripts/run_pipeline.py --restart-retrieval-keys bm25_plus faiss_hybrid

    # Skip retrieval entirely (use existing CSVs), redo generation + eval
    python scripts/run_pipeline.py --skip-retrieval --restart-generation --restart-eval

    # Run only a subset of backends end-to-end
    python scripts/run_pipeline.py --only-keys bm25_plus ivfpq_high faiss_hybrid

    # Restrict to one LLM
    python scripts/run_pipeline.py --models neo
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import dotenv
dotenv.load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("elastic_transport.transport").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

from src.process.pipeline.retrieval_runner import RetrievalConfig, RetrievalRunner
from src.process.pipeline.generating_runner import GeneratingConfig, GenerationBackend, LLMBackend
from src.process.pipeline.llm_eval_runner import EvalConfig, EvalBackend, LLMEvalRunner

# Import GeneratingRunner (class name may vary — handle both)
try:
    from src.process.pipeline.generating_runner import GeneratingRunner
except ImportError:
    from src.process.pipeline.generating_runner import GenerationRunner as GeneratingRunner  # type: ignore[no-redef]


# ═══════════════════════════════════════════════════════════════════════════════
# Defaults
# ═══════════════════════════════════════════════════════════════════════════════

_DEFAULT_BACKENDS: list[str] = [
    "zero_shot",
    "bm25_plus",
    "ivfpq_high",
    "es_approx",
    "es_hybrid",
    "router",
    "router_es",
    "faiss_hybrid",
]

_DEFAULT_MODELS: list[str] = ["neo", "qwen"]

_DEFAULT_CONTEXT_SIZES: list[int] = [3]


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _fmt_seconds(s: float) -> str:
    """Format elapsed seconds as a human-readable string."""
    h, rem = divmod(int(s), 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {sec}s"
    if m:
        return f"{m}m {sec}s"
    return f"{sec}s"


# ═══════════════════════════════════════════════════════════════════════════════
# Stage runners
# ═══════════════════════════════════════════════════════════════════════════════

def run_retrieval(
    *,
    collection: str,
    output_dir: str,
    top_k: int,
    questions_per_decile: int,
    only_keys: tuple[str, ...],
    restart: bool,
    restart_keys: tuple[str, ...],
) -> RetrievalConfig:
    """Run Stage 1 — retrieval.

    Returns the :class:`RetrievalConfig` used, so downstream stages can share it.
    """
    cfg = RetrievalConfig(
        collection_name=collection,
        output_dir=output_dir,
        top_k=top_k,
        questions_per_decile=questions_per_decile,
        restart=restart,
        restart_keys=restart_keys,
        only_keys=only_keys,
    )
    t0 = time.time()
    logger.info("═" * 60)
    logger.info("STAGE 1 — RETRIEVAL")
    logger.info("═" * 60)
    RetrievalRunner(cfg).run()
    logger.info("Stage 1 done in %s", _fmt_seconds(time.time() - t0))
    return cfg


def run_generation(
    retrieval_cfg: RetrievalConfig,
    *,
    backends: list[str],
    models: list[str],
    context_sizes: list[int],
    restart: bool,
) -> GeneratingConfig:
    """Run Stage 2 — generation.

    Returns the :class:`GeneratingConfig` used, so Stage 3 can share it.
    """
    gen_backends = [GenerationBackend(key=k) for k in backends]
    llm_models   = [LLMBackend(key=m, type=m) for m in models]

    cfg = GeneratingConfig(
        retrieval=retrieval_cfg,
        backends=gen_backends,
        models=llm_models,
        context_sizes=context_sizes,
        restart=restart,
    )
    t0 = time.time()
    logger.info("═" * 60)
    logger.info("STAGE 2 — GENERATION")
    logger.info("═" * 60)
    GeneratingRunner(cfg).run()
    logger.info("Stage 2 done in %s", _fmt_seconds(time.time() - t0))
    return cfg


def run_eval(
    generating_cfg: GeneratingConfig,
    *,
    backends: list[str],
    models: list[str],
    context_sizes: list[int],
    restart: bool,
) -> None:
    """Run Stage 3 — evaluation."""
    cfg = EvalConfig(
        generating=generating_cfg,
        backends=backends,
        models=models,
        context_sizes=context_sizes,
        evaluators=[EvalBackend(key="substring", type="substring")],
        restart=restart,
    )
    t0 = time.time()
    logger.info("═" * 60)
    logger.info("STAGE 3 — EVALUATION")
    logger.info("═" * 60)
    LLMEvalRunner(cfg).run()
    logger.info("Stage 3 done in %s", _fmt_seconds(time.time() - t0))


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-end RAG evaluation pipeline (retrieval → generation → eval).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Scope ──────────────────────────────────────────────────────────────────
    p.add_argument(
        "--only-keys", nargs="+", default=_DEFAULT_BACKENDS, metavar="KEY",
        help="Backend keys to run through all three stages.",
    )
    p.add_argument(
        "--models", nargs="+", default=_DEFAULT_MODELS, metavar="MODEL",
        help="LLM keys to use for generation and eval.",
    )
    p.add_argument(
        "--context-sizes", nargs="+", type=int, default=_DEFAULT_CONTEXT_SIZES, metavar="N",
        help="Context window sizes (number of retrieved docs fed to LLM).",
    )

    # ── Collection / retrieval settings ───────────────────────────────────────
    p.add_argument("--collection", "-c", default="wiki_full_bil")
    p.add_argument("--output-dir", "-o", default="all_qa_8k")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--questions-per-decile", type=int, default=800)

    # ── Skip flags ────────────────────────────────────────────────────────────
    p.add_argument("--skip-retrieval",  action="store_true", help="Skip Stage 1 (use existing CSVs).")
    p.add_argument("--skip-generation", action="store_true", help="Skip Stage 2 (use existing answer checkpoints).")
    p.add_argument("--skip-eval",       action="store_true", help="Skip Stage 3.")

    # ── Restart flags ─────────────────────────────────────────────────────────
    p.add_argument("--restart",            action="store_true", help="Wipe and redo all three stages.")
    p.add_argument("--restart-retrieval",  action="store_true", help="Wipe and redo Stage 1 only.")
    p.add_argument("--restart-generation", action="store_true", help="Wipe and redo Stage 2 only.")
    p.add_argument("--restart-eval",       action="store_true", help="Wipe and redo Stage 3 only.")
    p.add_argument(
        "--restart-retrieval-keys", nargs="+", default=[], metavar="KEY",
        help="Wipe retrieval CSVs only for these backend keys.",
    )

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    # Resolve effective restart flags
    restart_retrieval  = args.restart or args.restart_retrieval
    restart_generation = args.restart or args.restart_generation
    restart_eval       = args.restart or args.restart_eval

    # Backends active for generation and eval — always the full only_keys list
    # (retrieval --only-keys may be narrower during --restart-retrieval-keys runs)
    active_backends = args.only_keys

    t_total = time.time()
    logger.info("Pipeline starting — backends: %s", active_backends)
    logger.info("Models: %s | context_sizes: %s", args.models, args.context_sizes)

    # ── Stage 1: Retrieval ────────────────────────────────────────────────────
    if args.skip_retrieval:
        logger.info("Stage 1 — SKIPPED (--skip-retrieval)")
        # Still need a RetrievalConfig to pass downstream
        retrieval_cfg = RetrievalConfig(
            collection_name=args.collection,
            output_dir=args.output_dir,
            top_k=args.top_k,
            questions_per_decile=args.questions_per_decile,
        )
    else:
        retrieval_cfg = run_retrieval(
            collection=args.collection,
            output_dir=args.output_dir,
            top_k=args.top_k,
            questions_per_decile=args.questions_per_decile,
            only_keys=tuple(active_backends),
            restart=restart_retrieval,
            restart_keys=tuple(args.restart_retrieval_keys),
        )

    # ── Stage 2: Generation ───────────────────────────────────────────────────
    if args.skip_generation:
        logger.info("Stage 2 — SKIPPED (--skip-generation)")
        generating_cfg = GeneratingConfig(
            retrieval=retrieval_cfg,
            backends=[GenerationBackend(key=k) for k in active_backends],
            models=[LLMBackend(key=m, type=m) for m in args.models],
            context_sizes=args.context_sizes,
        )
    else:
        generating_cfg = run_generation(
            retrieval_cfg,
            backends=active_backends,
            models=args.models,
            context_sizes=args.context_sizes,
            restart=restart_generation,
        )

    # ── Stage 3: Eval ─────────────────────────────────────────────────────────
    if args.skip_eval:
        logger.info("Stage 3 — SKIPPED (--skip-eval)")
    else:
        run_eval(
            generating_cfg,
            backends=active_backends,
            models=args.models,
            context_sizes=args.context_sizes,
            restart=restart_eval,
        )

    logger.info("═" * 60)
    logger.info("Pipeline complete in %s", _fmt_seconds(time.time() - t_total))
    logger.info("═" * 60)


if __name__ == "__main__":
    main()
