"""LLM eval runner — Stage 3 (final) of the RAG evaluation pipeline.

Picks up ``answer_checkpoint_{llm_key}_{retrieval_key}_top{n}.csv`` files
written by :mod:`src.process.retrieval.generating_runner`, evaluates each
proposed answer with a judge LLM, and writes
``results_{llm_key}_{retrieval_key}_top{n}.parquet`` files consumed by the
eval notebooks.

Checkpoint behaviour
--------------------
- Answer CSV     (``answer_checkpoint_*``) — read-only input.
- Eval JSONL     (``eval_checkpoint_*``)   — incremental; reused unless
  ``--restart`` is passed.
- Final parquet  (``results_*``)           — skipped if it already exists.

Usage
-----
::

    python -m src.process.retrieval.llm_eval_runner
    python -m src.process.retrieval.llm_eval_runner --restart
    python -m src.process.retrieval.llm_eval_runner --help
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

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

from config import DATA_DIR
from src.corpus_handler.parquet_corpus_handler import ParquetCorpusHandler
from src.question_input.huggingface_cyro_input import HuggingFaceCyroInput
from src.evaluator.binary_evaluator import BinaryEvaluator
from src.evaluator.substring_evaluator import SubstringEvaluator
from src.evaluator.base import EvaluationObjects, EvaluationResult, EvaluatorBase
from src.process.retrieval.retrieval_runner import (
    RetrievalConfig,
    RetrievalRunner,
    load_retrieved_docs_csv,
    save_retrieved_docs_csv,
)
from src.process.retrieval.generating_runner import (
    GeneratingConfig,
    GenerationBackend,
    LLMBackend,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Settings
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EvalBackend:
    """Configures a single evaluator to run.

    Attributes:
        key: Short identifier used in output filenames
            (e.g. ``"binary_mistral"``, ``"substring"``).
        type: Which evaluator to use — ``"binary"`` (LLM judge) or
            ``"substring"`` (exact string match).
        llm_type: LLM service to use as the judge for ``"binary"`` evaluators.
            One of ``"mistral"``, ``"neo"``, ``"openai"``, ``"qwen"``.
            Ignored for ``"substring"``.
        llm_model_name: Model name forwarded to the LLM service when required
            (e.g. ``"gpt-4o"`` for ``"openai"``).
        case_sensitive: For ``"substring"`` only — whether the match is
            case-sensitive.
        llm_kwargs: Extra keyword arguments passed to the LLM service
            constructor for ``"binary"`` evaluators.
    """

    key:             str
    type:            Literal["binary", "substring"]  = "binary"
    llm_type:        Literal["mistral", "neo", "openai", "qwen"] = "mistral"
    llm_model_name:  str | None                      = None
    case_sensitive:  bool                            = False
    llm_kwargs:      dict[str, Any]                  = field(default_factory=dict)


@dataclass(frozen=True)
class EvalConfig:
    """Configuration for the evaluation stage (Stage 3).

    Mirrors :class:`~src.process.retrieval.generating_runner.GeneratingConfig`
    so you select the exact same (model, backend, context_size) combinations
    that were generated.

    Attributes:
        generating: Full generating config (contains retrieval config, models,
            backends, context_sizes).  Defaults to :class:`GeneratingConfig`
            defaults.
        models: Subset of LLM keys to evaluate.  Each key must match one in
            ``generating.models``.  Defaults to all.
        backends: Subset of retrieval backend keys to evaluate.  Each key must
            match one in ``generating.backends``.  Defaults to all.
        context_sizes: Subset of context sizes to evaluate.  Each value must
            exist in ``generating.context_sizes``.  Defaults to all.
        restart: If True, delete existing eval checkpoints and re-evaluate.
    """

    generating:    GeneratingConfig        = None   # type: ignore[assignment]
    models:        list[str]               = field(default_factory=lambda: ["neo", "qwen"])
    backends:      list[str]               = field(default_factory=lambda: ["zero_shot", "bm25_plus", "ivfpq_low", "ivfpq_high"])
    context_sizes: list[int]               = field(default_factory=lambda: [1, 3])
    evaluators:    list[EvalBackend]       = field(default_factory=lambda: [
        # EvalBackend(key="substring", type="substring"),
        EvalBackend(key="binary_mistral", type="binary", llm_type="mistral"),
    ])
    restart:       bool                    = False

    def __post_init__(self) -> None:
        if self.generating is None:
            object.__setattr__(self, "generating", GeneratingConfig())

    @property
    def retrieval(self) -> RetrievalConfig:
        return self.generating.retrieval


# ═══════════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════════

class LLMEvalRunner:
    """Stage 3 runner — evaluates generated answers and writes final parquet files.

    Args:
        cfg: Active :class:`EvalConfig`.
    """

    def __init__(self, cfg: EvalConfig) -> None:
        self.cfg = cfg
        self._rcfg              = cfg.retrieval
        self._gcfg              = cfg.generating
        self._collection_folder = DATA_DIR / self._rcfg.collection_name
        self._output_folder     = self._collection_folder / self._rcfg.output_dir

    # ── Evaluator factory ─────────────────────────────────────────────────────

    @staticmethod
    def _build_evaluator(eb: EvalBackend) -> EvaluatorBase:
        """Instantiate the appropriate evaluator for *eb*.

        Args:
            eb: :class:`EvalBackend` descriptor from the config.

        Returns:
            A ready-to-use :class:`~src.evaluator.base.EvaluatorBase` instance.

        Raises:
            ValueError: If ``eb.type`` is not a recognised evaluator type.
        """
        if eb.type == "substring":
            return SubstringEvaluator(case_sensitive=eb.case_sensitive)
        elif eb.type == "binary":
            kwargs = eb.llm_kwargs
            if eb.llm_type == "mistral":
                from src.llm.mistralLLMService import MistralLLMService
                llm = MistralLLMService(**kwargs)
            elif eb.llm_type == "neo":
                from src.llm.gptNeo27bLLMService import GPTNeo27bLLMService
                llm = GPTNeo27bLLMService(**kwargs)
            elif eb.llm_type == "openai":
                from src.llm.openAi_service import OpenAIService
                model_name = eb.llm_model_name or "gpt-4o-mini"
                llm = OpenAIService(model_name=model_name, **kwargs)
            elif eb.llm_type == "qwen":
                from src.llm.qwenLLMService import QwenLLMService
                llm = QwenLLMService(**kwargs)
            else:
                raise ValueError(f"Unknown LLM type for binary evaluator: {eb.llm_type!r}")
            return BinaryEvaluator(evaluation_service=llm)
        else:
            raise ValueError(f"Unknown evaluator type: {eb.type!r}")

    def run(self) -> None:
        """Evaluate all configured (model × backend × context_size) triples.

        For each combination:

        1. Skip if final parquet already exists (unless ``cfg.restart``).
        2. Load answer checkpoint — raises if missing (run generating_runner first).
        3. Load retrieval checkpoint for doc metadata.
        4. Run LLM judge evaluation, resuming from eval checkpoint if present.
        5. Write ``results_{llm_key}_{retrieval_key}_top{n}.parquet``.
        """
        self._output_folder.mkdir(parents=True, exist_ok=True)

        # ── Resolve active combinations ───────────────────────────────────────
        gen_model_keys   = {m.key for m in self._gcfg.models}
        gen_backend_keys = {b.key for b in self._gcfg.backends}
        gen_ctx_sizes    = set(self._gcfg.context_sizes)

        unknown_models   = set(self.cfg.models)   - gen_model_keys
        unknown_backends = set(self.cfg.backends) - gen_backend_keys
        unknown_sizes    = set(self.cfg.context_sizes) - gen_ctx_sizes
        if unknown_models:
            raise ValueError(f"EvalConfig.models contains keys not in generating config: {unknown_models}")
        if unknown_backends:
            raise ValueError(f"EvalConfig.backends contains keys not in generating config: {unknown_backends}")
        if unknown_sizes:
            raise ValueError(f"EvalConfig.context_sizes contains values not in generating config: {unknown_sizes}")

        active_models    = [m for m in self._gcfg.models   if m.key in set(self.cfg.models)]
        active_backends  = [b for b in self._gcfg.backends if b.key in set(self.cfg.backends)]
        active_sizes     = [n for n in self._gcfg.context_sizes if n in set(self.cfg.context_sizes)]

        # Also resolve full RetrievalBackend objects for retrieval checkpoints
        retrieval_backends = {b.key: b for b in self._rcfg.backends}

        n_zero_shot = sum(1 for b in active_backends if b.key == "zero_shot")
        n_retrieval = len(active_backends) - n_zero_shot
        n_total = len(active_models) * (n_zero_shot + n_retrieval * len(active_sizes)) * len(self.cfg.evaluators)
        logger.info(
            "Starting eval: %d model(s) × (%d zero-shot + %d backend(s) × %d context size(s)) × %d evaluator(s) = %d runs",
            len(active_models), n_zero_shot, n_retrieval, len(active_sizes), len(self.cfg.evaluators), n_total,
        )

        # ── Load questions ────────────────────────────────────────────────────
        corpus_handler = ParquetCorpusHandler(
            corpus_path=self._collection_folder / "wiki_corpus.parquet",
            metadata_path=self._collection_folder / "metadata.json",
        )
        question_input = HuggingFaceCyroInput(
            dataset_names=list(self._rcfg.dataset_names),
            corpus_handler=corpus_handler,
            parquet_path=self._output_folder / "cyro_qa_cache.parquet",
            balance_deciles=True,
            balance_datasets=True,
            target_per_decile=self._rcfg.questions_per_decile,
            shuffle=True,
            balance_decile_mode=self._rcfg.balance_decile_mode,
        )
        question_input.load()
        question_data = question_input.get_items()
        logger.info("Loaded %d questions", len(question_data))

        question_ids = [item.question_id for item in question_data]

        run_idx = 0
        for llm_model in active_models:
            logger.info("── Model: %s ──", llm_model.key)

            for backend in active_backends:
                retrieval_checkpoint = self._output_folder / f"retrieved_docs_{backend.key}.csv"
                retrieved_docs = load_retrieved_docs_csv(retrieval_checkpoint, question_ids)
                if retrieved_docs is None:
                    logger.warning(
                        "  [%s] no retrieval checkpoint found — doc metadata will be empty",
                        backend.key,
                    )
                    retrieved_docs = [[] for _ in question_data]

                if len(retrieved_docs) != len(question_data):
                    retrieved_docs = retrieved_docs[: len(question_data)]

                for ctx_n in ([active_sizes[0]] if backend.key == "zero_shot" else active_sizes):
                    # zero_shot: always use the first context_size checkpoint (all are identical)
                    ctx_label = "zero" if backend.key == "zero_shot" else f"top{ctx_n}"
                    answer_checkpoint = (
                        self._output_folder
                        / f"answer_checkpoint_{llm_model.key}_{backend.key}_top{ctx_n}.csv"
                    )

                    # ── Load answer checkpoint once per (model, backend, ctx_n) ─
                    if not answer_checkpoint.exists():
                        raise FileNotFoundError(
                            f"Answer checkpoint not found: {answer_checkpoint}\n"
                            f"Run generating_runner first."
                        )
                    with open(answer_checkpoint, newline="", encoding="utf-8") as f:
                        rows = list(csv.DictReader(f))
                    # Build a question_id → answer mapping so that answers are
                    # always paired with the correct question regardless of the
                    # order rows were written to the checkpoint CSV.
                    answer_by_id: dict[str, str] = {
                        str(r["question_id"]): r.get("answer", r.get("generated_answer", ""))
                        for r in rows
                        if "question_id" in r
                    }
                    # Fall back to positional list when question_id is absent
                    # (old-format checkpoints).
                    positional_answers = [
                        r.get("answer", r.get("generated_answer", "")) for r in rows
                    ]
                    if len(positional_answers) > len(question_data):
                        positional_answers = positional_answers[: len(question_data)]

                    def _get_answer(q: "QuestionData", pos: int) -> str:  # noqa: F821
                        if answer_by_id:
                            ans = answer_by_id.get(str(q.question_id))
                            if ans is not None:
                                return ans
                        # question_id not found in map — use positional fallback
                        return positional_answers[pos] if pos < len(positional_answers) else ""

                    # ── Build evaluation objects once per (model, backend, ctx_n) ─
                    docs_sliced = [docs[:ctx_n] for docs in retrieved_docs]
                    evaluation_objects = [
                        EvaluationObjects(
                            id=q.question_id,
                            question=q.question_text,
                            answers=q.answer_texts,
                            page_content=q.page_content,
                            proposed_answer=_get_answer(q, i),
                            retrieved_docs=docs,
                            metadata={
                                "llm_key":                  llm_model.key,
                                "wikipedia_id":             q.wikipedia_id,
                                "decile":                   q.decile,
                                "decile_unweighted":        q.decile_unweighted,
                                "decile_chunk_weighted":    q.decile_chunk_weighted,
                                "popularity_avg":           q.popularity_avg,
                                "dataset":                  q.dataset,
                                "strategy":                 backend.key,
                                "context_size":             0 if backend.key == "zero_shot" else ctx_n,
                                "retrieved_doc_popularity": [doc.metadata.get("popularity", 0) for doc in docs],
                                "retrieved_doc_ids":        [doc.metadata.get("wikipedia_id", "") for doc in docs],
                            },
                        )
                        for i, (q, docs) in enumerate(zip(question_data, docs_sliced))
                    ]

                    for eval_backend in self.cfg.evaluators:
                        run_idx += 1
                        tag = f"{llm_model.key} | {backend.key} | {ctx_label} | {eval_backend.key}"

                        eval_checkpoint = (
                            self._output_folder
                            / f"eval_checkpoint_{llm_model.key}_{backend.key}_{ctx_label}_{eval_backend.key}.jsonl"
                        )
                        parquet_out = (
                            self._output_folder
                            / f"results_{llm_model.key}_{backend.key}_{ctx_label}_{eval_backend.key}.parquet"
                        )

                        if self.cfg.restart:
                            for p in (parquet_out, eval_checkpoint):
                                if p.exists():
                                    p.unlink()
                                    logger.info("  [%d/%d] %s — deleted %s", run_idx, n_total, tag, p.name)

                        # ── Diff: find which question ids still need evaluation ─
                        done_ids: set[str] = set()
                        existing_df: pd.DataFrame | None = None
                        if parquet_out.exists():
                            try:
                                existing_df = pd.read_parquet(parquet_out)
                                done_ids = set(existing_df["id"].astype(str))
                            except Exception:
                                existing_df = None

                        pending_objects = [obj for obj in evaluation_objects if str(obj.id) not in done_ids]

                        if not pending_objects:
                            logger.info("  [%d/%d] %s — all %d evals done, skipping", run_idx, n_total, tag, len(evaluation_objects))
                            continue

                        if done_ids:
                            logger.info(
                                "  [%d/%d] %s — %d / %d evals missing, resuming…",
                                run_idx, n_total, tag, len(pending_objects), len(evaluation_objects),
                            )

                        evaluator = self._build_evaluator(eval_backend)

                        logger.info(
                            "  [%d/%d] %s — evaluating %d answers…",
                            run_idx, n_total, tag, len(pending_objects),
                        )
                        new_results: list[EvaluationResult] = evaluator.evaluate(
                            pending_objects,
                            checkpoint_path=eval_checkpoint,
                        )
                        logger.info(
                            "  [%d/%d] %s — done, %d new results",
                            run_idx, n_total, tag, len(new_results),
                        )

                        # Merge new results with any existing rows, rewrite parquet
                        new_df = pd.DataFrame([r.__dict__ for r in new_results])
                        new_df["evaluator"] = eval_backend.key
                        if existing_df is not None and not existing_df.empty:
                            results_df = pd.concat([existing_df, new_df], ignore_index=True)
                        else:
                            results_df = new_df
                        results_df.to_parquet(parquet_out, index=False)
                        logger.info("  saved → %s  (%d total rows)", parquet_out.name, len(results_df))

    # ── Entry point ───────────────────────────────────────────────────────────

    @classmethod
    def main(cls, argv: list[str] | None = None) -> None:
        """Parse CLI arguments, instantiate the runner, and call :meth:`run`."""
        p = argparse.ArgumentParser(
            description="Stage 3 — evaluate generated answers and write final parquet files.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        _rd = RetrievalConfig()
        p.add_argument("--collection", "-c", default=_rd.collection_name)
        p.add_argument("--output-dir", "-o", default=_rd.output_dir)
        p.add_argument("--top-k", type=int, default=_rd.top_k)
        p.add_argument("--questions-per-decile", type=int, default=_rd.questions_per_decile)
        p.add_argument("--restart", action="store_true",
                       help="Delete existing eval checkpoints and re-evaluate.")
        args = p.parse_args(argv)

        cfg = EvalConfig(
            generating=GeneratingConfig(
                retrieval=RetrievalConfig(
                    collection_name=args.collection,
                    output_dir=args.output_dir,
                    top_k=args.top_k,
                    questions_per_decile=args.questions_per_decile,
                ),
            ),
            restart=args.restart,
        )
        cls(cfg).run()


if __name__ == "__main__":
    LLMEvalRunner.main()
