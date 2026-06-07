"""Generating runner — Stage 2 of the RAG evaluation pipeline.

Picks up ``retrieved_docs_<key>.csv`` checkpoints written by
:mod:`src.process.pipeline.retrieval_runner`, generates an answer for each
question using the configured LLM, and saves ``answer_checkpoint_<key>.csv``.

The next stage (:mod:`src.process.pipeline.llm_eval_runner`) picks those up.

Checkpoint behaviour
--------------------
- ``answer_checkpoint_<key>.csv`` is reused if it exists.
- Pass ``--restart`` to delete existing answer checkpoints and regenerate.
- Retrieval checkpoints are read-only input; never deleted by this stage.
- If a retrieval checkpoint is missing, retrieval is run automatically.

Usage
-----
::

    python -m src.process.pipeline.generating_runner
    python -m src.process.pipeline.generating_runner --restart
    python -m src.process.pipeline.generating_runner --help
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
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

from typing import Any, Literal

from config import DATA_DIR
from src.corpus_handler.parquet_corpus_handler import ParquetCorpusHandler
from src.question_input.huggingface_cyro_input import HuggingFaceCyroInput
from src.llm.base import LLMBase
from src.process.pipeline.retrieval_runner import (
    RetrievalConfig,
    RetrievalRunner,
    load_retrieved_docs_csv,
    save_retrieved_docs_csv,
)
from src.process.pipeline.latency_utils import time_batch, save_latency


# ═══════════════════════════════════════════════════════════════════════════════
# Settings
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LLMBackend:
    """Selects and configures an LLM for the generation stage.

    Attributes:
        key: Short identifier used in checkpoint filenames (e.g. ``"neo"``,
            ``"mistral"``).  Must be unique across the ``models`` list.
        type: Which service class to instantiate.  One of ``"neo"``,
            ``"mistral"``, ``"modal"``, ``"openai"``, ``"qwen"``.
        model_name: Model identifier forwarded to the service (required for
            ``"modal"`` and ``"openai"``; ignored by the others).
        service_kwargs: Extra keyword arguments passed verbatim to the service
            constructor (e.g. ``{"max_new_tokens": 256}``).
    """

    key:               str
    type:              Literal["neo", "mistral", "modal", "openai", "qwen"] = "neo"
    model_name:        str | None                                            = None
    request_batch_size: int                                                  = 128
    gpu_batch_size:    int                                                   = 32
    service_kwargs:    dict[str, Any]                                        = field(default_factory=dict)


@dataclass
class GenerationBackend:
    """Selects a single retrieval backend for the generation stage.

    Attributes:
        key: Must match a :attr:`RetrievalBackend.key` in the retrieval config.
    """

    key: str


@dataclass(frozen=True)
class GeneratingConfig:
    """Configuration for the answer-generation stage.

    Attributes:
        retrieval: Full retrieval configuration (backends, paths, top_k…).
        models: LLM backends to evaluate.  Each model is run against every
            active retrieval backend, producing separate checkpoint files
            named ``answer_checkpoint_{llm_key}_{retrieval_key}_top{n}.csv``.
        backends: Subset of retrieval backends to generate answers for.
            Each entry's ``key`` must correspond to a backend defined in
            ``retrieval.backends``.  Defaults to all retrieval backends.
        context_sizes: How many retrieved documents to feed into the prompt.
            Each value produces a separate set of checkpoints.  For example
            ``[1, 3, 5]`` runs every (model × backend) combination three
            times, once with the top-1, top-3, and top-5 retrieved docs.
            All values must be ≤ ``retrieval.top_k``.
        prompt_template: Python format string used to build each LLM prompt.
            Available placeholders:

            - ``{question}``  — the question text
            - ``{documents}`` — retrieved document contents joined by the
              ``document_separator`` (empty string for zero-shot)
            - ``{dataset}``   — source dataset name (e.g. ``"natural_questions"``)

        document_separator: String used to join individual document texts
            inside ``{documents}``.  Defaults to a blank line between docs.
        restart: If True, delete existing answer checkpoints and regenerate.
    """

    retrieval:          RetrievalConfig         = None  # type: ignore[assignment]
    models:             list[LLMBackend]        = field(default_factory=lambda: [
        LLMBackend(key="neo",     type="neo"),
        LLMBackend(key="qwen",    type="qwen"),
    ])
    backends:           list[GenerationBackend] = field(default_factory=lambda: [
        GenerationBackend(key="zero_shot"),
        GenerationBackend(key="bm25_plus"),
        GenerationBackend(key="ivfpq_high"),
        GenerationBackend(key="ivfpq_extremely_high"),
        # GenerationBackend(key="es_approx"),
        # GenerationBackend(key="es_hybrid"),
        # GenerationBackend(key="router"),
        # GenerationBackend(key="router_es"),
        # GenerationBackend(key="faiss_hybrid"),
    ])
    context_sizes:      list[int]               = field(default_factory=lambda: [3])
    prompt_template:    str                     = (
        "Documents: {documents}\n \n \n Question: {question}"
    )
    document_separator: str                     = "\n"
    restart:            bool                    = False

    def __post_init__(self) -> None:
        if self.retrieval is None:
            object.__setattr__(self, "retrieval", RetrievalConfig())
        invalid = [n for n in self.context_sizes if n < 1]
        if invalid:
            raise ValueError(f"context_sizes must all be ≥ 1, got: {invalid}")
        try:
            self.prompt_template.format_map(
                {"question": "", "documents": "", "dataset": ""}
            )
        except KeyError as exc:
            raise ValueError(
                f"prompt_template contains unknown placeholder {exc}. "
                f"Allowed: {{question}}, {{documents}}, {{dataset}}"
            ) from exc


# ═══════════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════════

class GeneratingRunner:
    """Stage 2 runner — generates answers from retrieved documents.

    Args:
        cfg: Active :class:`GeneratingConfig`.
    """

    def __init__(self, cfg: GeneratingConfig) -> None:
        self.cfg = cfg
        self._rcfg             = cfg.retrieval
        self._collection_folder = DATA_DIR / self._rcfg.collection_name
        self._output_folder     = self._collection_folder / self._rcfg.output_dir

    # ── LLM factory ──────────────────────────────────────────────────────────

    @staticmethod
    def _build_llm_service(llm: LLMBackend) -> LLMBase:
        """Instantiate the appropriate LLM service for *llm*.

        Args:
            llm: :class:`LLMBackend` descriptor from the config.

        Returns:
            A ready-to-use :class:`~src.llm.base.LLMBase` instance.

        Raises:
            ValueError: If ``llm.type`` is not a recognised service type.
        """
        kwargs = {"request_batch_size": llm.request_batch_size, "gpu_batch_size": llm.gpu_batch_size, **llm.service_kwargs}
        if llm.type == "neo":
            from src.llm.gptNeo27bLLMService import GPTNeo27bLLMService
            return GPTNeo27bLLMService(**kwargs)
        elif llm.type == "mistral":
            from src.llm.mistralLLMService import MistralLLMService
            return MistralLLMService(**kwargs)
        elif llm.type == "modal":
            from src.llm.modalLLMService import ModalLLMService
            model_name = llm.model_name or ""
            return ModalLLMService(model_name=model_name, **kwargs)
        elif llm.type == "openai":
            from src.llm.openAi_service import OpenAIService
            model_name = llm.model_name or "gpt-4o-mini"
            return OpenAIService(model_name=model_name, **kwargs)
        elif llm.type == "qwen":
            from src.llm.qwenLLMService import QwenLLMService
            return QwenLLMService(**kwargs)
        else:
            raise ValueError(f"Unknown LLM type: {llm.type!r}")

    def run(self) -> None:
        """Generate answers for all configured LLM × retrieval backend × context size triples.

        For each (model, retrieval backend, context size) combination:

        1. Skip if answer checkpoint already exists (unless ``cfg.restart``).
        2. Load retrieval checkpoint; run Stage 1 automatically if missing.
        3. Slice retrieved docs to ``context_size`` and build prompts.
        4. Generate answers via the LLM.
        5. Save ``answer_checkpoint_{llm_key}_{retrieval_key}_top{n}.csv``.
        """
        self._output_folder.mkdir(parents=True, exist_ok=True)

        active_keys = {b.key for b in self.cfg.backends}
        active_backends = [b for b in self._rcfg.backends if b.key in active_keys]
        unknown = active_keys - {b.key for b in self._rcfg.backends}
        if unknown:
            raise ValueError(
                f"GeneratingConfig.backends contains keys not found in retrieval config: {unknown}"
            )

        oversized = [n for n in self.cfg.context_sizes if n > self._rcfg.top_k]
        if oversized:
            raise ValueError(
                f"context_sizes {oversized} exceed retrieval top_k={self._rcfg.top_k}"
            )

        corpus_handler = ParquetCorpusHandler(
            corpus_path=self._collection_folder / "wiki_corpus.parquet",
            metadata_path=self._collection_folder / "metadata.json",
        )
        # Disable balancing when questions_per_decile is -1 (use all questions as-is)
        use_all_questions = self._rcfg.questions_per_decile == -1
        question_input = HuggingFaceCyroInput(
            dataset_names=list(self._rcfg.dataset_names),
            corpus_handler=corpus_handler,
            parquet_path=self._output_folder / "cyro_qa_cache.parquet",
            balance_deciles=not use_all_questions,
            balance_datasets=not use_all_questions,
            target_per_decile=None if use_all_questions else self._rcfg.questions_per_decile,
            shuffle=True,
            balance_decile_mode=self._rcfg.balance_decile_mode,
        )
        question_input.load()
        question_data = question_input.get_items()
        logger.info("Loaded %d questions", len(question_data))

        questions    = [item.question_text for item in question_data]
        question_ids = [item.question_id   for item in question_data]

        n_models    = len(self.cfg.models)
        n_backends  = len(active_backends)
        n_zero_shot = sum(1 for b in active_backends if b.key == "zero_shot")
        n_retrieval = n_backends - n_zero_shot
        n_sizes     = len(self.cfg.context_sizes)
        n_total     = n_models * (n_zero_shot + n_retrieval * n_sizes)
        logger.info(
            "Starting generation: %d model(s) × (%d zero-shot + %d backend(s) × %d context size(s)) = %d runs",
            n_models, n_zero_shot, n_retrieval, n_sizes, n_total,
        )

        retrieval_runner = RetrievalRunner(self._rcfg)
        run_idx = 0

        for llm_model in self.cfg.models:
            logger.info("── Model: %s (%s) ──", llm_model.key, llm_model.type)
            llm_service = self._build_llm_service(llm_model)

            for backend in active_backends:
                retrieval_checkpoint = self._output_folder / f"retrieved_docs_{backend.key}.csv"

                # ── Load or produce retrieval docs (shared across context sizes) ──
                retrieved_docs = load_retrieved_docs_csv(retrieval_checkpoint, question_ids)
                if retrieved_docs is None:
                    logger.info("  [%s] no retrieval checkpoint — running Stage 1", backend.key)
                    retrieved_docs = retrieval_runner.retrieve_for_backend(backend, questions)
                    save_retrieved_docs_csv(retrieved_docs, question_ids, retrieval_checkpoint)

                if len(retrieved_docs) != len(questions):
                    logger.warning(
                        "  [%s] checkpoint length mismatch (%d vs %d) — truncating",
                        backend.key, len(retrieved_docs), len(questions),
                    )
                    retrieved_docs = retrieved_docs[: len(questions)]

                for ctx_n in ([self.cfg.context_sizes[0]] if backend.key == "zero_shot" else self.cfg.context_sizes):
                    run_idx += 1
                    tag = f"{llm_model.key} | {backend.key} | top{ctx_n}"
                    answer_checkpoint = (
                        self._output_folder
                        / f"answer_checkpoint_{llm_model.key}_{backend.key}_top{ctx_n}.csv"
                    )

                    if self.cfg.restart and answer_checkpoint.exists():
                        logger.info("  [%d/%d] %s — restarting, deleting checkpoint", run_idx, n_total, tag)
                        answer_checkpoint.unlink()

                    # ── Diff: find which question_ids still need answers ───────
                    done_map: dict[str, str] = {}
                    if answer_checkpoint.exists():
                        try:
                            import pandas as _pd
                            _df = _pd.read_csv(answer_checkpoint)
                            if "question_id" in _df.columns and "answer" in _df.columns:
                                done_map = dict(zip(_df["question_id"].astype(str), _df["answer"].fillna("")))
                        except Exception:
                            pass

                    missing_mask = [qid not in done_map for qid in question_ids]
                    missing_count = sum(missing_mask)

                    if missing_count == 0:
                        logger.info("  [%d/%d] %s — all %d answers done, skipping", run_idx, n_total, tag, len(question_ids))
                        continue

                    if done_map:
                        logger.info(
                            "  [%d/%d] %s — %d / %d answers missing, resuming…",
                            run_idx, n_total, tag, missing_count, len(question_ids),
                        )

                    # ── Build prompts only for missing questions ───────────────
                    pending_data     = [(q, docs, qid) for (q, docs, qid, m) in zip(question_data, retrieved_docs, question_ids, missing_mask) if m]
                    pending_prompts  = [
                        self.cfg.prompt_template.format_map({
                            "question":  q.question_text,
                            "documents": self.cfg.document_separator.join(
                                doc.page_content for doc in docs[:ctx_n]
                            ),
                            "dataset":   q.dataset,
                        })
                        for q, docs, _ in pending_data
                    ]
                    pending_ids = [qid for _, _, qid in pending_data]

                    logger.info(
                        "  [%d/%d] %s — generating %d answers…",
                        run_idx, n_total, tag, len(pending_prompts),
                    )
                    new_answers, gen_latencies_ms = time_batch(
                        lambda: llm_service.batch_generate(
                            pending_prompts,
                            checkpoint_path=answer_checkpoint,
                            question_ids=pending_ids,
                        ),
                        pending_ids,
                    )

                    # Merge new answers into done_map, then write full checkpoint
                    for qid, ans in zip(pending_ids, new_answers):
                        done_map[str(qid)] = ans
                    answers = [done_map.get(str(qid), "") for qid in question_ids]

                    import pandas as _pd
                    _pd.DataFrame({"question_id": question_ids, "answer": answers}).to_csv(answer_checkpoint, index=False)

                    logger.info(
                        "  [%d/%d] %s — done, %d answers saved to %s",
                        run_idx, n_total, tag, len(answers), answer_checkpoint.name,
                    )

                    latency_path = (
                        self._output_folder
                        / f"latency_generation_{llm_model.key}_{backend.key}_top{ctx_n}.json"
                    )
                    save_latency(
                        path=latency_path,
                        backend_key=f"{llm_model.key}_{backend.key}_top{ctx_n}",
                        stage="generation",
                        question_ids=question_ids,
                        latencies_ms=gen_latencies_ms,
                    )

    # ── Entry point ───────────────────────────────────────────────────────────

    @classmethod
    def main(cls, argv: list[str] | None = None) -> None:
        """Parse CLI arguments, instantiate the runner, and call :meth:`run`."""
        p = argparse.ArgumentParser(
            description="Stage 2 — generate answers from retrieved docs and save CSV checkpoints.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        _rd = RetrievalConfig()
        p.add_argument("--collection", "-c", default=_rd.collection_name)
        p.add_argument("--output-dir", "-o", default=_rd.output_dir)
        p.add_argument("--top-k", type=int, default=_rd.top_k)
        p.add_argument("--questions-per-decile", type=int, default=_rd.questions_per_decile)
        p.add_argument("--restart", action="store_true",
                       help="Delete existing answer checkpoints and regenerate.")
        args = p.parse_args(argv)

        cfg = GeneratingConfig(
            retrieval=RetrievalConfig(
                collection_name=args.collection,
                output_dir=args.output_dir,
                top_k=args.top_k,
                questions_per_decile=args.questions_per_decile,
            ),
            restart=args.restart,
        )
        cls(cfg).run()


if __name__ == "__main__":
    GeneratingRunner.main()
