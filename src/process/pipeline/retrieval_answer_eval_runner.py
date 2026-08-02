"""Evaluate retrieved chunks by answer containment rather than document identity.

This runner consumes ``retrieved_docs_<backend>.csv`` checkpoints produced by
the retrieval stage and writes one ``retrieval_answer_eval_<backend>.parquet``
file per backend. A retrieval is successful when any non-empty gold answer
alias occurs in a retrieved chunk, regardless of the chunk's Wikipedia ID.

Usage::

    python -m src.process.pipeline.retrieval_answer_eval_runner
    python -m src.process.pipeline.retrieval_answer_eval_runner \
        --output-dir all_qa_60k_balanced --only-keys bm25_plus ivfpq_high
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from config import DATA_DIR

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalAnswerEvalConfig:
    """Configuration for answer-containment retrieval evaluation.

    Args:
        collection_name: Corpus folder under ``DATA_DIR``.
        output_dir: Folder containing retrieval CSV checkpoints.
        backend_keys: Backends to evaluate. An empty tuple discovers every
            ``retrieved_docs_*.csv`` checkpoint in the output folder.
        qa_path: QA parquet containing ``question_id`` and ``answer_texts``.
            Defaults to ``cyro_qa_cache.parquet`` beside the checkpoints, with
            ``<collection>/all_qa_8k.parquet`` as a fallback.
        k_values: Retrieval cutoffs for answer Recall@k.
        case_sensitive: Whether answer containment is case-sensitive.
        restart: Recompute outputs even when they are newer than their inputs.
    """

    collection_name: str = "wiki_full_bil"
    output_dir: str = "all_qa_8k"
    backend_keys: tuple[str, ...] = ()
    qa_path: Path | None = None
    k_values: tuple[int, ...] = (1, 3, 5, 10)
    case_sensitive: bool = False
    restart: bool = False

    def __post_init__(self) -> None:
        if not self.k_values or any(k <= 0 for k in self.k_values):
            raise ValueError("k_values must contain positive integers")


def parse_answer_texts(value: object) -> list[str]:
    """Convert a parquet or serialized answer-alias value to clean strings.

    Args:
        value: A sequence of aliases, a serialized sequence, or one answer.

    Returns:
        Non-empty answer aliases in source order with duplicates removed.
    """
    if value is None or value is pd.NA or (
        isinstance(value, (float, np.floating)) and np.isnan(value)
    ):
        return []

    parsed = value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped[0] in "[({":
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(stripped)
                    break
                except (ValueError, SyntaxError, json.JSONDecodeError):
                    continue
        else:
            parsed = [stripped]

    if isinstance(parsed, np.ndarray):
        parsed = parsed.tolist()
    if not isinstance(parsed, (list, tuple, set)):
        parsed = [parsed]

    answers: list[str] = []
    seen: set[str] = set()
    for answer in parsed:
        if answer is None:
            continue
        text = str(answer).strip()
        if not text or text in seen:
            continue
        answers.append(text)
        seen.add(text)
    return answers


def _normalise_text(value: object, *, case_sensitive: bool) -> str:
    text = unicodedata.normalize("NFKC", "" if value is None else str(value))
    text = re.sub(r"\s+", " ", text)
    return text if case_sensitive else text.casefold()


def find_answer_rank(
    chunks: Sequence[object],
    answer_texts: Sequence[str],
    *,
    case_sensitive: bool = False,
) -> int | None:
    """Return the 1-based rank of the first chunk containing a gold answer.

    Args:
        chunks: Retrieved chunk texts in rank order.
        answer_texts: Acceptable gold answer aliases.
        case_sensitive: Whether matching preserves letter case.

    Returns:
        First matching chunk rank, or ``None`` if no alias occurs.
    """
    needles = [
        _normalise_text(answer, case_sensitive=case_sensitive)
        for answer in answer_texts
        if str(answer).strip()
    ]
    if not needles:
        return None

    for rank, chunk in enumerate(chunks, start=1):
        haystack = _normalise_text(chunk, case_sensitive=case_sensitive)
        if any(needle in haystack for needle in needles):
            return rank
    return None


def evaluate_retrieval_answers(
    retrieved: pd.DataFrame,
    questions: pd.DataFrame,
    *,
    k_values: Sequence[int],
    case_sensitive: bool = False,
) -> pd.DataFrame:
    """Build per-question answer-containment retrieval metrics.

    Questions without a gold answer alias are retained with ``is_evaluable``
    set to false and null metrics, so they do not count as retrieval failures.

    Args:
        retrieved: Retrieval checkpoint rows with chunk text and rank.
        questions: QA rows containing answer aliases and analysis metadata.
        k_values: Positive retrieval cutoffs to evaluate.
        case_sensitive: Whether answer matching is case-sensitive.

    Returns:
        One row per QA question with answer rank, reciprocal rank, and Recall@k.

    Raises:
        KeyError: If required input columns are absent.
    """
    retrieved_required = {"question_id", "doc_rank", "page_content"}
    question_required = {"question_id", "answer_texts"}
    missing_retrieved = retrieved_required - set(retrieved.columns)
    missing_questions = question_required - set(questions.columns)
    if missing_retrieved:
        raise KeyError(f"Retrieval checkpoint missing columns: {sorted(missing_retrieved)}")
    if missing_questions:
        raise KeyError(f"QA metadata missing columns: {sorted(missing_questions)}")

    cutoffs = sorted(set(k_values))
    if not cutoffs or any(k <= 0 for k in cutoffs):
        raise ValueError("k_values must contain positive integers")

    raw = retrieved.loc[:, ["question_id", "doc_rank", "page_content"]].copy()
    raw["question_id"] = raw["question_id"].astype(str)
    raw["doc_rank"] = pd.to_numeric(raw["doc_rank"], errors="coerce")
    raw = raw.dropna(subset=["doc_rank"])
    raw = raw[raw["doc_rank"] < max(cutoffs)].sort_values(["question_id", "doc_rank"])
    chunks_by_id = (
        raw.groupby("question_id", sort=False)["page_content"].apply(list).to_dict()
    )

    qa = questions.copy()
    qa["question_id"] = qa["question_id"].astype(str)
    qa = qa.drop_duplicates(subset="question_id", keep="first")

    rows: list[dict[str, object]] = []
    metadata_columns = [
        "question_text",
        "wikipedia_id",
        "wikipedia_title",
        "popularity_avg",
        "dataset",
        "decile",
        "decile_unweighted",
        "decile_chunk_weighted",
        "pop_decile_unweighted",
        "pop_decile_chunk_weighted",
    ]
    for question in qa.to_dict(orient="records"):
        question_id = question["question_id"]
        answers = parse_answer_texts(question.get("answer_texts"))
        chunks = chunks_by_id.get(question_id, [])
        answer_rank = find_answer_rank(
            chunks,
            answers,
            case_sensitive=case_sensitive,
        )
        is_evaluable = bool(answers)
        row: dict[str, object] = {
            "question_id": question_id,
            "answer_texts": answers,
            "is_evaluable": is_evaluable,
            "n_retrieved_chunks": len(chunks),
            "answer_rank": answer_rank,
            "answer_reciprocal_rank": (
                1.0 / answer_rank if is_evaluable and answer_rank is not None else 0.0
            ) if is_evaluable else np.nan,
        }
        for column in metadata_columns:
            if column in qa.columns:
                row[column] = question.get(column)
        for k in cutoffs:
            row[f"answer_recall@{k}"] = (
                float(answer_rank is not None and answer_rank <= k)
                if is_evaluable
                else np.nan
            )
        rows.append(row)

    return pd.DataFrame(rows)


class RetrievalAnswerEvalRunner:
    """Evaluate existing retrieval checkpoints using answer containment."""

    def __init__(self, cfg: RetrievalAnswerEvalConfig) -> None:
        self.cfg = cfg
        self._collection_folder = DATA_DIR / cfg.collection_name
        self._output_folder = self._collection_folder / cfg.output_dir

    def _resolve_qa_path(self) -> Path:
        if self.cfg.qa_path is not None:
            return Path(self.cfg.qa_path)
        cache_path = self._output_folder / "cyro_qa_cache.parquet"
        if cache_path.exists():
            return cache_path
        return self._collection_folder / "all_qa_8k.parquet"

    def _backend_keys(self) -> list[str]:
        if self.cfg.backend_keys:
            return list(self.cfg.backend_keys)
        prefix = "retrieved_docs_"
        return [
            path.stem[len(prefix):]
            for path in sorted(self._output_folder.glob(f"{prefix}*.csv"))
        ]

    def run(self) -> dict[str, pd.DataFrame]:
        """Evaluate configured backends and write parquet and summary outputs.

        Returns:
            Mapping from backend key to its per-question result DataFrame.

        Raises:
            FileNotFoundError: If QA metadata or retrieval checkpoints are absent.
        """
        qa_path = self._resolve_qa_path()
        if not qa_path.exists():
            raise FileNotFoundError(f"QA metadata not found: {qa_path}")

        backend_keys = self._backend_keys()
        if not backend_keys:
            raise FileNotFoundError(
                f"No retrieved_docs_*.csv checkpoints in {self._output_folder}"
            )

        questions = pd.read_parquet(qa_path)
        outputs: dict[str, pd.DataFrame] = {}
        summary_rows: list[dict[str, object]] = []
        for backend_key in backend_keys:
            source_path = self._output_folder / f"retrieved_docs_{backend_key}.csv"
            output_path = self._output_folder / f"retrieval_answer_eval_{backend_key}.parquet"
            if not source_path.exists():
                raise FileNotFoundError(f"Retrieval checkpoint not found: {source_path}")

            inputs_mtime = max(source_path.stat().st_mtime, qa_path.stat().st_mtime)
            can_reuse = (
                output_path.exists()
                and not self.cfg.restart
                and output_path.stat().st_mtime >= inputs_mtime
            )
            if can_reuse:
                logger.info("[%s] Reusing %s", backend_key, output_path.name)
                result = pd.read_parquet(output_path)
            else:
                logger.info("[%s] Evaluating answer containment", backend_key)
                retrieved = pd.read_csv(source_path, dtype={"question_id": str})
                result = evaluate_retrieval_answers(
                    retrieved,
                    questions,
                    k_values=self.cfg.k_values,
                    case_sensitive=self.cfg.case_sensitive,
                )
                result.to_parquet(output_path, index=False)
                logger.info(
                    "[%s] Saved %d question rows to %s",
                    backend_key,
                    len(result),
                    output_path,
                )

            outputs[backend_key] = result
            evaluable = result[result["is_evaluable"]]
            summary: dict[str, object] = {
                "backend": backend_key,
                "n_questions": len(result),
                "n_evaluable": len(evaluable),
                "answer_mrr": evaluable["answer_reciprocal_rank"].mean(),
            }
            for k in sorted(set(self.cfg.k_values)):
                summary[f"answer_recall@{k}"] = evaluable[
                    f"answer_recall@{k}"
                ].mean()
            summary_rows.append(summary)

        summary_df = pd.DataFrame(summary_rows)
        summary_path = self._output_folder / "retrieval_answer_eval_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info("Saved answer-retrieval summary to %s", summary_path)
        return outputs

    @classmethod
    def main(cls, argv: list[str] | None = None) -> None:
        """Parse command-line arguments and run answer retrieval evaluation."""
        defaults = RetrievalAnswerEvalConfig()
        parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument("--collection", "-c", default=defaults.collection_name)
        parser.add_argument("--output-dir", "-o", default=defaults.output_dir)
        parser.add_argument("--only-keys", nargs="+", default=[])
        parser.add_argument("--qa-path", type=Path, default=None)
        parser.add_argument(
            "--k-values",
            nargs="+",
            type=int,
            default=list(defaults.k_values),
        )
        parser.add_argument("--case-sensitive", action="store_true")
        parser.add_argument("--restart", action="store_true")
        args = parser.parse_args(argv)

        cfg = RetrievalAnswerEvalConfig(
            collection_name=args.collection,
            output_dir=args.output_dir,
            backend_keys=tuple(args.only_keys),
            qa_path=args.qa_path,
            k_values=tuple(args.k_values),
            case_sensitive=args.case_sensitive,
            restart=args.restart,
        )
        cls(cfg).run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    RetrievalAnswerEvalRunner.main()
