"""Build a flat analysis dataset from all pipeline results parquets.

Reads every ``results_<llm>_<backend>_<ctx>_<eval_type>.parquet`` from the
results directory and concatenates them into a single parquet with one row
per (question, run-configuration).

Edit :class:`BuildConfig` at the bottom of this file to control which
LLMs, backends, context labels, and evaluator types are included.

Output columns
--------------
question_id     : str   — row ``id`` from the results parquet
wikipedia_id    : str   — from metadata dict
question_text   : str   — the ``question`` column verbatim from the results parquet
answers         : list  — gold answer strings
proposed_answer : str   — LLM answer (may be raw generation pre-parse)
performance     : bool  — evaluation_score (True = correct)
popularity      : float — popularity_avg from metadata
decile          : int   — canonical decile (0–9) from metadata
llm             : str   — e.g. "neo", "qwen"
backend         : str   — e.g. "bm25_plus", "ivfpq_low", "ivfpq_high", "zero_shot"
ctx_size        : int   — context_size from metadata (0 for zero_shot)
ctx_label       : str   — "zero", "top1", "top3"
eval_type       : str   — "substring" or "binary_mistral"
run_type        : str   — compound label e.g. "ivfpq_low_top1"
dataset         : str   — source QA dataset name from metadata
retrieved_doc_ids      : list[int] | None
retrieved_doc_popularity : list[float] | None
source_file     : str   — originating parquet filename (for debugging)
"""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd

# ── project root on path ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)

# ── default paths ─────────────────────────────────────────────────────────────
_DEFAULT_RESULTS_DIR = DATA_DIR / "wiki_full_bil" / "all_qa_8k"
_DEFAULT_OUTPUT_PATH = DATA_DIR / "wiki_full_bil" / "analysis_dataset.parquet"

# ── filename regex ─────────────────────────────────────────────────────────────
# Pattern:  results_<llm>_<backend_and_ctx>_<eval_type>.parquet
# Examples:
#   results_neo_bm25_plus_top1_binary_mistral.parquet
#   results_qwen_ivfpq_low_top3_substring.parquet
#   results_neo_zero_shot_zero_binary_mistral.parquet
_FILE_RE = re.compile(
    r"^results_"
    r"(?P<llm>neo|qwen)_"
    r"(?P<backend_ctx>.+?)_"
    r"(?P<eval_type>binary_mistral|substring)"
    r"\.parquet$"
)
_CTX_TOKENS = {"top1", "top3", "zero"}


# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class BuildConfig:
    """Controls which results parquets are included in the analysis dataset.

    Set any list to a non-empty value to restrict to those values only.
    Leave as the default (all options listed) to include everything.

    Attributes:
        llms: LLM generator keys to include.  Known values: ``"neo"``, ``"qwen"``.
        backends: Retrieval backend keys to include.  Known values:
            ``"bm25_plus"``, ``"ivfpq_low"``, ``"ivfpq_high"``, ``"zero_shot"``.
        ctx_labels: Context-size labels to include.  Known values:
            ``"top1"``, ``"top3"``, ``"zero"``.
        eval_types: Evaluator types to include.  Known values:
            ``"substring"``, ``"binary_mistral"``.
        results_dir: Directory that contains ``results_*.parquet`` files.
        output_path: Destination parquet file.
    """

    llms:        list[Literal["neo", "qwen"]]                         = field(default_factory=lambda: ["neo", "qwen"])
    backends:    list[str]                                             = field(default_factory=lambda: ["bm25_plus", "ivfpq_low", "ivfpq_high", "zero_shot"])
    ctx_labels:  list[Literal["top1", "top3", "zero"]]                = field(default_factory=lambda: ["top1", "top3", "zero"])
    eval_types:  list[Literal["substring", "binary_mistral"]]         = field(default_factory=lambda: ["substring", "binary_mistral"])
    results_dir: Path                                                  = field(default_factory=lambda: _DEFAULT_RESULTS_DIR)
    output_path: Path                                                  = field(default_factory=lambda: _DEFAULT_OUTPUT_PATH)


# ═══════════════════════════════════════════════════════════════════════════════
# Internals
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_filename(name: str) -> dict[str, str] | None:
    """Return parsed fields from a results parquet filename, or None if unrecognised."""
    m = _FILE_RE.match(name)
    if not m:
        return None

    llm = m.group("llm")
    backend_ctx = m.group("backend_ctx")
    eval_type = m.group("eval_type")

    parts = backend_ctx.rsplit("_", 1)
    if len(parts) == 2 and parts[1] in _CTX_TOKENS:
        backend, ctx_label = parts[0], parts[1]
    else:
        backend, ctx_label = backend_ctx, "unknown"

    return {"llm": llm, "backend": backend, "ctx_label": ctx_label, "eval_type": eval_type}


def _matches_config(parsed: dict[str, str], cfg: BuildConfig) -> bool:
    """Return True if parsed filename fields pass all config filters."""
    return (
        parsed["llm"]       in cfg.llms
        and parsed["backend"]   in cfg.backends
        and parsed["ctx_label"] in cfg.ctx_labels
        and parsed["eval_type"] in cfg.eval_types
    )


def _extract_metadata_fields(meta: dict) -> dict:
    """Pull scalar fields out of the per-row metadata dict."""
    doc_ids = meta.get("retrieved_doc_ids")
    doc_pop = meta.get("retrieved_doc_popularity")

    return {
        "wikipedia_id": str(meta.get("wikipedia_id", "")),
        "popularity":   float(meta.get("popularity_avg", float("nan"))),
        "decile":       int(meta.get("decile", meta.get("decile_chunk_weighted", -1))),
        "ctx_size":     int(meta.get("context_size", 0)),
        "dataset":      str(meta.get("dataset", "")),
        "retrieved_doc_ids":       list(doc_ids) if doc_ids is not None else None,
        "retrieved_doc_popularity": list(doc_pop) if doc_pop is not None else None,
    }


def _process_file(path: Path, parsed: dict[str, str]) -> pd.DataFrame:
    """Load one results parquet and return a normalised DataFrame."""
    df = pd.read_parquet(path)

    meta_df = pd.DataFrame(list(df["metadata"].apply(_extract_metadata_fields)))

    out = pd.DataFrame()
    out["question_id"]             = df["id"].astype(str)
    out["wikipedia_id"]            = meta_df["wikipedia_id"]
    out["question_text"]           = df["question"].astype(str)
    out["answers"]                 = df["answers"]
    out["proposed_answer"]         = df["proposed_answer"].astype(str)
    out["performance"]             = df["evaluation_score"].astype(bool)
    out["popularity"]              = meta_df["popularity"]
    out["decile"]                  = meta_df["decile"]
    out["llm"]                     = parsed["llm"]
    out["backend"]                 = parsed["backend"]
    out["ctx_label"]               = parsed["ctx_label"]
    out["ctx_size"]                = meta_df["ctx_size"]
    out["eval_type"]               = parsed["eval_type"]
    out["run_type"]                = parsed["backend"] + "_" + parsed["ctx_label"]
    out["dataset"]                 = meta_df["dataset"]
    out["retrieved_doc_ids"]       = meta_df["retrieved_doc_ids"]
    out["retrieved_doc_popularity"] = meta_df["retrieved_doc_popularity"]
    out["source_file"]             = path.name

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Public build function
# ═══════════════════════════════════════════════════════════════════════════════

def build(cfg: BuildConfig | None = None) -> pd.DataFrame:
    """Scan ``cfg.results_dir``, filter by config, and write the flat dataset.

    Args:
        cfg: :class:`BuildConfig` controlling which parquets to include.
            Defaults to :class:`BuildConfig` defaults (all parquets).

    Returns:
        The combined DataFrame that was written to ``cfg.output_path``.
    """
    if cfg is None:
        cfg = BuildConfig()

    all_files = sorted(cfg.results_dir.glob("results_*.parquet"))
    matched:  list[tuple[Path, dict]] = []
    filtered: list[str] = []
    skipped:  list[str] = []

    for f in all_files:
        parsed = _parse_filename(f.name)
        if parsed is None:
            skipped.append(f.name)
        elif _matches_config(parsed, cfg):
            matched.append((f, parsed))
        else:
            filtered.append(f.name)

    if skipped:
        logger.warning("Skipped %d unrecognised files: %s", len(skipped), skipped)
    if filtered:
        logger.info("Filtered out %d files (not in config): %s", len(filtered), filtered)

    if not matched:
        raise RuntimeError(
            f"No matching result parquets found in {cfg.results_dir} "
            f"for config llms={cfg.llms} backends={cfg.backends} "
            f"ctx_labels={cfg.ctx_labels} eval_types={cfg.eval_types}"
        )

    logger.info("Processing %d result parquets …", len(matched))

    frames: list[pd.DataFrame] = []
    for i, (path, parsed) in enumerate(matched, 1):
        logger.info(
            "[%d/%d] %s  llm=%s backend=%s ctx=%s eval=%s",
            i, len(matched), path.name,
            parsed["llm"], parsed["backend"], parsed["ctx_label"], parsed["eval_type"],
        )
        frames.append(_process_file(path, parsed))

    combined = pd.concat(frames, ignore_index=True)

    logger.info(
        "Combined dataset: %d rows  %d columns  %d unique run_types",
        len(combined), len(combined.columns), combined["run_type"].nunique(),
    )
    logger.info("Run types: %s", sorted(combined["run_type"].unique()))

    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(cfg.output_path, index=False)
    logger.info("Wrote → %s", cfg.output_path)

    return combined


# ═══════════════════════════════════════════════════════════════════════════════
# !! Edit BuildConfig here to control what gets included in the output !!
# ═══════════════════════════════════════════════════════════════════════════════

ACTIVE_CONFIG = BuildConfig(
    llms        = ["neo", "qwen"],
    backends    = ["bm25_plus", "ivfpq_low", "ivfpq_high", "zero_shot"],
    ctx_labels  = ["top1", "top3", "zero"],
    eval_types  = ["substring", "binary_mistral"],
    # results_dir = DATA_DIR / "wiki_full_bil" / "all_qa_8k",   # override if needed
    # output_path = DATA_DIR / "wiki_full_bil" / "analysis_dataset.parquet",
)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="Override results directory from ACTIVE_CONFIG")
    parser.add_argument("--output", type=Path, default=None,
                        help="Override output parquet path from ACTIVE_CONFIG")
    parser.add_argument("--llms", nargs="+", default=None,
                        metavar="LLM", help="Override llms filter (e.g. --llms neo)")
    parser.add_argument("--backends", nargs="+", default=None,
                        metavar="BACKEND", help="Override backends filter (e.g. --backends bm25_plus ivfpq_low)")
    parser.add_argument("--ctx-labels", nargs="+", default=None,
                        metavar="CTX", help="Override ctx_labels filter (e.g. --ctx-labels top1 top3)")
    parser.add_argument("--eval-types", nargs="+", default=None,
                        metavar="EVAL", help="Override eval_types filter (e.g. --eval-types substring)")
    args = parser.parse_args()

    # Start from ACTIVE_CONFIG, apply any CLI overrides
    cfg = BuildConfig(
        llms        = args.llms        or ACTIVE_CONFIG.llms,
        backends    = args.backends    or ACTIVE_CONFIG.backends,
        ctx_labels  = args.ctx_labels  or ACTIVE_CONFIG.ctx_labels,
        eval_types  = args.eval_types  or ACTIVE_CONFIG.eval_types,
        results_dir = args.results_dir or ACTIVE_CONFIG.results_dir,
        output_path = args.output      or ACTIVE_CONFIG.output_path,
    )

    build(cfg)
