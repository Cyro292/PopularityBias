"""Generate supported thesis figures and synchronize canonical artifacts."""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
PAPER_DIR = ROOT_DIR / "paper_figures"
THESIS_DIR = ROOT_DIR / "data" / "thesis_latex_tum_updated"
THESIS_FIGURES = THESIS_DIR / "figures"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FigureArtifact:
    """Describe one thesis figure and its canonical repository source."""

    thesis_name: str
    source: Path
    status: str


ARTIFACTS = (
    FigureArtifact("mean_chunks_per_article_by_chunk_weighted_decile.png", PAPER_DIR / "mean_chunks_per_article_by_chunk_weighted_decile.png", "notebook"),
    FigureArtifact("mrr_bm25_by_dataset_and_decile_60k_balanced.png", PAPER_DIR / "mrr_bm25_by_dataset_and_decile_60k_balanced.png", "script"),
    FigureArtifact("mrr_faiss_high_by_dataset_and_decile_60k_balanced.png", PAPER_DIR / "mrr_faiss_high_by_dataset_and_decile_60k_balanced.png", "script"),
    FigureArtifact("delta_vs_bm25_retrieved_docs_ivfpq_high_mrr_60k_balanced.png", PAPER_DIR / "delta_vs_bm25_retrieved_docs_ivfpq_high_mrr_60k_balanced.png", "script"),
    FigureArtifact("bm25_candidate_recall_by_decile_60k_balanced.png", PAPER_DIR / "bm25_candidate_recall_by_decile_60k_balanced.png", "expensive"),
    FigureArtifact("bm25_lexical_competition_by_decile_60k_balanced.png", PAPER_DIR / "bm25_lexical_competition_by_decile_60k_balanced.png", "expensive"),
    FigureArtifact("similarity_score_distribution.png", PAPER_DIR / "analogue_similarity_score_distribution.png", "script-existing-scores"),
    FigureArtifact("pref-curve.png", PAPER_DIR / "wrong_retrieval_preference_equal_article_60k.png", "script"),
    FigureArtifact("qwen_generation_retrieval_accuracy_by_decile.png", ROOT_DIR / "notebooks" / "full_pipe_eval" / "images" / "qwen_generation_retrieval_accuracy_by_decile.png", "notebook-8k"),
    FigureArtifact("qwen_retrieval_hit_lift_by_decile.png", ROOT_DIR / "notebooks" / "full_pipe_eval" / "images" / "qwen_retrieval_hit_lift_by_decile.png", "notebook-8k"),
)


def _run(command: list[str]) -> None:
    logger.info("Running: %s", " ".join(command))
    subprocess.run(command, cwd=ROOT_DIR, check=True)


def generate_script_figures(*, include_expensive: bool) -> None:
    """Generate figures with maintained Python modules."""
    python = sys.executable
    common = [
        "--results-dir", "data/wiki_full_bil/all_qa_60k_balanced",
        "--qa-path", "data/wiki_full_bil/all_qa_60k_balanced/cyro_qa_cache.parquet",
        "--metric", "mrr",
        "--exclude-datasets", "hotpot_qa",
    ]
    _run([python, "-m", "src.process.analysis.plot_dataset_recall_by_decile", *common, "--panel", "bm25", "--output-path", "paper_figures/mrr_bm25_by_dataset_and_decile_60k_balanced.png"])
    _run([python, "-m", "src.process.analysis.plot_dataset_recall_by_decile", *common, "--panel", "faiss-high", "--output-path", "paper_figures/mrr_faiss_high_by_dataset_and_decile_60k_balanced.png"])
    _run([python, "-m", "src.process.analysis.plot_dataset_recall_by_decile", *common, "--figure", "delta", "--output-path", "paper_figures/delta_vs_bm25_retrieved_docs_ivfpq_high_mrr_60k_balanced.png"])
    _run([python, "-m", "src.process.analysis.plot_wrong_retrieval_preference_curve"])
    _run([python, "-m", "src.process.analysis.plot_analogue_similarity"])
    if include_expensive:
        _run([python, "-m", "src.process.analysis.plot_bm25_60k_diagnostics"])


def sync_figures(*, strict: bool) -> list[FigureArtifact]:
    """Copy canonical artifacts into the thesis and return missing entries."""
    THESIS_FIGURES.mkdir(parents=True, exist_ok=True)
    missing: list[FigureArtifact] = []
    for artifact in ARTIFACTS:
        if not artifact.source.exists():
            missing.append(artifact)
            logger.warning("Missing %s source: %s", artifact.status, artifact.source)
            continue
        destination = THESIS_FIGURES / artifact.thesis_name
        shutil.copy2(artifact.source, destination)
        logger.info("Synchronized %s", destination.relative_to(ROOT_DIR))
    if strict and missing:
        names = ", ".join(artifact.thesis_name for artifact in missing)
        raise FileNotFoundError(f"Missing thesis figure sources: {names}")
    return missing


def main() -> None:
    """Generate maintained outputs and synchronize thesis figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true", help="Run maintained non-expensive figure generators before copying.")
    parser.add_argument("--include-expensive", action="store_true", help="Also run live BM25 depth-100 diagnostics.")
    parser.add_argument("--strict", action="store_true", help="Fail if any canonical source is missing.")
    args = parser.parse_args()
    if args.include_expensive and not args.generate:
        parser.error("--include-expensive requires --generate")
    if args.generate:
        generate_script_figures(include_expensive=args.include_expensive)
    missing = sync_figures(strict=args.strict)
    logger.info("Synchronization complete (%d missing)", len(missing))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
