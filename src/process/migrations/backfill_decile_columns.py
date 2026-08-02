"""backfill_decile_columns.py — Add pop_decile_unweighted / pop_decile_chunk_weighted
to legacy results parquets and cyro_qa_cache parquets that were written before both
columns were introduced.

Usage:
    python -m src.process.migrations.backfill_decile_columns [--dry-run]

Strategy
--------
For every results_*.parquet and cyro_qa_cache.parquet under data/ that is missing
either decile column:

1. Locate the nearest ancestor directory that contains a metadata.json with both
   boundary arrays.
2. Load boundaries via ``load_boundaries_from_metadata``.
3. Compute both decile columns from the ``popularity_avg`` column (already present
   in all legacy files).
4. Write the enriched parquet back in-place (atomic rename via a .tmp file).
5. Invalidate any enriched_*.parquet caches in sibling _cache/ directories so that
   shared_setup.py recomputes them on next run.

The legacy ``decile`` column is left unchanged (it was unweighted in old files, or
absent in very old ones — we do not modify it).

CSV files (evaluation_results_*.csv) carry no popularity/decile data at all — they
are joined downstream in the pipeline notebook. No action is taken on them here.
"""
from __future__ import annotations

import argparse
import gc
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_DIR, ROOT_DIR
from src.metrics.decile_utils import (
    assign_decile,
    load_boundaries_from_metadata,
    COL_DECILE_UNWEIGHTED,
    COL_DECILE_CHUNK_WEIGHTED,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_metadata(start: Path) -> Path | None:
    """Walk upward from *start* until a metadata.json with boundaries is found."""
    for parent in [start, *start.parents]:
        candidate = parent / "metadata.json"
        if candidate.exists():
            import json
            with open(candidate) as f:
                d = json.load(f)
            if "decile_boundaries_unweighted" in d and "decile_boundaries_chunk_weighted" in d:
                return candidate
        if parent == DATA_DIR:
            break
    return None


def _invalidate_cache(parquet_path: Path) -> None:
    """Remove enriched_*.parquet from a sibling _cache/ dir, if present."""
    cache_dir = parquet_path.parent / "_cache"
    if not cache_dir.is_dir():
        return
    stem = parquet_path.stem  # e.g. "results_bm25" or "cyro_qa_cache"
    # For results_*.parquet → enriched_bm25.parquet / enriched_approximation.parquet
    strategy = stem.replace("results_", "")
    for pattern in [f"enriched_{strategy}.parquet", "metrics_by_strategy.json", "decile_metrics_by_strategy.json"]:
        target = cache_dir / pattern
        if target.exists():
            target.unlink()
            logger.info("  cache invalidated: %s", target.name)


def backfill_parquet(path: Path, dry_run: bool) -> bool:
    """Backfill *path* in-place. Returns True if the file was modified."""
    df = pd.read_parquet(path)

    needs_uw = COL_DECILE_UNWEIGHTED not in df.columns
    needs_cw = COL_DECILE_CHUNK_WEIGHTED not in df.columns
    if not needs_uw and not needs_cw:
        return False

    if "popularity_avg" not in df.columns:
        logger.warning("  SKIP %s — no popularity_avg column", path)
        return False

    metadata_path = _find_metadata(path.parent)
    if metadata_path is None:
        logger.warning("  SKIP %s — no metadata.json with boundaries found", path)
        return False

    boundaries_uw, boundaries_cw, _ = load_boundaries_from_metadata(metadata_path)

    pop = df["popularity_avg"].values.astype(float)

    if needs_uw:
        df[COL_DECILE_UNWEIGHTED] = assign_decile(pop, boundaries_uw)
        n_null = df[COL_DECILE_UNWEIGHTED].isna().sum()
        if n_null:
            logger.warning("  %d rows with unmapped unweighted decile in %s", n_null, path.name)
        else:
            df[COL_DECILE_UNWEIGHTED] = df[COL_DECILE_UNWEIGHTED].astype(int)

    if needs_cw:
        df[COL_DECILE_CHUNK_WEIGHTED] = assign_decile(pop, boundaries_cw)
        n_null = df[COL_DECILE_CHUNK_WEIGHTED].isna().sum()
        if n_null:
            logger.warning("  %d rows with unmapped chunk-weighted decile in %s", n_null, path.name)
        else:
            df[COL_DECILE_CHUNK_WEIGHTED] = df[COL_DECILE_CHUNK_WEIGHTED].astype(int)

    added = []
    if needs_uw:
        added.append(COL_DECILE_UNWEIGHTED)
    if needs_cw:
        added.append(COL_DECILE_CHUNK_WEIGHTED)
    logger.info("  + columns %s → %s", added, path)

    if dry_run:
        logger.info("  [dry-run] would write %s", path)
        return True

    tmp = path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)

    _invalidate_cache(path)

    del df
    gc.collect()
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing files")
    args = parser.parse_args()

    if args.dry_run:
        logger.info("DRY-RUN mode — no files will be modified")

    # Collect candidate files
    candidates: list[Path] = []
    for pattern in ["**/results_*.parquet", "**/cyro_qa_cache.parquet"]:
        for p in DATA_DIR.rglob(pattern.lstrip("**/")):
            # Skip cache directories and enriched files
            if "_cache" in p.parts or p.stem.startswith("enriched_"):
                continue
            candidates.append(p)

    candidates.sort()
    logger.info("Found %d candidate parquet files", len(candidates))

    modified = 0
    skipped = 0
    already_ok = 0

    for path in candidates:
        logger.info("Checking: %s", path.relative_to(ROOT_DIR))
        result = backfill_parquet(path, dry_run=args.dry_run)
        if result:
            modified += 1
        else:
            # Distinguish already-ok from skipped-with-warning
            df_check = pd.read_parquet(path, columns=None).iloc[:0]
            if COL_DECILE_UNWEIGHTED in df_check.columns and COL_DECILE_CHUNK_WEIGHTED in df_check.columns:
                already_ok += 1
            else:
                skipped += 1

    logger.info(
        "Done. modified=%d  already_ok=%d  skipped=%d",
        modified, already_ok, skipped,
    )
    if skipped:
        logger.warning("%d file(s) could not be backfilled (see warnings above)", skipped)


if __name__ == "__main__":
    main()
