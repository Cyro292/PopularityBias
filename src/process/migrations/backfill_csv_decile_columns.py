"""Backfill decile_unweighted and decile_chunk_weighted into evaluation CSV metadata blobs.

For each evaluation_results_*.csv under data/wiki_full_bil/, this script:
  1. Finds the sibling cyro_qa_cache.parquet in the same directory.
  2. Builds a wikipedia_id -> (decile_unweighted, decile_chunk_weighted) lookup.
  3. Parses each row's metadata blob (Python repr dict), injects both new keys.
  4. Writes the updated CSV back atomically via a .tmp file (original kept as .backup).

Run with:
    python -m src.process.migrations.backfill_csv_decile_columns
"""
from __future__ import annotations

import ast
import logging
import shutil
from pathlib import Path

import pandas as pd

from config import DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# === Constants ===

DATA_ROOT = DATA_DIR / "wiki_full_bil"
COL_UNWEIGHTED = "pop_decile_unweighted"
COL_CHUNK_WEIGHTED = "pop_decile_chunk_weighted"


# === Helpers ===

def load_decile_lookup(cache_parquet: Path) -> dict[int, tuple[int, int]]:
    """Load wikipedia_id -> (decile_unweighted, decile_chunk_weighted) from parquet.

    Args:
        cache_parquet: Path to cyro_qa_cache.parquet.

    Returns:
        Dict mapping wikipedia_id (int) to a tuple of (unweighted, chunk_weighted) decile ints.

    Raises:
        FileNotFoundError: If cache_parquet does not exist.
        KeyError: If required columns are missing.
    """
    df = pd.read_parquet(cache_parquet, columns=["wikipedia_id", COL_UNWEIGHTED, COL_CHUNK_WEIGHTED])
    lookup: dict[int, tuple[int, int]] = {
        int(row.wikipedia_id): (int(row.pop_decile_unweighted), int(row.pop_decile_chunk_weighted))
        for row in df.itertuples(index=False)
    }
    logger.info(f"  Loaded {len(lookup):,} entries from {cache_parquet.name}")
    return lookup


def patch_metadata(raw: str, decile_unweighted: int, decile_chunk_weighted: int) -> str:
    """Parse a Python-repr metadata string, inject decile keys, re-serialise.

    Args:
        raw: Raw metadata string as stored in the CSV (Python dict repr).
        decile_unweighted: Value to write for 'decile_unweighted'.
        decile_chunk_weighted: Value to write for 'decile_chunk_weighted'.

    Returns:
        Updated repr string with both new keys added (or overwritten if already present).

    Raises:
        ValueError: If raw cannot be parsed as a dict.
    """
    try:
        meta: dict = ast.literal_eval(raw)
    except Exception as exc:
        raise ValueError(f"Cannot parse metadata blob: {exc!r}") from exc
    meta["decile_unweighted"] = decile_unweighted
    meta["decile_chunk_weighted"] = decile_chunk_weighted
    return repr(meta)


def backfill_csv(csv_path: Path, lookup: dict[int, tuple[int, int]]) -> None:
    """Inject decile columns into a single CSV's metadata blobs.

    Args:
        csv_path: Path to the evaluation results CSV.
        lookup: wikipedia_id -> (decile_unweighted, decile_chunk_weighted).
    """
    df = pd.read_csv(csv_path)

    if "metadata" not in df.columns:
        logger.warning(f"  Skipping {csv_path.name}: no 'metadata' column")
        return

    missing = 0
    already_done = 0

    def _patch_row(raw: str) -> str:
        nonlocal missing, already_done
        try:
            meta: dict = ast.literal_eval(raw)
        except Exception:
            return raw  # leave unparseable rows as-is

        # Skip rows that already have both fields correctly set
        if "decile_unweighted" in meta and "decile_chunk_weighted" in meta:
            already_done += 1
            return raw

        wiki_id = int(meta.get("wikipedia_id", -1))
        if wiki_id not in lookup:
            missing += 1
            # Preserve existing decile as fallback for both fields
            fallback = int(meta.get("decile", -1))
            meta.setdefault("decile_unweighted", fallback)
            meta.setdefault("decile_chunk_weighted", fallback)
            return repr(meta)

        uw, cw = lookup[wiki_id]
        meta["decile_unweighted"] = uw
        meta["decile_chunk_weighted"] = cw
        return repr(meta)

    df["metadata"] = df["metadata"].apply(_patch_row)

    if already_done == len(df):
        logger.info(f"  {csv_path.name}: already fully backfilled, skipping write")
        return

    # Atomic write: write to .tmp then rename; keep .backup of original
    backup_path = csv_path.with_suffix(".backup.csv")
    tmp_path = csv_path.with_suffix(".tmp.csv")

    if not backup_path.exists():
        shutil.copy2(csv_path, backup_path)
        logger.info(f"  Backup saved → {backup_path.name}")

    df.to_csv(tmp_path, index=False)
    tmp_path.rename(csv_path)

    logger.info(
        f"  {csv_path.name}: {len(df)} rows patched "
        f"({missing} missing in lookup, {already_done} already done)"
    )


# === Main ===

def main() -> None:
    """Iterate all evaluation folders and backfill decile columns in every CSV."""
    eval_dirs = sorted(DATA_ROOT.glob("evaluation_results_*/"))
    if not eval_dirs:
        logger.error(f"No evaluation_results_* directories found under {DATA_ROOT}")
        return

    total_csv = 0
    for eval_dir in eval_dirs:
        cache_parquet = eval_dir / "cyro_qa_cache.parquet"
        if not cache_parquet.exists():
            logger.warning(f"No cyro_qa_cache.parquet in {eval_dir.name}, skipping")
            continue

        logger.info(f"Processing {eval_dir.name} ...")
        lookup = load_decile_lookup(cache_parquet)

        csv_files = sorted(eval_dir.glob("evaluation_results_*.csv"))
        # Exclude backup files
        csv_files = [p for p in csv_files if ".backup" not in p.name]

        for csv_path in csv_files:
            backfill_csv(csv_path, lookup)
            total_csv += 1

    logger.info(f"Done. Processed {total_csv} CSV files across {len(eval_dirs)} directories.")


if __name__ == "__main__":
    main()
