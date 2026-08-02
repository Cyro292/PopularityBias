"""Backfill ``retrieved_doc_popularity`` and ``popularity_avg`` into legacy evaluation CSVs.

One-shot migration script.  For every ``evaluation_results_*.csv`` under
``data/``, this script:

1. **Copies** the original CSV to ``<name>.backup.csv`` (never deletes).
2. Parses the ``metadata`` column (Python-repr dict).
3. Looks up the ``popularity_avg`` of each ID in ``retrieved_doc_ids``
   using a pre-built lookup from the corpus parquet.
4. Injects ``"retrieved_doc_popularity": [...]`` into the metadata dict.
5. Looks up the target document's ``popularity_avg`` via its ``wikipedia_id``
   and injects ``"popularity_avg": <float>`` into the metadata dict.
6. Writes the updated CSV back to the original path.

Usage::

    python -m src.process.migrations.backfill_retrieved_popularity

The corpus parquet is scanned **once** to build an in-memory
``wikipedia_id → popularity_avg`` mapping, then reused for all CSVs.
"""

from __future__ import annotations

import ast
import gc
import logging
import shutil
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from config import DATA_DIR
from src.metrics.decile_utils import COL_POPULARITY

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────
COLLECTION_NAME = "wiki_full_bil"
CORPUS_PATH = DATA_DIR / COLLECTION_NAME / "wiki_corpus.parquet"
BATCH_SIZE = 100_000


# === Step 1: build wikipedia_id → popularity_avg lookup from corpus ========

def build_popularity_lookup(corpus_path: Path) -> dict[int, float]:
    """Stream the corpus parquet and return {wikipedia_id: popularity_avg}.

    Only reads two columns so memory stays manageable even for large corpora.
    """
    logger.info("Building popularity lookup from %s …", corpus_path)
    pf = pq.ParquetFile(corpus_path)
    total_rows = pf.metadata.num_rows
    n_batches = (total_rows + BATCH_SIZE - 1) // BATCH_SIZE

    lookup: dict[int, float] = {}
    for batch in tqdm(
        pf.iter_batches(batch_size=BATCH_SIZE, columns=["wikipedia_id", COL_POPULARITY]),
        total=n_batches,
        desc="Corpus popularity scan",
        unit="batch",
    ):
        wids = batch.column("wikipedia_id").to_pylist()
        pops = batch.column(COL_POPULARITY).to_pylist()
        for wid, pop in zip(wids, pops):
            if wid is None or pop is None:
                continue
            wid_int = int(wid)
            if wid_int not in lookup:
                lookup[wid_int] = float(pop)

        del wids, pops
        gc.collect()

    logger.info("Lookup built: %s unique wikipedia_ids.", f"{len(lookup):,}")
    return lookup


# === Step 2: discover all evaluation CSVs ==================================

def discover_csvs(base_dir: Path) -> list[Path]:
    """Find all evaluation_results_*.csv files under *base_dir*."""
    csvs = sorted(base_dir.rglob("evaluation_results_*.csv"))
    # Exclude backup files
    csvs = [p for p in csvs if ".backup." not in p.name]
    return csvs


# === Step 3: parse & update metadata =======================================

def _parse_metadata(val: str) -> dict:
    """Parse a Python-repr metadata string back into a dict."""
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except Exception:
            pass
    return {}


def _parse_id_list(val) -> list[int]:
    """Coerce retrieved_doc_ids (may be list or repr string) to list[int]."""
    if isinstance(val, list):
        return [int(v) for v in val]
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list):
                return [int(v) for v in parsed]
        except Exception:
            pass
    return []


def update_csv(
    csv_path: Path,
    lookup: dict[int, float],
) -> None:
    """Back up and update a single evaluation CSV.

    Backfills two fields into each row's metadata dict:
      - ``retrieved_doc_popularity``: list of popularity values for retrieved docs.
      - ``popularity_avg``: the target/query document's own popularity value.

    Fields that already exist are left untouched.
    """
    # ── Backup ─────────────────────────────────────────────────────────────
    backup_path = csv_path.with_suffix(".backup.csv")
    if backup_path.exists():
        logger.info("  Backup already exists: %s — skipping backup step.", backup_path.name)
    else:
        shutil.copy2(csv_path, backup_path)
        logger.info("  Created backup: %s", backup_path.name)

    # ── Load ───────────────────────────────────────────────────────────────
    df = pd.read_csv(csv_path)
    if "metadata" not in df.columns:
        logger.warning("  No 'metadata' column — skipping %s", csv_path)
        return

    updated_rdp = 0   # rows where retrieved_doc_popularity was injected
    updated_pa = 0     # rows where popularity_avg was injected
    skipped_rdp = 0
    skipped_pa = 0

    new_meta_strs: list[str] = []
    for raw_meta in df["metadata"]:
        meta = _parse_metadata(raw_meta)

        # ── retrieved_doc_popularity ───────────────────────────────────────
        if "retrieved_doc_popularity" in meta:
            skipped_rdp += 1
        else:
            doc_ids = _parse_id_list(meta.get("retrieved_doc_ids", []))
            if not doc_ids:
                meta["retrieved_doc_popularity"] = []
            else:
                meta["retrieved_doc_popularity"] = [
                    lookup.get(wid, 0) for wid in doc_ids
                ]
            updated_rdp += 1

        # ── popularity_avg (target doc popularity) ─────────────────────────
        if "popularity_avg" in meta and meta["popularity_avg"] is not None:
            skipped_pa += 1
        else:
            wid_raw = meta.get("wikipedia_id")
            if wid_raw is not None:
                try:
                    wid_int = int(wid_raw)
                    meta["popularity_avg"] = lookup.get(wid_int)
                    updated_pa += 1
                except (ValueError, TypeError):
                    meta["popularity_avg"] = None
                    updated_pa += 1
            else:
                meta["popularity_avg"] = None
                updated_pa += 1

        new_meta_strs.append(str(meta))

    df["metadata"] = new_meta_strs
    df.to_csv(csv_path, index=False)
    logger.info(
        "  retrieved_doc_popularity — updated: %s, skipped: %s",
        f"{updated_rdp:,}", f"{skipped_rdp:,}",
    )
    logger.info(
        "  popularity_avg — updated: %s, skipped: %s. Saved → %s",
        f"{updated_pa:,}", f"{skipped_pa:,}", csv_path.name,
    )


# === Main ===================================================================

def main() -> None:
    if not CORPUS_PATH.exists():
        logger.error("Corpus not found at %s", CORPUS_PATH)
        sys.exit(1)

    lookup = build_popularity_lookup(CORPUS_PATH)

    csvs = discover_csvs(DATA_DIR)
    if not csvs:
        logger.warning("No evaluation CSVs found under %s", DATA_DIR)
        sys.exit(0)

    logger.info("Found %d evaluation CSVs to process.", len(csvs))

    for csv_path in csvs:
        logger.info("Processing: %s", csv_path.relative_to(DATA_DIR))
        update_csv(csv_path, lookup)

    logger.info("Done — all CSVs updated.  Backups saved as *.backup.csv.")


if __name__ == "__main__":
    main()
