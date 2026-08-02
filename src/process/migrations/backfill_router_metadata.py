"""Back-fill ``models/metadata.json`` from existing ``router_*.pt`` files.

Every router checkpoint written by ``RouterService.save_model`` contains the
final ``model_config`` (architecture) and the per-epoch ``history`` (training
metrics). What the ``.pt`` file does **not** contain is the training
configuration that produced it — label mode, retrieval metric, learning rate,
``filter_no_result``, etc. For models trained before the metadata-recording
hook in :mod:`src.router.train_router` existed, we recover what we can from
the ``.pt`` file and infer the rest from filename conventions.

Strategy
--------
For every ``models/router_*.pt``:

1. Read architecture and final metrics from the ``.pt`` payload.
2. Compute file metadata (size, mtime, SHA-256).
3. Infer training-config fields from filename patterns:

     * ``mrr``    → ``label_mode="retrieval"``, ``retrieval_metric="mrr"``
     * ``recall`` → ``retrieval_metric="recall"``
     * ``no_pop`` → ``include_popularity=False``
     * ``unfreeze1`` → ``unfreeze_layers=1``
     * ``filter`` → ``filter_no_result=True``
     * default → ``label_mode="answer"``, ``retrieval_metric=null``

4. Cross-check inferred values against the architecture read from the ``.pt``
   file. Disagreements are logged as warnings and the ``.pt`` value wins.
5. Mark the entry with ``"backfilled": true`` and list every guessed field
   in ``"inferred_fields"``.
6. Hand-curated ``notes`` are added per model based on the analysis from the
   popularity-bias notebooks.

Usage::

    python -m src.process.migrations.backfill_router_metadata --dry-run
    python -m src.process.migrations.backfill_router_metadata

Existing entries with ``"backfilled": false`` are never overwritten — those
were recorded at training time and are authoritative.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from src.router.metadata import (
    BACKFILL_INFERRED_FIELDS,
    SCHEMA_VERSION,
    compute_file_meta,
    extract_from_pt,
    load_metadata,
    save_metadata,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Hand-curated notes for the 8 known models ───────────────────────────────
# Keys are model stems (e.g. "router_bert_answer"). Unknown models get "".

PREFILLED_NOTES: dict[str, str] = {
    "router_bert_answer": (
        "answer-mode baseline with popularity feature. Trained 4 epochs only. "
        "Locked to ~85% BM25 (mild popularity insensitivity). "
        "Performance ≈ bm25_plus alone."
    ),
    "router_mrr_no_pop": (
        "Retrieval-mode (MRR@20), no popularity feature. Trained 4 epochs only. "
        "Best overall accuracy at 59.8% (+2.3pp vs faiss_hybrid). "
        "Shows BERT picks up popularity from query text alone — explicit pop "
        "feature not strictly necessary."
    ),
    "router_plain_bert": (
        "answer-mode with popularity, frozen BERT — 'plain' baseline config. "
        "Trained 4 epochs only. Likely a re-run of router_bert_answer with "
        "same hyperparameters."
    ),
    "router_plain_bert_mrr_filter": (
        "Retrieval-mode (MRR@20) + filter_no_result + popularity. Trained 4 epochs only. "
        "CURRENT BEST: 60.2% overall (+2.8pp vs faiss_hybrid). Cleanest "
        "popularity-conditional routing — P(bm25) drops from 0.67 (low-pop) "
        "to 0.51 (high-pop). Wins +7.3pp on pop_qa high-pop slice."
    ),
    "router_plain_bert_no_pop_answer": (
        "answer-mode, no popularity, frozen BERT. Trained 4 epochs only. "
        "Locked to ~84% BM25 — without pop feature the answer-mode loss "
        "fails to differentiate."
    ),
    "router_pop_after_bert": (
        "Architectural variant: popularity concatenated AFTER BERT CLS token. "
        "include_popularity=None in .pt (not the standard path). "
        "Produces CONSTANT P(bm25)=0.487 across all deciles — popularity "
        "feature is effectively a no-op. Architecture is broken by design."
    ),
    "router_unfreeze1_bert_answer": (
        "answer-mode + unfreeze_layers=1 + popularity. Trained 4 epochs only. "
        "Collapsed to 100% BM25 (mode collapse). Unfreezing with answer-mode "
        "labels caused memorisation of the majority class."
    ),
    "router_unfreeze1_no_pop_answer": (
        "answer-mode + unfreeze_layers=1, no popularity. Trained 4 epochs only. "
        "Collapsed to 100% BM25 (mode collapse) — same failure as "
        "router_unfreeze1_bert_answer. Confirms unfreezing + answer-mode loss "
        "is the broken combination, independent of popularity feature."
    ),
}


# ─── Filename-based training-config inference ────────────────────────────────

def infer_training_from_name(stem: str) -> dict[str, Any]:
    """Infer training-config fields from a router filename stem.

    Args:
        stem: Filename without extension, e.g. ``"router_plain_bert_mrr_filter"``.

    Returns:
        Dict of training-config values (only fields that can be inferred
        from the name). Other fields are filled by the caller.
    """
    name = stem.lower()
    inferred: dict[str, Any] = {
        "collection_name": "wiki_full_bil",
        "dataset_dir": "all_qa_8k",
        "backends_to_train": ["bm25_plus", "ivfpq_high"],
        "exclude_datasets": [],
        "llm": "neo",
        "batch_size": 32,
        "lr": 0.001,
        "bert_lr": 2e-5,
        "retrieval_k": None,
        "filter_no_result": False,
        "label_mode": "answer",
        "retrieval_metric": None,
    }

    # label_mode / retrieval_metric
    if "mrr" in name:
        inferred["label_mode"] = "retrieval"
        inferred["retrieval_metric"] = "mrr"
        inferred["retrieval_k"] = 20
    elif "recall" in name:
        inferred["label_mode"] = "retrieval"
        inferred["retrieval_metric"] = "recall"
        inferred["retrieval_k"] = 10

    # include_popularity (cross-checked later against .pt)
    if "no_pop" in name:
        inferred["include_popularity"] = False
    else:
        inferred["include_popularity"] = True

    # unfreeze_layers (cross-checked later against .pt)
    inferred["unfreeze_layers"] = 1 if "unfreeze1" in name else 0

    # filter_no_result
    if "filter" in name:
        inferred["filter_no_result"] = True

    return inferred


def cross_check_inference(
    inferred: dict[str, Any], architecture: dict[str, Any], stem: str
) -> list[str]:
    """Verify name-based guesses against the architecture read from .pt.

    Args:
        inferred: Training-config dict (mutated in place to fix disagreements).
        architecture: ``architecture`` block read from the .pt file.
        stem: Model stem, for warning messages.

    Returns:
        List of human-readable warning strings (empty if all checks pass).
    """
    warnings: list[str] = []

    # include_popularity: trust .pt over name
    pt_pop = architecture.get("include_popularity")
    if pt_pop is not None and inferred.get("include_popularity") != pt_pop:
        warnings.append(
            f"  ⚠ {stem}: include_popularity inferred={inferred['include_popularity']} "
            f"but .pt says {pt_pop} — trusting .pt"
        )
        inferred["include_popularity"] = pt_pop

    # unfreeze_layers: trust .pt over name
    pt_unfreeze = architecture.get("unfreeze_layers", 0)
    if inferred.get("unfreeze_layers") != pt_unfreeze:
        warnings.append(
            f"  ⚠ {stem}: unfreeze_layers inferred={inferred['unfreeze_layers']} "
            f"but .pt says {pt_unfreeze} — trusting .pt"
        )
        inferred["unfreeze_layers"] = pt_unfreeze

    return warnings


# ─── Per-model entry assembly ────────────────────────────────────────────────

def build_entry(pt_path: Path) -> dict[str, Any]:
    """Build a full backfilled metadata entry for one ``.pt`` file.

    Args:
        pt_path: Path to the router checkpoint.

    Returns:
        Metadata entry dict (ready to slot under ``models/<name>``).
    """
    pt_path = Path(pt_path).resolve()
    stem = pt_path.stem

    architecture, metrics = extract_from_pt(pt_path)
    file_meta = compute_file_meta(pt_path)
    inferred_training = infer_training_from_name(stem)

    cross_check_warnings = cross_check_inference(inferred_training, architecture, stem)
    for w in cross_check_warnings:
        logger.warning(w)

    # epochs: best guess is the actual number completed (from history length)
    epochs_completed = metrics.get("epochs_completed") or 0
    inferred_training["epochs"] = epochs_completed

    notes = PREFILLED_NOTES.get(stem, "")

    return {
        "file": file_meta,
        "training": inferred_training,
        "architecture": architecture,
        "metrics": metrics,
        "notes": notes,
        "backfilled": True,
        "inferred_fields": list(BACKFILL_INFERRED_FIELDS),
    }


# ─── Main loop ───────────────────────────────────────────────────────────────

def backfill(
    models_dir: Path,
    metadata_path: Path,
    *,
    dry_run: bool = False,
    overwrite_backfilled: bool = False,
) -> dict[str, Any]:
    """Walk *models_dir* for ``router_*.pt`` files and update metadata.

    Args:
        models_dir: Directory containing the ``.pt`` files.
        metadata_path: Path to the ``metadata.json`` to update.
        dry_run: If True, only print what would change — do not write.
        overwrite_backfilled: If True, also overwrite entries that were
            previously backfilled. Entries with ``backfilled=false`` are
            *always* preserved (they were recorded at training time).

    Returns:
        The new metadata dict.
    """
    models_dir = Path(models_dir).resolve()
    pt_files = sorted(models_dir.glob("router_*.pt"))
    if not pt_files:
        logger.warning("No router_*.pt files found in %s — nothing to backfill.", models_dir)
        return load_metadata(metadata_path)

    metadata = load_metadata(metadata_path)
    metadata.setdefault("schema_version", SCHEMA_VERSION)
    models = metadata.setdefault("models", {})

    added: list[str] = []
    updated: list[str] = []
    skipped: list[str] = []

    for pt in pt_files:
        name = pt.stem
        existing = models.get(name)

        if existing and not existing.get("backfilled", False) and not overwrite_backfilled:
            # Recorded at training time — authoritative, do not touch.
            skipped.append(f"{name} (recorded at training time)")
            continue

        if existing and existing.get("backfilled", False) and not overwrite_backfilled:
            # Already backfilled — skip unless explicitly overwriting.
            skipped.append(f"{name} (already backfilled)")
            continue

        try:
            entry = build_entry(pt)
        except Exception as e:
            logger.error("  ✗ %s: %s", name, e)
            continue

        if existing:
            # Preserve any user-edited notes if the backfill entry has none.
            if not entry.get("notes") and existing.get("notes"):
                entry["notes"] = existing["notes"]
            updated.append(name)
        else:
            added.append(name)

        models[name] = entry

        logger.info(
            "  ✓ %s — pop=%s, unfreeze=%s, label=%s, %d epochs, test_acc=%s",
            name,
            entry["training"].get("include_popularity"),
            entry["training"].get("unfreeze_layers"),
            entry["training"].get("label_mode"),
            entry["metrics"].get("epochs_completed"),
            (
                f"{entry['metrics']['final_test_acc']:.2%}"
                if entry["metrics"].get("final_test_acc") is not None
                else "n/a"
            ),
        )

    logger.info(
        "Backfill summary: %d added, %d updated, %d skipped",
        len(added), len(updated), len(skipped),
    )
    if skipped:
        for s in skipped:
            logger.info("  · skip %s", s)

    if not dry_run and (added or updated):
        save_metadata(metadata, metadata_path)
    elif dry_run and (added or updated):
        logger.info("Dry-run — metadata.json NOT written.")

    return metadata


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Back-fill models/metadata.json from existing router_*.pt files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models-dir", type=Path, default=Path("models"),
        help="Directory containing router_*.pt files.",
    )
    parser.add_argument(
        "--metadata-path", type=Path, default=None,
        help="Output metadata.json path (default: <models-dir>/metadata.json).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be written, but don't touch the file.",
    )
    parser.add_argument(
        "--overwrite-backfilled", action="store_true",
        help="Re-backfill entries that were already backfilled. "
             "Training-time entries (backfilled=false) are always preserved.",
    )
    args = parser.parse_args(argv)

    metadata_path = args.metadata_path or (args.models_dir / "metadata.json")
    backfill(
        args.models_dir,
        metadata_path,
        dry_run=args.dry_run,
        overwrite_backfilled=args.overwrite_backfilled,
    )


if __name__ == "__main__":
    main()
