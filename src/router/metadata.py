"""Persist training metadata for router models.

Every trained router in ``models/<name>.pt`` is described by a single entry in
``models/metadata.json``. The metadata captures:

- **Training config** — the :class:`~src.router.train_router.RouterTrainingConfig`
  fields used (label mode, retrieval metric, hyperparameters, etc.).
- **Architecture** — read directly from the ``model_config`` block inside the
  ``.pt`` file (input dim, hidden dims, dropout, unfreeze_layers, …).
- **Training metrics** — final train/test loss + accuracy and number of
  epochs actually completed (from the ``history`` block of the ``.pt`` file).
- **File metadata** — size, mtime and SHA-256 of the ``.pt`` file.
- **Notes** — free-form, hand-editable string for hypotheses and observations.

For models trained *before* this module existed, run
``python -m src.process.migrations.backfill_router_metadata`` to retroactively populate
their entries from the ``.pt`` contents and filename conventions. Such
backfilled entries carry ``"backfilled": true`` and list every field that
had to be guessed in ``"inferred_fields"``.

The metadata file uses ``schema_version`` so future format changes can be
detected and migrated.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from config import ROOT_DIR

logger = logging.getLogger(__name__)


# ─── Constants ───────────────────────────────────────────────────────────────

SCHEMA_VERSION: int = 1
METADATA_PATH: Path = ROOT_DIR / "models" / "metadata.json"

# Fields that are always inferred during backfill (cannot be recovered from
# the .pt file alone). Used by the router metadata migration.
BACKFILL_INFERRED_FIELDS: tuple[str, ...] = (
    "collection_name",
    "dataset_dir",
    "backends_to_train",
    "exclude_datasets",
    "llm",
    "label_mode",
    "retrieval_metric",
    "retrieval_k",
    "epochs",
    "batch_size",
    "lr",
    "bert_lr",
    "filter_no_result",
    "patience",
)


# ─── Public helpers ──────────────────────────────────────────────────────────

def load_metadata(path: Path | None = None) -> dict[str, Any]:
    """Load the router metadata JSON, returning an empty skeleton if missing.

    Args:
        path: Override the default ``models/metadata.json`` location.

    Returns:
        Dict with ``{"schema_version": int, "models": {<name>: {...}}}``.
    """
    p = Path(path) if path else METADATA_PATH
    if not p.exists():
        return {"schema_version": SCHEMA_VERSION, "models": {}}
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("metadata.json unreadable (%s) — returning empty skeleton", e)
        return {"schema_version": SCHEMA_VERSION, "models": {}}

    # Normalise legacy shapes
    if "models" not in data:
        data = {"schema_version": SCHEMA_VERSION, "models": data}
    data.setdefault("schema_version", SCHEMA_VERSION)
    data["models"] = data.get("models", {})
    return data


def save_metadata(metadata: dict[str, Any], path: Path | None = None) -> None:
    """Atomically write the metadata JSON.

    Args:
        metadata: Full metadata dict (with ``schema_version`` and ``models``).
        path: Override the default ``models/metadata.json`` location.
    """
    p = Path(path) if path else METADATA_PATH
    p.parent.mkdir(parents=True, exist_ok=True)

    # Atomic write: tempfile in same directory, then rename.
    fd, tmp = tempfile.mkstemp(prefix=".metadata.", suffix=".json", dir=str(p.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=False)
            f.write("\n")
        os.replace(tmp, p)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    logger.info("Wrote metadata → %s (%d models)", p, len(metadata.get("models", {})))


def compute_file_meta(pt_path: Path) -> dict[str, Any]:
    """Compute size, mtime and SHA-256 for a router ``.pt`` file.

    Args:
        pt_path: Absolute path to the ``.pt`` file.

    Returns:
        Dict with ``path`` (basename), ``size_bytes``, ``mtime`` (ISO 8601)
        and ``sha256``.
    """
    pt_path = Path(pt_path).resolve()
    stat = pt_path.stat()
    mtime_iso = _dt.datetime.fromtimestamp(
        stat.st_mtime, tz=_dt.timezone.utc
    ).isoformat(timespec="seconds")

    sha = hashlib.sha256()
    with pt_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha.update(chunk)

    return {
        "path": pt_path.name,
        "size_bytes": int(stat.st_size),
        "mtime": mtime_iso,
        "sha256": sha.hexdigest(),
    }


def extract_from_pt(pt_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read architecture and final-epoch metrics from a router ``.pt`` file.

    Args:
        pt_path: Path to the ``.pt`` file produced by ``RouterService.save_model``.

    Returns:
        Tuple of ``(architecture_dict, metrics_dict)``.

    Raises:
        FileNotFoundError: If *pt_path* does not exist.
        KeyError: If the ``.pt`` payload is missing ``model_config`` or
            ``history`` (i.e. not a router checkpoint).
    """
    pt_path = Path(pt_path).resolve()
    if not pt_path.exists():
        raise FileNotFoundError(f"extract_from_pt: {pt_path} does not exist")

    # Heavy import — keep it lazy so importing metadata.py is cheap.
    import torch

    payload = torch.load(pt_path, map_location="cpu", weights_only=False)

    cfg = payload.get("model_config")
    if cfg is None:
        raise KeyError(f"{pt_path.name}: missing 'model_config' block")
    hist = payload.get("history", {})

    architecture = {
        "input_dim": cfg.get("input_dim"),
        "hidden_dim1": cfg.get("hidden_dim1"),
        "hidden_dim2": cfg.get("hidden_dim2"),
        "num_classes": cfg.get("num_classes"),
        "dropout": cfg.get("dropout"),
        "unfreeze_layers": cfg.get("unfreeze_layers", 0),
        "include_popularity": cfg.get("include_popularity"),
    }

    metrics: dict[str, Any] = {
        "epochs_completed": len(hist.get("train_loss", [])),
        # best_epoch / best_test_loss / best_test_acc / stopped_early were
        # added after these models were trained — absent from the .pt files.
        "best_epoch": None,
        "best_test_loss": None,
        "best_test_acc": None,
        "stopped_early": None,
    }
    for key, src in (
        ("final_train_loss", "train_loss"),
        ("final_train_acc", "train_acc"),
        ("final_test_loss", "test_loss"),
        ("final_test_acc", "test_acc"),
    ):
        values = hist.get(src, [])
        metrics[key] = float(values[-1]) if values else None

    return architecture, metrics


def record_training(
    cfg: Any,
    result: dict[str, Any],
    model_path: Path,
    *,
    notes: str | None = None,
    history_file: Path | None = None,
    path: Path | None = None,
) -> dict[str, Any]:
    """Append or overwrite an entry for a freshly-trained model.

    Called from :func:`src.router.train_router.train_router` immediately after
    ``RouterService.save_model``. Reads architecture/metrics from *result* and
    file metadata from disk, then merges into ``models/metadata.json``.

    Existing ``notes`` are preserved unless an explicit *notes* argument is
    supplied (incl. empty string, which clears them).

    Args:
        cfg: The :class:`RouterTrainingConfig` used for the run.
        result: The dict returned by ``RouterService.train`` (contains
            ``model_config`` and ``history``).
        model_path: Path to the saved ``.pt`` file.
        notes: Optional override for the free-form notes field.
        history_file: Optional path to a per-epoch training history JSON file.
        path: Override the metadata file location.

    Returns:
        The entry that was written.
    """
    model_path = Path(model_path).resolve()
    name = model_path.stem  # e.g. "router_mrr_filter"

    metadata = load_metadata(path)
    models = metadata.setdefault("models", {})

    # Preserve any existing notes unless caller overrode them.
    prior = models.get(name, {})
    if notes is None:
        notes = prior.get("notes", "")

    arch_from_cfg = result.get("model_config", {})
    history = result.get("history", {})

    entry: dict[str, Any] = {
        "file": compute_file_meta(model_path),
        "history_file": str(history_file) if history_file is not None else None,
        "training": _training_dict_from_cfg(cfg),
        "architecture": {
            "input_dim": arch_from_cfg.get("input_dim"),
            "hidden_dim1": arch_from_cfg.get("hidden_dim1"),
            "hidden_dim2": arch_from_cfg.get("hidden_dim2"),
            "num_classes": arch_from_cfg.get("num_classes"),
            "dropout": arch_from_cfg.get("dropout"),
            "unfreeze_layers": arch_from_cfg.get("unfreeze_layers", 0),
            "include_popularity": arch_from_cfg.get("include_popularity"),
        },
        "metrics": {
            "epochs_completed": len(history.get("train_loss", [])),
            "best_epoch": result.get("best_epoch"),
            "best_test_loss": result.get("best_test_loss"),
            "best_test_acc": result.get("best_test_acc"),
            "stopped_early": result.get("stopped_early"),
            "final_train_loss": (
                float(history["train_loss"][-1]) if history.get("train_loss") else None
            ),
            "final_train_acc": (
                float(history["train_acc"][-1]) if history.get("train_acc") else None
            ),
            "final_test_loss": (
                float(history["test_loss"][-1]) if history.get("test_loss") else None
            ),
            "final_test_acc": (
                float(history["test_acc"][-1]) if history.get("test_acc") else None
            ),
        },
        "notes": notes,
        "backfilled": False,
        "inferred_fields": [],
    }

    models[name] = entry
    save_metadata(metadata, path)
    return entry


# ─── Private helpers ─────────────────────────────────────────────────────────

def _training_dict_from_cfg(cfg: Any) -> dict[str, Any]:
    """Pull every training-config attribute we care about from *cfg*.

    Handles both real :class:`RouterTrainingConfig` instances and plain dicts.
    """
    keys = (
        "collection_name",
        "dataset_dir",
        "backends_to_train",
        "exclude_datasets",
        "llm",
        "label_mode",
        "retrieval_metric",
        "retrieval_k",
        "epochs",
        "batch_size",
        "lr",
        "unfreeze_layers",
        "bert_lr",
        "include_popularity",
        "keep_ties",
        "patience",
        "dropout",
        "use_scheduler",
        "seed",
        "weight_decay",
        "bert_weight_decay",
        "wandb_project",
        "wandb_run_name",
        "warmup_epochs",
        "min_lr_ratio",
    )
    out: dict[str, Any] = {}
    for k in keys:
        out[k] = getattr(cfg, k, None) if not isinstance(cfg, dict) else cfg.get(k)
    return out
