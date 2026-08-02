"""Train query-conditioned backend transformations for fixed RRF.

The model learns how to reshape BM25 and FAISS internal rankings per query.
It does not learn fusion weights. Each backend is reranked using predicted
transformed position scores, and the final fusion remains plain unweighted RRF.

Training loss is a soft-rank MRR objective on the final fused ranking.

Usage
-----
    python -m src.router.train_fusion \\
        --collection wiki_full_bil \\
        --dataset-dir all_qa_8k \\
        --model-name fusion_v1 \\
        --backends bm25_plus ivfpq_high \\
        --rrf-depth 60 \\
        --epochs 80
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import DATA_DIR
from src.corpus_handler.analysis_dataset_handler import AnalysisDatasetHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

class FusionTrainingConfig:
    """Configuration for fixed-RRF backend transformation training.

    Attributes:
        collection_name: Data folder name under DATA_DIR.
        dataset_dir: Subdirectory containing the analysis dataset.
        model_name: Name for saved model file (``models/{model_name}.pt``).
        backends: Pair of backend keys ``[bm25_name, faiss_name]``.
        exclude_datasets: Dataset names to exclude (e.g. ``['fever']``).
        rrf_depth: Retrieval depth per backend (must match CSV checkpoint depth).
        rrf_k: RRF smoothing constant.
        temperature: Soft-rank temperature ``tau`` for backend and fused ranks.
        use_bert: Whether to encode questions with BERT.
        include_popularity: Concatenate popularity to BERT embedding.
        epochs: Maximum number of training epochs.
        batch_size: Training batch size.
        lr: Learning rate for the classifier head.
        unfreeze_layers: Number of BERT layers to unfreeze.
        bert_lr: Learning rate for unfrozen BERT layers.
        patience: Early-stopping patience on test MRR.
        dropout: Override classifier dropout (``None`` = auto).
        use_scheduler: Enable cosine LR scheduler.
        seed: Random seed.
        weight_decay: L2 weight decay for classifier head.
        bert_weight_decay: L2 weight decay for unfrozen BERT params.
        wandb_project: W&B project name.
        wandb_run_name: W&B run name override.
        save_history: Optional path to write per-epoch metrics JSON.
        warmup_epochs: Linear warmup epochs before cosine decay.
        min_lr_ratio: Minimum LR as fraction of initial LR.
    """

    collection_name: str = "wiki_full_bil"
    dataset_dir: str = "all_qa_8k"
    model_name: str = "fusion_v1"

    backends: list[str] | None = None
    exclude_datasets: list[str] | None = None

    rrf_depth: int = 60
    rrf_k: int = 60
    temperature: float = 0.5
    residual_scale: float = 1.0
    loss_top_k: int = 20

    use_bert: bool = True
    include_popularity: bool = True
    epochs: int = 80
    batch_size: int = 32
    lr: float = 0.001
    unfreeze_layers: int = 0
    bert_lr: float = 2e-5
    patience: int = 10
    dropout: float | None = None
    use_scheduler: bool = True
    seed: int = 42
    weight_decay: float = 1e-3
    bert_weight_decay: float = 1e-2
    warmup_epochs: int = 0
    min_lr_ratio: float = 0.1

    wandb_project: str = "popularity-bias-fusion"
    wandb_run_name: str | None = None
    save_history: str | None = None

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        if self.backends is None:
            self.backends = ["bm25_plus", "ivfpq_high"]
        if self.exclude_datasets is None:
            self.exclude_datasets = []


# ═══════════════════════════════════════════════════════════════════════════════
# Data preparation
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_fusion_data(cfg: FusionTrainingConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and prepare fusion training data.

    Returns:
        ``(train_df, test_df)`` DataFrames with per-backend doc ID lists
        and ground-truth wikipedia IDs.
    """
    logger.info("Loading fusion training data...")
    handler = AnalysisDatasetHandler(
        collection_name=cfg.collection_name,
        dataset_dir=cfg.dataset_dir,
    )

    df = handler.build_fusion_training_data(
        backends=cfg.backends,
        rrf_depth=cfg.rrf_depth,
        exclude_datasets=cfg.exclude_datasets,
    )

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    logger.info(f"Train: {len(train_df):,} | Test: {len(test_df):,}")

    _log_sample_weights(train_df, cfg.backends, cfg.rrf_k)

    return train_df, test_df


def _log_sample_weights(
    df: pd.DataFrame,
    backends: list[str],
    rrf_k: int,
) -> None:
    """Log example backend gold ranks for a few questions."""
    bm25_key = f"{backends[0]}_doc_ids"
    faiss_key = f"{backends[1]}_doc_ids"

    logger.info("Sample backend gold ranks (first 3 questions):")
    for i in range(min(3, len(df))):
        row = df.iloc[i]
        gold = str(row["wikipedia_id"])
        bm25_ids = row[bm25_key]
        faiss_ids = row[faiss_key]

        bm25_rank = None
        for r, did in enumerate(bm25_ids, start=1):
            if did == gold:
                bm25_rank = r
                break

        faiss_rank = None
        for r, did in enumerate(faiss_ids, start=1):
            if did == gold:
                faiss_rank = r
                break

        if bm25_rank and faiss_rank:
            b_score = 1.0 / (rrf_k + bm25_rank)
            f_score = 1.0 / (rrf_k + faiss_rank)
            logger.info(
                f"  q={row['question_id']}: BM25 rank={bm25_rank}, "
                f"FAISS rank={faiss_rank}, baseline RRF score={b_score + f_score:.4f}"
            )
        elif bm25_rank:
            logger.info(f"  q={row['question_id']}: BM25 rank={bm25_rank}, FAISS not found → alpha≈1.0")
        elif faiss_rank:
            logger.info(f"  q={row['question_id']}: BM25 not found, FAISS rank={faiss_rank} → beta≈1.0")
        else:
            logger.info(f"  q={row['question_id']}: Gold not found in either backend")


# ═══════════════════════════════════════════════════════════════════════════════
# Training (Modal GPU)
# ═══════════════════════════════════════════════════════════════════════════════

def train_fusion(cfg: FusionTrainingConfig) -> dict:
    """Train the backend transformation model on Modal GPU.

    Returns:
        Training result dict containing model weights, scaler params,
        config, and history.
    """
    train_df, test_df = prepare_fusion_data(cfg)

    logger.info("Connecting to Modal GPU service...")
    from src.router.fusion_modal_service import FusionModalService

    service = FusionModalService()

    logger.info(
        f"Training fusion transform model (epochs={cfg.epochs}, batch_size={cfg.batch_size}, "
        f"patience={cfg.patience}, temperature={cfg.temperature})..."
    )

    train_records = train_df.to_dict("records")
    test_records = test_df.to_dict("records")

    result = service.train(
        train_data=train_records,
        test_data=test_records,
        backend_names=cfg.backends,
        rrf_k=cfg.rrf_k,
        rrf_depth=cfg.rrf_depth,
        temperature=cfg.temperature,
        use_bert=cfg.use_bert,
        include_popularity=cfg.include_popularity,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        unfreeze_layers=cfg.unfreeze_layers,
        bert_lr=cfg.bert_lr,
        patience=cfg.patience,
        dropout=cfg.dropout,
        use_scheduler=cfg.use_scheduler,
        seed=cfg.seed,
        weight_decay=cfg.weight_decay,
        bert_weight_decay=cfg.bert_weight_decay,
        wandb_key=os.getenv("WEIGHTS_AND_BIASES_API_KEY"),
        wandb_project=cfg.wandb_project,
        wandb_run_name=cfg.wandb_run_name or cfg.model_name,
        warmup_epochs=cfg.warmup_epochs,
        min_lr_ratio=cfg.min_lr_ratio,
        loss_top_k=cfg.loss_top_k,
        residual_scale=cfg.residual_scale,
    )

    model_path = Path("models") / f"{cfg.model_name}.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    import torch
    torch.save(result, str(model_path))
    logger.info(f"Model saved to {model_path}")

    history_path = _write_history(cfg, result, model_path)

    _print_summary(cfg, result, model_path, history_path)

    return result


def _write_history(
    cfg: FusionTrainingConfig,
    result: dict,
    model_path: Path,
) -> Path | None:
    """Write per-epoch training history to JSON."""
    if not cfg.save_history:
        return None

    history = result.get("history", {})
    epochs_completed = len(history.get("train_loss", []))
    history_path = Path(cfg.save_history)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_name": cfg.model_name,
        "model_path": str(model_path),
        "config": {
            "collection_name": cfg.collection_name,
            "dataset_dir": cfg.dataset_dir,
            "backends": cfg.backends,
            "exclude_datasets": cfg.exclude_datasets,
            "rrf_depth": cfg.rrf_depth,
            "rrf_k": cfg.rrf_k,
            "temperature": cfg.temperature,
            "loss_top_k": cfg.loss_top_k,
            "use_bert": cfg.use_bert,
            "include_popularity": cfg.include_popularity,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "unfreeze_layers": cfg.unfreeze_layers,
            "bert_lr": cfg.bert_lr,
            "patience": cfg.patience,
            "dropout": cfg.dropout,
            "use_scheduler": cfg.use_scheduler,
            "seed": cfg.seed,
            "weight_decay": cfg.weight_decay,
            "bert_weight_decay": cfg.bert_weight_decay,
            "warmup_epochs": cfg.warmup_epochs,
            "min_lr_ratio": cfg.min_lr_ratio,
        },
        "epoch": list(range(epochs_completed)),
        "train_loss": history.get("train_loss", []),
        "train_mrr": history.get("train_mrr", []),
        "test_loss": history.get("test_loss", []),
        "test_mrr": history.get("test_mrr", []),
        "best_epoch": result.get("best_epoch"),
        "best_test_mrr": result.get("best_test_mrr"),
        "stopped_early": result.get("stopped_early"),
    }
    history_path.write_text(json.dumps(payload, indent=2) + "\n")
    logger.info(f"History saved to {history_path}")
    return history_path


def _print_summary(
    cfg: FusionTrainingConfig,
    result: dict,
    model_path: Path,
    history_path: Path | None,
) -> None:
    """Print a training summary."""
    logger.info("\n" + "=" * 60)
    logger.info("Fusion Training Complete!")
    logger.info("=" * 60)
    if result.get("best_epoch") is not None:
        logger.info(f"Best epoch:       {result['best_epoch'] + 1}")
        logger.info(f"Best test MRR:    {result['best_test_mrr']:.4f}")
        if result.get("stopped_early"):
            logger.info(f"Stopped early:    True (patience={cfg.patience})")
    hist = result.get("history", {})
    if hist.get("train_loss"):
        logger.info(f"Final train loss: {hist['train_loss'][-1]:.4f}")
    if hist.get("train_mrr"):
        logger.info(f"Final train MRR:  {hist['train_mrr'][-1]:.4f}")
    if hist.get("test_loss"):
        logger.info(f"Final test loss:  {hist['test_loss'][-1]:.4f}")
    if hist.get("test_mrr"):
        logger.info(f"Final test MRR:   {hist['test_mrr'][-1]:.4f}")
    logger.info(f"Model path:       {model_path}")
    if history_path:
        logger.info(f"History path:     {history_path}")
    logger.info("=" * 60 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Train query-conditioned backend transformations for fixed RRF",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--collection", default="wiki_full_bil")
    parser.add_argument("--dataset-dir", default="all_qa_8k")
    parser.add_argument("--model-name", default="fusion_v1")
    parser.add_argument("--backends", nargs="+", default=["bm25_plus", "ivfpq_high"])
    parser.add_argument("--exclude-datasets", nargs="+", default=[])
    parser.add_argument("--rrf-depth", type=int, default=60)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="Soft-rank temperature tau (default: 0.5)")
    parser.add_argument("--residual-scale", type=float, default=1.0,
                        help="Max magnitude of learned position residuals (default: 1.0)")
    parser.add_argument("--loss-top-k", type=int, default=20,
                        help="Legacy no-op retained for CLI compatibility")
    parser.add_argument("--no-bert", action="store_true")
    parser.add_argument("--no-popularity", action="store_true")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--unfreeze-layers", type=int, default=0)
    parser.add_argument("--bert-lr", type=float, default=2e-5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--no-scheduler", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--bert-weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--no-early-stop", action="store_true")
    parser.add_argument("--wandb-project", default="popularity-bias-fusion")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--save-history", default=None)

    args = parser.parse_args()

    cfg = FusionTrainingConfig(
        collection_name=args.collection,
        dataset_dir=args.dataset_dir,
        model_name=args.model_name,
        backends=args.backends,
        exclude_datasets=args.exclude_datasets,
        rrf_depth=args.rrf_depth,
        rrf_k=args.rrf_k,
        temperature=args.temperature,
        residual_scale=args.residual_scale,
        loss_top_k=args.loss_top_k,
        use_bert=not args.no_bert,
        include_popularity=not args.no_popularity,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        unfreeze_layers=args.unfreeze_layers,
        bert_lr=args.bert_lr,
        patience=0 if args.no_early_stop else args.patience,
        dropout=args.dropout,
        use_scheduler=not args.no_scheduler,
        seed=args.seed,
        weight_decay=args.weight_decay,
        bert_weight_decay=args.bert_weight_decay,
        warmup_epochs=args.warmup_epochs,
        min_lr_ratio=args.min_lr_ratio,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        save_history=args.save_history,
    )

    train_fusion(cfg)


if __name__ == "__main__":
    main()
