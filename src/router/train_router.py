"""Train neural router to select between BM25 and FAISS retrieval backends.

Usage:
    python -m src.router.train_router \\
        --collection wiki_full_bil \\
        --dataset-dir all_qa_8k \\
        --model-name router_v1 \\
        --backends bm25_plus ivfpq_high \\
        --exclude-datasets fever \\
        --epochs 80

This script:
1. Loads the analysis dataset from the specified collection
2. Prepares training/test splits with question-level labels
3. Trains a BERT-based router on Modal GPU
4. Saves trained model weights to models/{model_name}.pt
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import DATA_DIR
from src.router.router_service import RouterService
from src.router.metadata import record_training
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

class RouterTrainingConfig:
    """Configuration for router training.
    
    Attributes:
        collection_name: Data folder name under DATA_DIR
        dataset_dir: Subdirectory containing the analysis dataset
        model_name: Name for saved model file (models/{model_name}.pt)
        backends_to_train: List of backend names to train router on (e.g., ['bm25_plus', 'ivfpq_high'])
        backends_to_eval: List of backend names to include in evaluation dataset
        exclude_datasets: List of dataset names to exclude (e.g., ['fever'])
        llm: LLM name to filter by (e.g., 'neo')
        
        # Label mode selection
        label_mode: 'answer' or 'retrieval'
            - 'answer': Train on answer generation performance (requires generation stage)
            - 'retrieval': Train on retrieval metrics (MRR/Recall, no generation needed)
        
        # Retrieval mode options (only used if label_mode='retrieval')
        retrieval_metric: 'mrr' or 'recall'
        retrieval_k: Top-k cutoff for retrieval metric (e.g., 20)
        
        # Data filtering
        keep_ties: If True, do NOT filter tie questions (where all backends
            have identical metric values). Default is False — ties are always
            removed because they provide no discriminative signal and inflate
            the first backend's class share via argmax.

        # Training hyperparameters
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate for classifier head
        unfreeze_layers: Number of BERT layers to unfreeze (0=all frozen)
        bert_lr: Learning rate for unfrozen BERT layers
    """
    
    collection_name: str = "wiki_full_bil"
    dataset_dir: str = "all_qa_8k"
    model_name: str = "router_v1"
    
    backends_to_train: list[str] = None
    backends_to_eval: list[str] = None
    exclude_datasets: list[str] = None
    llm: str = "neo"
    
    # Label mode
    label_mode: str = "answer"  # 'answer' or 'retrieval'
    retrieval_metric: str = "mrr"  # 'mrr' or 'recall'
    retrieval_k: int = 20
    
    # Training hyperparameters
    epochs: int = 80
    batch_size: int = 32
    lr: float = 0.001
    unfreeze_layers: int = 0
    bert_lr: float = 2e-5
    include_popularity: bool = True
    keep_ties: bool = False
    patience: int = 10
    dropout: float | None = None
    use_scheduler: bool = True
    seed: int = 42
    save_history: str | None = None
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Set defaults
        if self.backends_to_train is None:
            self.backends_to_train = ['bm25_plus', 'ivfpq_high']
        if self.backends_to_eval is None:
            self.backends_to_eval = self.backends_to_train + ['zero_shot', 'faiss_hybrid']
        if self.exclude_datasets is None:
            self.exclude_datasets = []


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════

def create_labels(row, backends: list[str]) -> list[float]:
    """Create label vector from backend performance.
    
    For each question, create label vector [bm25_success, faiss_success].
    If both failed, use uniform [0.5, 0.5] (router should guess randomly).
    """
    labels = [float(row[b]) if pd.notna(row[b]) else 0.0 for b in backends]
    return labels if sum(labels) > 0 else [1.0 / len(backends)] * len(backends)


def prepare_training_data(cfg: RouterTrainingConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and prepare training data.
    
    Returns:
        (train_questions, test_questions) DataFrames with columns:
            - question_id, question_text, popularity, dataset, split
            - {backend_name} columns with metric values
            - router_labels: list of label probabilities for training
    """
    logger.info("Loading analysis dataset...")
    handler = AnalysisDatasetHandler(
        collection_name=cfg.collection_name,
        dataset_dir=cfg.dataset_dir,
    )
    
    # Different data preparation based on label mode
    if cfg.label_mode == "retrieval":
        logger.info(f"Label mode: RETRIEVAL ({cfg.retrieval_metric}@{cfg.retrieval_k})")
        return prepare_retrieval_mode_data(handler, cfg)
    elif cfg.label_mode == "answer":
        logger.info("Label mode: ANSWER (generation performance)")
        return prepare_answer_mode_data(handler, cfg)
    else:
        raise ValueError(f"Unknown label_mode: {cfg.label_mode}. Use 'answer' or 'retrieval'.")


def prepare_retrieval_mode_data(
    handler: AnalysisDatasetHandler,
    cfg: RouterTrainingConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare data for retrieval metric mode (MRR/Recall).
    
    This mode computes retrieval metrics directly from retrieved_doc_ids,
    without needing any generation or answer evaluation stage.
    """
    # Compute retrieval metrics for train split
    logger.info("Computing retrieval metrics for train split...")
    train_questions = handler.compute_retrieval_metrics_for_backends(
        backends=cfg.backends_to_train,
        metric=cfg.retrieval_metric,
        k=cfg.retrieval_k,
        split='train',
    )
    
    # Filter by dataset exclusions
    if cfg.exclude_datasets:
        train_questions = train_questions[~train_questions['dataset'].isin(cfg.exclude_datasets)]
        logger.info(f"Excluded datasets {cfg.exclude_datasets}: {len(train_questions):,} train questions")
    
    # Compute for test split
    logger.info("Computing retrieval metrics for test split...")
    test_questions = handler.compute_retrieval_metrics_for_backends(
        backends=cfg.backends_to_train,
        metric=cfg.retrieval_metric,
        k=cfg.retrieval_k,
        split='test',
    )
    
    if cfg.exclude_datasets:
        test_questions = test_questions[~test_questions['dataset'].isin(cfg.exclude_datasets)]
        logger.info(f"Excluded datasets {cfg.exclude_datasets}: {len(test_questions):,} test questions")
    
    # ── Standard tie filtering ──────────────────────────────────────────────
    # Remove questions where all backends have identical metric values.
    # Ties provide no discriminative signal: argmax picks index 0 (first
    # backend) for all ties, inflating its class share and eventually causing
    # majority-class collapse. ~47% of questions are ties.
    train_questions, test_questions = _filter_tie_questions(
        train_questions, test_questions,
        backend_cols=cfg.backends_to_train,
        metric_name=f"{cfg.retrieval_metric}@{cfg.retrieval_k}",
        keep_ties=cfg.keep_ties,
    )
    
    # Create router labels from retrieval metrics
    # For retrieval metrics, we use the actual metric values as soft labels
    # Example: if BM25 MRR=0.33 and FAISS MRR=1.0, labels=[0.33, 1.0] normalized to [0.25, 0.75]
    logger.info(f"Creating labels from {cfg.retrieval_metric}@{cfg.retrieval_k} values...")
    train_questions['router_labels'] = train_questions.apply(
        lambda r: create_labels_from_metrics(r, cfg.backends_to_train),
        axis=1
    )
    test_questions['router_labels'] = test_questions.apply(
        lambda r: create_labels_from_metrics(r, cfg.backends_to_train),
        axis=1
    )
    
    logger.info(f"Train: {len(train_questions):,} | Test: {len(test_questions):,}")
    
    # Show example labels
    logger.info("\nExample labels (first 3 train questions):")
    for i in range(min(3, len(train_questions))):
        row = train_questions.iloc[i]
        backend_values = [f"{b}={row[b]:.3f}" for b in cfg.backends_to_train]
        logger.info(f"  {' | '.join(backend_values)} → labels={row['router_labels']}")
    
    return train_questions, test_questions


def prepare_answer_mode_data(
    handler: AnalysisDatasetHandler,
    cfg: RouterTrainingConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare data for answer performance mode.
    
    This is the original mode that uses binary answer generation performance.
    """
    df_raw = handler.load_analysis_dataset()
    logger.info(f"Loaded {len(df_raw):,} rows")
    
    # Filter by LLM and exclude datasets
    if cfg.llm:
        df_raw = df_raw[df_raw['llm'] == cfg.llm]
        logger.info(f"Filtered to llm={cfg.llm}: {len(df_raw):,} rows")
    
    if cfg.exclude_datasets:
        df_raw = df_raw[~df_raw['dataset'].isin(cfg.exclude_datasets)]
        logger.info(f"Excluded datasets {cfg.exclude_datasets}: {len(df_raw):,} rows")
    
    # Filter to backends we care about
    df_filtered = df_raw[df_raw['backend'].isin(cfg.backends_to_eval)]
    logger.info(f"Filtered to backends {cfg.backends_to_eval}: {len(df_filtered):,} rows")
    
    # Pivot to question-centric format
    logger.info("Pivoting to question-centric format...")
    questions = df_filtered.pivot_table(
        index='question_id',
        columns='backend',
        values='performance',
        aggfunc='first'
    ).reset_index()
    
    # Merge with metadata
    meta = df_filtered.groupby('question_id').agg({
        'question_text': 'first',
        'popularity': 'first',
        'dataset': 'first',
        'split': 'first'
    }).reset_index()
    
    questions = meta.merge(questions, on='question_id')
    logger.info(f"Prepared {len(questions):,} questions")
    
    # Create router labels (binary mode)
    logger.info(f"Creating labels from backends: {cfg.backends_to_train}")
    questions['router_labels'] = questions.apply(
        lambda r: create_labels_binary(r, cfg.backends_to_train),
        axis=1
    )
    
    # Split train/test
    train_questions = questions[questions['split'] == 'train'].reset_index(drop=True)
    test_questions = questions[questions['split'] == 'test'].reset_index(drop=True)
    
    # ── Standard tie filtering ──────────────────────────────────────────────
    train_questions, test_questions = _filter_tie_questions(
        train_questions, test_questions,
        backend_cols=cfg.backends_to_train,
        metric_name="answer performance",
        keep_ties=cfg.keep_ties,
    )
    
    logger.info(f"Train: {len(train_questions):,} | Test: {len(test_questions):,}")
    
    return train_questions, test_questions


def create_labels_from_metrics(row, backends: list[str]) -> list[float]:
    """Create label vector from retrieval metric values (MRR/Recall).
    
    Uses actual metric values as soft labels, normalized to sum to 1.
    
    Example:
        If BM25 MRR=0.33 and FAISS MRR=1.0
        Raw: [0.33, 1.0]
        Normalized: [0.33/(0.33+1.0), 1.0/(0.33+1.0)] = [0.248, 0.752]
    
    If both are 0, use uniform distribution [0.5, 0.5].
    """
    values = [float(row[b]) if pd.notna(row[b]) else 0.0 for b in backends]
    total = sum(values)
    
    if total == 0:
        # Both failed → uniform distribution
        return [1.0 / len(backends)] * len(backends)
    
    # Normalize to sum to 1
    return [v / total for v in values]


def create_labels_binary(row, backends: list[str]) -> list[float]:
    """Create label vector from binary performance (0/1).
    
    Original mode: if backend succeeded, label=1.0, else 0.0.
    If both failed, use uniform [0.5, 0.5].
    """
    labels = [float(row[b]) if pd.notna(row[b]) else 0.0 for b in backends]
    return labels if sum(labels) > 0 else [1.0 / len(backends)] * len(backends)


def _filter_tie_questions(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    backend_cols: list[str],
    metric_name: str,
    *,
    keep_ties: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove questions where all backends have identical metric values.

    Tie questions (both backends achieve the same score) provide no
    discriminative signal for the router. When argmax is applied to create
    hard labels, ties collapse to class 0 (first backend), inflating its
    prevalence and eventually causing majority-class collapse.

    Args:
        train_df: Training split with per-backend metric columns.
        test_df: Test split with per-backend metric columns.
        backend_cols: Column names holding the per-backend metric values.
        metric_name: Human-readable metric name for logging.
        keep_ties: If True, skip filtering entirely (for experimentation).

    Returns:
        Filtered ``(train_df, test_df)`` with tie rows removed and index reset.
    """
    if keep_ties:
        return train_df, test_df

    before_train, before_test = len(train_df), len(test_df)

    train_mask = train_df[backend_cols].nunique(axis=1) > 1
    test_mask = test_df[backend_cols].nunique(axis=1) > 1

    train_df = train_df[train_mask].reset_index(drop=True)
    test_df = test_df[test_mask].reset_index(drop=True)

    logger.info(
        f"Filtered tie questions (identical {metric_name} across backends): "
        f"train {before_train:,} → {len(train_df):,} "
        f"({before_train - len(train_df):,} removed), "
        f"test {before_test:,} → {len(test_df):,} "
        f"({before_test - len(test_df):,} removed)"
    )
    return train_df, test_df


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════

def train_router(cfg: RouterTrainingConfig) -> dict:
    """Train router model on Modal GPU.
    
    Returns:
        Training result dict containing:
            - classifier_state: Trained model weights
            - scaler_mean, scaler_scale: Normalization parameters
            - model_config: Architecture configuration
            - history: Training metrics by epoch
    """
    # Prepare data
    train_questions, test_questions = prepare_training_data(cfg)
    
    # Initialize Modal service
    logger.info("Connecting to Modal GPU service...")
    service = RouterService()
    
    # Train
    logger.info(
        f"Training router (epochs={cfg.epochs}, batch_size={cfg.batch_size}, "
        f"patience={cfg.patience}, dropout={cfg.dropout}, "
        f"scheduler={cfg.use_scheduler}, seed={cfg.seed})..."
    )
    logger.info(f"  Classifier LR: {cfg.lr}")
    logger.info(f"  Include popularity: {cfg.include_popularity}")
    logger.info(f"  Unfrozen BERT layers: {cfg.unfreeze_layers}")
    if cfg.unfreeze_layers > 0:
        logger.info(f"  BERT LR: {cfg.bert_lr}")
    
    result = service.train(
        train_questions=train_questions['question_text'].tolist(),
        train_popularity=train_questions['popularity'].tolist(),
        train_labels=train_questions['router_labels'].tolist(),
        test_questions=test_questions['question_text'].tolist(),
        test_popularity=test_questions['popularity'].tolist(),
        test_labels=test_questions['router_labels'].tolist(),
        num_classes=len(cfg.backends_to_train),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        unfreeze_layers=cfg.unfreeze_layers,
        bert_lr=cfg.bert_lr,
        include_popularity=cfg.include_popularity,
        patience=cfg.patience,
        dropout=cfg.dropout,
        use_scheduler=cfg.use_scheduler,
        seed=cfg.seed,
    )
    
    # Save model
    model_path = Path("models") / f"{cfg.model_name}.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    RouterService.save_model(result, str(model_path))
    logger.info(f"Model saved to {model_path}")

    history_path = _write_history(cfg, result, model_path)

    # Record metadata (atomic, preserves existing notes if this name is a re-train)
    try:
        record_training(cfg, result, model_path, history_file=history_path)
    except Exception as e:
        logger.warning(f"Could not write metadata.json: {e}")
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("Training Complete!")
    logger.info("="*60)
    if result.get('best_epoch') is not None:
        logger.info(f"Best epoch:       {result['best_epoch'] + 1} / {len(result['history']['train_loss'])}")
        logger.info(f"Best test loss:   {result['best_test_loss']:.4f}")
        logger.info(f"Best test acc:    {result['best_test_acc']:.2%}")
        if result.get('stopped_early'):
            logger.info(f"Stopped early:    True (patience={cfg.patience})")
    logger.info(f"Final train loss: {result['history']['train_loss'][-1]:.4f}")
    logger.info(f"Final train acc:  {result['history']['train_acc'][-1]:.2%}")
    logger.info(f"Final test loss:  {result['history']['test_loss'][-1]:.4f}")
    logger.info(f"Final test acc:   {result['history']['test_acc'][-1]:.2%}")
    logger.info(f"Model path:       {model_path}")
    if history_path is not None:
        logger.info(f"History path:     {history_path}")
    logger.info("="*60 + "\n")
    
    return result


def _write_history(
    cfg: RouterTrainingConfig,
    result: dict,
    model_path: Path,
) -> Path | None:
    """Write per-epoch training history to JSON when requested.

    Args:
        cfg: Training configuration for the run.
        result: Result dictionary returned by ``RouterService.train``.
        model_path: Path to the model checkpoint.

    Returns:
        Path to the written history file, or ``None`` if history logging was not
        requested.
    """
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
            "backends_to_train": cfg.backends_to_train,
            "exclude_datasets": cfg.exclude_datasets,
            "llm": cfg.llm,
            "label_mode": cfg.label_mode,
            "retrieval_metric": cfg.retrieval_metric,
            "retrieval_k": cfg.retrieval_k,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "unfreeze_layers": cfg.unfreeze_layers,
            "bert_lr": cfg.bert_lr,
            "include_popularity": cfg.include_popularity,
            "keep_ties": cfg.keep_ties,
            "patience": cfg.patience,
            "dropout": cfg.dropout,
            "use_scheduler": cfg.use_scheduler,
            "seed": cfg.seed,
        },
        "epoch": list(range(epochs_completed)),
        "train_loss": history.get("train_loss", []),
        "train_acc": history.get("train_acc", []),
        "test_loss": history.get("test_loss", []),
        "test_acc": history.get("test_acc", []),
        "best_epoch": result.get("best_epoch"),
        "best_test_loss": result.get("best_test_loss"),
        "best_test_acc": result.get("best_test_acc"),
        "stopped_early": result.get("stopped_early"),
    }
    history_path.write_text(json.dumps(payload, indent=2) + "\n")
    logger.info(f"History saved to {history_path}")
    return history_path


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Train neural router for RAG backend selection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Data configuration
    parser.add_argument("--collection", default="wiki_full_bil",
                        help="Collection name (folder under DATA_DIR)")
    parser.add_argument("--dataset-dir", default="all_qa_8k",
                        help="Dataset directory within collection")
    parser.add_argument("--model-name", default="router_v1",
                        help="Model name for saving (models/{name}.pt)")
    
    # Backend configuration
    parser.add_argument("--backends", nargs="+", default=["bm25_plus", "ivfpq_high"],
                        help="Backends to train router on")
    parser.add_argument("--eval-backends", nargs="+", default=None,
                        help="Additional backends to include in eval dataset (only for answer mode)")
    parser.add_argument("--exclude-datasets", nargs="+", default=[],
                        help="Dataset names to exclude (e.g., fever)")
    parser.add_argument("--llm", default="neo",
                        help="LLM to filter by (only for answer mode)")
    
    # Label mode selection
    parser.add_argument("--label-mode", default="answer", choices=["answer", "retrieval"],
                        help="Training mode: 'answer' (needs generation) or 'retrieval' (MRR/Recall only)")
    parser.add_argument("--retrieval-metric", default="mrr", choices=["mrr", "recall"],
                        help="Retrieval metric (only for retrieval mode)")
    parser.add_argument("--retrieval-k", type=int, default=20,
                        help="Top-k cutoff for retrieval metric (only for retrieval mode)")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=80,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Classifier learning rate")
    parser.add_argument("--unfreeze-layers", type=int, default=0,
                        help="Number of BERT layers to unfreeze (0=all frozen)")
    parser.add_argument("--bert-lr", type=float, default=2e-5,
                        help="BERT learning rate (only if layers unfrozen)")
    parser.add_argument("--no-popularity", action="store_true",
                        help="Exclude popularity feature (BERT-only input)")
    parser.add_argument("--keep-ties", action="store_true",
                        help="Do NOT filter tie questions (where all backends "
                             "have identical metric values). By default ties "
                             "are always filtered.")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience — stop after this many epochs "
                             "without test_loss improvement (0 disables).")
    parser.add_argument("--dropout", type=float, default=None,
                        help="Classifier dropout override. If omitted, uses router defaults.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible training")
    parser.add_argument("--no-scheduler", action="store_true",
                        help="Disable cosine LR scheduler and use constant learning rate")
    parser.add_argument("--no-early-stop", action="store_true",
                        help="Disable early stopping and run all epochs")
    parser.add_argument("--save-history", default=None,
                        help="Path to write per-epoch train/test metrics JSON")
    
    args = parser.parse_args()
    
    # Build config
    cfg = RouterTrainingConfig(
        collection_name=args.collection,
        dataset_dir=args.dataset_dir,
        model_name=args.model_name,
        backends_to_train=args.backends,
        backends_to_eval=args.eval_backends if args.eval_backends else args.backends + ["zero_shot", "faiss_hybrid"],
        exclude_datasets=args.exclude_datasets,
        llm=args.llm,
        label_mode=args.label_mode,
        retrieval_metric=args.retrieval_metric,
        retrieval_k=args.retrieval_k,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        unfreeze_layers=args.unfreeze_layers,
        bert_lr=args.bert_lr,
        include_popularity=not args.no_popularity,
        keep_ties=args.keep_ties,
        patience=0 if args.no_early_stop else args.patience,
        dropout=args.dropout,
        use_scheduler=not args.no_scheduler,
        seed=args.seed,
        save_history=args.save_history,
    )
    
    # Train
    train_router(cfg)


if __name__ == "__main__":
    main()
