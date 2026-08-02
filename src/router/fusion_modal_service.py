"""Modal GPU service for learned backend transformations with fixed RRF.

The model learns query-conditioned score transformations for BM25 and FAISS
*within each backend*. Final fusion remains plain unweighted RRF.
"""
from __future__ import annotations

import math
import random
from typing import Dict, List

import logging
import modal

logger = logging.getLogger(__name__)

APP_NAME = "FusionTrainingService"
GPU_CONFIG = "H100"
MAX_CONTAINERS = 2
CONTAINER_TIMEOUT = 60
FUNCTION_TIMEOUT = 7200
MAX_RETRIES = 2


def download_models() -> None:
    from transformers import AutoModel, AutoTokenizer

    AutoTokenizer.from_pretrained("bert-base-uncased")
    AutoModel.from_pretrained("bert-base-uncased")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "pandas",
        "pyarrow",
        "scikit-learn",
        "tqdm",
        "wandb",
    )
    .run_function(download_models)
)

app = modal.App(APP_NAME)


class _RankTransformPredictor:
    """MLP that predicts transformed rank-position scores per backend."""

    def __init__(
        self,
        input_dim: int = 769,
        rrf_depth: int = 60,
        hidden_dim1: int = 32,
        hidden_dim2: int = 16,
        dropout: float = 0.3,
        residual_scale: float = 1.0,
    ):
        import torch
        import torch.nn as nn

        self.rrf_depth = rrf_depth
        self.residual_scale = residual_scale
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim2, 2 * rrf_depth),
        )
        base = torch.linspace(1.0, 0.0, steps=rrf_depth, dtype=torch.float32)
        self.base_position_scores = base.view(1, 1, rrf_depth)

    def __call__(self, x):
        import torch

        residual = self.network(x).view(-1, 2, self.rrf_depth)
        residual = self.residual_scale * torch.tanh(residual)
        base = self.base_position_scores.to(x.device)
        return base + residual

    def to(self, device):
        self.network = self.network.to(device)
        self.base_position_scores = self.base_position_scores.to(device)
        return self

    def train(self):
        self.network.train()

    def eval(self):
        self.network.eval()

    def parameters(self):
        return self.network.parameters()

    def state_dict(self):
        return self.network.state_dict()

    def load_state_dict(self, state_dict):
        self.network.load_state_dict(state_dict)


def _build_candidate_matrices(bm25_doc_ids, faiss_doc_ids, ground_truth_ids, device):
    import torch

    batch_size = len(ground_truth_ids)
    max_union = max(
        len(set(bm25_doc_ids[i]) | set(faiss_doc_ids[i])) for i in range(batch_size)
    ) if batch_size else 1

    bm25_pos_idx = torch.zeros(batch_size, max_union, dtype=torch.long, device=device)
    faiss_pos_idx = torch.zeros(batch_size, max_union, dtype=torch.long, device=device)
    bm25_present = torch.zeros(batch_size, max_union, dtype=torch.bool, device=device)
    faiss_present = torch.zeros(batch_size, max_union, dtype=torch.bool, device=device)
    gold_idx = torch.full((batch_size,), -1, dtype=torch.long, device=device)

    for i in range(batch_size):
        union_docs = sorted(set(bm25_doc_ids[i]) | set(faiss_doc_ids[i]))
        bm25_lookup = {doc_id: pos for pos, doc_id in enumerate(bm25_doc_ids[i])}
        faiss_lookup = {doc_id: pos for pos, doc_id in enumerate(faiss_doc_ids[i])}

        for j, doc_id in enumerate(union_docs):
            if doc_id in bm25_lookup:
                bm25_pos_idx[i, j] = bm25_lookup[doc_id]
                bm25_present[i, j] = True
            if doc_id in faiss_lookup:
                faiss_pos_idx[i, j] = faiss_lookup[doc_id]
                faiss_present[i, j] = True
            if doc_id == ground_truth_ids[i]:
                gold_idx[i] = j

    return bm25_pos_idx, faiss_pos_idx, bm25_present, faiss_present, gold_idx


def _soft_backend_ranks(doc_scores, present_mask, tau):
    import torch

    num_docs = doc_scores.shape[1]
    s_i = doc_scores.unsqueeze(2)
    s_j = doc_scores.unsqueeze(1)
    pairwise = torch.sigmoid((s_i - s_j) / tau)

    valid_pairs = present_mask.unsqueeze(2) & present_mask.unsqueeze(1)
    eye = torch.eye(num_docs, dtype=torch.bool, device=doc_scores.device).unsqueeze(0)
    valid_pairs = valid_pairs & ~eye
    return 1.0 + (pairwise * valid_pairs.to(doc_scores.dtype)).sum(dim=1)


def _fused_scores_from_transforms(transformed_scores, batch_data, backend_names, rrf_k, tau):
    import torch

    device = transformed_scores.device
    depth = transformed_scores.shape[2]
    bm25_key = f"{backend_names[0]}_doc_ids"
    faiss_key = f"{backend_names[1]}_doc_ids"

    bm25_doc_ids = [list(d[bm25_key])[:depth] for d in batch_data]
    faiss_doc_ids = [list(d[faiss_key])[:depth] for d in batch_data]
    ground_truth_ids = [str(d["wikipedia_id"]) for d in batch_data]

    bm25_pos_idx, faiss_pos_idx, bm25_present, faiss_present, gold_idx = _build_candidate_matrices(
        bm25_doc_ids, faiss_doc_ids, ground_truth_ids, device
    )

    bm25_position_scores = transformed_scores[:, 0, :]
    faiss_position_scores = transformed_scores[:, 1, :]

    bm25_doc_scores = bm25_position_scores.gather(1, bm25_pos_idx)
    faiss_doc_scores = faiss_position_scores.gather(1, faiss_pos_idx)

    bm25_doc_scores = bm25_doc_scores.masked_fill(~bm25_present, 0.0)
    faiss_doc_scores = faiss_doc_scores.masked_fill(~faiss_present, 0.0)

    fused_scores = bm25_doc_scores + faiss_doc_scores

    candidate_mask = bm25_present | faiss_present
    return fused_scores, gold_idx, candidate_mask


def _fusion_rrf_loss(transformed_scores, batch_data, backend_names, rrf_k, temperature, loss_top_k=20):
    import torch

    fused_scores, gold_idx, candidate_mask = _fused_scores_from_transforms(
        transformed_scores,
        batch_data,
        backend_names,
        rrf_k,
        max(temperature, 1e-8),
    )

    valid_mask = gold_idx >= 0
    if not valid_mask.any():
        return torch.tensor(0.0, device=transformed_scores.device, requires_grad=True)

    fused_scores = fused_scores[valid_mask]
    gold_idx = gold_idx[valid_mask]
    candidate_mask = candidate_mask[valid_mask]

    batch_indices = torch.arange(fused_scores.shape[0], device=transformed_scores.device)
    gold_scores = fused_scores[batch_indices, gold_idx].unsqueeze(1)
    pairwise_probs = torch.sigmoid((fused_scores - gold_scores) / max(temperature, 1e-8))

    gold_mask = torch.zeros_like(candidate_mask)
    gold_mask[batch_indices, gold_idx] = True
    competitor_mask = candidate_mask & ~gold_mask

    soft_rank = 1.0 + (pairwise_probs * competitor_mask.to(fused_scores.dtype)).sum(dim=1)
    return torch.log(soft_rank).mean()


def _compute_mrr(transformed_scores, batch_data, backend_names, rrf_k=60):
    reciprocal_ranks = []
    found_count = 0
    bm25_key = f"{backend_names[0]}_doc_ids"
    faiss_key = f"{backend_names[1]}_doc_ids"

    for i, row in enumerate(batch_data):
        gold_id = str(row["wikipedia_id"])
        bm25_docs = list(row[bm25_key])
        faiss_docs = list(row[faiss_key])

        bm25_scores = transformed_scores[i, 0, : len(bm25_docs)].detach().cpu().tolist()
        faiss_scores = transformed_scores[i, 1, : len(faiss_docs)].detach().cpu().tolist()

        bm25_ranked = [doc for _, doc in sorted(zip(bm25_scores, bm25_docs), reverse=True)]
        faiss_ranked = [doc for _, doc in sorted(zip(faiss_scores, faiss_docs), reverse=True)]

        fused = {}
        for rank_0, doc_id in enumerate(bm25_ranked):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (rrf_k + rank_0 + 1)
        for rank_0, doc_id in enumerate(faiss_ranked):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (rrf_k + rank_0 + 1)

        rr = 0.0
        for rank, (doc_id, _) in enumerate(sorted(fused.items(), key=lambda x: x[1], reverse=True), start=1):
            if doc_id == gold_id:
                rr = 1.0 / rank
                found_count += 1
                break
        reciprocal_ranks.append(rr)

    n = len(reciprocal_ranks)
    return {
        "mrr": sum(reciprocal_ranks) / n if n > 0 else 0.0,
        "recall": found_count / n if n > 0 else 0.0,
        "count": n,
    }


@app.cls(
    image=image,
    gpu=GPU_CONFIG,
    timeout=FUNCTION_TIMEOUT,
    max_containers=MAX_CONTAINERS,
    scaledown_window=CONTAINER_TIMEOUT,
    retries=MAX_RETRIES,
)
class FusionModel:
    @modal.enter()
    def enter(self):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.bert = AutoModel.from_pretrained("bert-base-uncased").to(self.device)
        for param in self.bert.parameters():
            param.requires_grad = False

    @modal.method()
    def train(
        self,
        train_data: List[dict],
        test_data: List[dict],
        backend_names: List[str] = None,
        rrf_k: int = 60,
        rrf_depth: int = 60,
        temperature: float = 0.02,
        use_bert: bool = True,
        include_popularity: bool = True,
        epochs: int = 80,
        batch_size: int = 32,
        lr: float = 0.001,
        unfreeze_layers: int = 0,
        bert_lr: float = 2e-5,
        patience: int = 10,
        dropout: float | None = None,
        use_scheduler: bool = True,
        seed: int = 42,
        weight_decay: float = 1e-3,
        bert_weight_decay: float = 1e-2,
        wandb_key: str | None = None,
        wandb_project: str = "popularity-bias-fusion",
        wandb_run_name: str | None = None,
        warmup_epochs: int = 0,
        min_lr_ratio: float = 0.1,
        loss_top_k: int = 20,
        residual_scale: float = 1.0,
    ) -> Dict:
        import numpy as np
        import torch
        from sklearn.preprocessing import StandardScaler
        from torch.utils.data import DataLoader, TensorDataset
        from tqdm import tqdm

        backend_names = backend_names or ["bm25_plus", "ivfpq_high"]

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if unfreeze_layers > 0:
            total_layers = len(self.bert.encoder.layer)
            unfreeze_layers = min(unfreeze_layers, total_layers)
            for layer in self.bert.encoder.layer[-unfreeze_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        train_popularity = [d["popularity"] for d in train_data]
        test_popularity = [d["popularity"] for d in test_data]

        if use_bert:
            train_questions = [d["question_text"] for d in train_data]
            test_questions = [d["question_text"] for d in test_data]
            train_tokens = self.tokenizer(train_questions, padding=True, truncation=True, max_length=128, return_tensors="pt")
            test_tokens = self.tokenizer(test_questions, padding=True, truncation=True, max_length=128, return_tensors="pt")

        scaler = StandardScaler()
        train_pop_tensor = torch.tensor(scaler.fit_transform([[p] for p in train_popularity]), dtype=torch.float32)
        test_pop_tensor = torch.tensor(scaler.transform([[p] for p in test_popularity]), dtype=torch.float32)

        input_dim = (768 if use_bert else 0) + (1 if include_popularity else 0)
        if input_dim == 0:
            input_dim = 1
        predictor_dropout = dropout if dropout is not None else (0.5 if unfreeze_layers > 0 else 0.3)
        predictor = _RankTransformPredictor(
            input_dim=input_dim,
            rrf_depth=rrf_depth,
            hidden_dim1=32,
            hidden_dim2=16,
            dropout=predictor_dropout,
            residual_scale=residual_scale,
        ).to(self.device)

        optimizer_params = [{"params": predictor.parameters(), "lr": lr, "weight_decay": weight_decay}]
        if unfreeze_layers > 0:
            bert_params = [p for p in self.bert.parameters() if p.requires_grad]
            if bert_params:
                optimizer_params.insert(0, {"params": bert_params, "lr": bert_lr, "weight_decay": bert_weight_decay})
        opt = torch.optim.AdamW(optimizer_params)

        if use_scheduler:
            eta_min = lr * min_lr_ratio
            if warmup_epochs > 0:
                cosine_epochs = max(1, epochs - warmup_epochs)

                def _warmup_cosine(epoch: int) -> float:
                    if epoch < warmup_epochs:
                        return (epoch + 1) / warmup_epochs
                    progress = (epoch - warmup_epochs) / cosine_epochs
                    return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

                scheduler = torch.optim.lr_scheduler.LambdaLR(opt, _warmup_cosine)
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=eta_min)
        else:
            scheduler = None

        if use_bert:
            train_dataset = TensorDataset(train_tokens["input_ids"], train_tokens["attention_mask"], train_pop_tensor, torch.arange(len(train_data)))
            test_dataset = TensorDataset(test_tokens["input_ids"], test_tokens["attention_mask"], test_pop_tensor, torch.arange(len(test_data)))
        else:
            train_dataset = TensorDataset(train_pop_tensor, torch.arange(len(train_data)))
            test_dataset = TensorDataset(test_pop_tensor, torch.arange(len(test_data)))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        wandb_run = None
        if wandb_key:
            try:
                import wandb

                wandb.login(key=wandb_key)
                wandb_run = wandb.init(
                    project=wandb_project,
                    name=wandb_run_name,
                    config={
                        "model": "backend_rank_transformer",
                        "rrf_k": rrf_k,
                        "rrf_depth": rrf_depth,
                        "temperature": temperature,
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "lr": lr,
                        "use_bert": use_bert,
                        "include_popularity": include_popularity,
                    },
                )
            except Exception as e:
                print(f"Warning: W&B init failed ({e}), continuing without logging.")

        history = {"train_loss": [], "train_mrr": [], "test_loss": [], "test_mrr": []}
        best_test_mrr = -1.0
        best_epoch = 0
        best_predictor_state = None
        epochs_without_improve = 0
        min_delta = 1e-4
        stopped_early = False

        pbar = tqdm(range(epochs), desc="Training")
        for epoch in pbar:
            predictor.train()
            if unfreeze_layers > 0:
                self.bert.train()

            epoch_loss = 0.0
            epoch_mrr_sum = 0.0
            epoch_mrr_count = 0

            for batch in train_loader:
                if use_bert:
                    input_ids, attention_mask, pop, idx_tensor = batch
                    input_ids = input_ids.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    pop = pop.to(self.device)
                else:
                    pop, idx_tensor = batch
                    pop = pop.to(self.device)
                batch_data = [train_data[i] for i in idx_tensor.tolist()]

                if use_bert and unfreeze_layers > 0:
                    outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                    cls_embedding = outputs.last_hidden_state[:, 0, :]
                elif use_bert:
                    with torch.no_grad():
                        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                    cls_embedding = outputs.last_hidden_state[:, 0, :]
                else:
                    cls_embedding = None

                if use_bert and include_popularity:
                    combined = torch.cat([cls_embedding, pop], dim=1)
                elif use_bert:
                    combined = cls_embedding
                elif include_popularity:
                    combined = pop
                else:
                    combined = torch.ones((pop.shape[0], 1), dtype=torch.float32, device=self.device)
                transformed_scores = predictor(combined)

                loss = _fusion_rrf_loss(transformed_scores, batch_data, backend_names, rrf_k, temperature, loss_top_k)
                opt.zero_grad()
                loss.backward()
                if unfreeze_layers > 0:
                    torch.nn.utils.clip_grad_norm_([p for group in opt.param_groups for p in group["params"]], max_norm=1.0)
                opt.step()
                epoch_loss += loss.item()

                with torch.no_grad():
                    mrr_dict = _compute_mrr(transformed_scores.detach(), batch_data, backend_names, rrf_k)
                    epoch_mrr_sum += mrr_dict["mrr"] * mrr_dict["count"]
                    epoch_mrr_count += mrr_dict["count"]

            if scheduler is not None:
                scheduler.step()

            avg_train_loss = epoch_loss / max(len(train_loader), 1)
            avg_train_mrr = epoch_mrr_sum / max(epoch_mrr_count, 1)

            predictor.eval()
            self.bert.eval()
            test_loss_sum = 0.0
            test_mrr_sum = 0.0
            test_mrr_count = 0
            with torch.no_grad():
                for batch in test_loader:
                    if use_bert:
                        input_ids, attention_mask, pop, idx_tensor = batch
                        input_ids = input_ids.to(self.device)
                        attention_mask = attention_mask.to(self.device)
                        pop = pop.to(self.device)
                    else:
                        pop, idx_tensor = batch
                        pop = pop.to(self.device)
                    batch_data = [test_data[i] for i in idx_tensor.tolist()]

                    if use_bert:
                        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                        cls_embedding = outputs.last_hidden_state[:, 0, :]
                    else:
                        cls_embedding = None

                    if use_bert and include_popularity:
                        combined = torch.cat([cls_embedding, pop], dim=1)
                    elif use_bert:
                        combined = cls_embedding
                    elif include_popularity:
                        combined = pop
                    else:
                        combined = torch.ones((pop.shape[0], 1), dtype=torch.float32, device=self.device)
                    transformed_scores = predictor(combined)

                    test_loss_sum += _fusion_rrf_loss(transformed_scores, batch_data, backend_names, rrf_k, temperature, loss_top_k).item()
                    mrr_dict = _compute_mrr(transformed_scores, batch_data, backend_names, rrf_k)
                    test_mrr_sum += mrr_dict["mrr"] * mrr_dict["count"]
                    test_mrr_count += mrr_dict["count"]

            avg_test_loss = test_loss_sum / max(len(test_loader), 1)
            avg_test_mrr = test_mrr_sum / max(test_mrr_count, 1)
            history["train_loss"].append(avg_train_loss)
            history["train_mrr"].append(avg_train_mrr)
            history["test_loss"].append(avg_test_loss)
            history["test_mrr"].append(avg_test_mrr)

            pbar.set_postfix({
                "train_loss": f"{avg_train_loss:.4f}",
                "train_mrr": f"{avg_train_mrr:.4f}",
                "test_loss": f"{avg_test_loss:.4f}",
                "test_mrr": f"{avg_test_mrr:.4f}",
            })
            print(
                f"Epoch {epoch+1:03d} | Train Loss: {avg_train_loss:.4f} | "
                f"Train MRR: {avg_train_mrr:.4f} | Test Loss: {avg_test_loss:.4f} | "
                f"Test MRR: {avg_test_mrr:.4f}"
            )

            if wandb_run is not None:
                wandb_run.log({
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "train_mrr": avg_train_mrr,
                    "test_loss": avg_test_loss,
                    "test_mrr": avg_test_mrr,
                    "learning_rate": opt.param_groups[0]["lr"],
                })

            if avg_test_mrr > best_test_mrr + min_delta:
                best_test_mrr = avg_test_mrr
                best_epoch = epoch
                best_predictor_state = {k: v.detach().cpu().clone() for k, v in predictor.state_dict().items()}
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1
                if patience > 0 and epochs_without_improve >= patience:
                    print(
                        f"Early stopping at epoch {epoch+1} — no test_mrr improvement for {patience} epochs "
                        f"(best at epoch {best_epoch+1}: {best_test_mrr:.4f})"
                    )
                    stopped_early = True
                    break

        state_dict_cpu = {k: v.cpu() for k, v in predictor.state_dict().items()}

        if wandb_run is not None:
            wandb_run.log({
                "stopped_early": int(stopped_early),
                "best_epoch": best_epoch + 1 if best_predictor_state is not None else 0,
                "best_test_mrr": best_test_mrr if best_predictor_state is not None else 0.0,
                "epochs_completed": epoch + 1,
            })
            wandb_run.finish()

        return {
            "predictor_state": state_dict_cpu,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "history": history,
            "best_epoch": best_epoch,
            "best_test_mrr": best_test_mrr if best_predictor_state is not None else None,
            "stopped_early": stopped_early,
            "model_config": {
                "input_dim": input_dim,
                "hidden_dim1": 32,
                "hidden_dim2": 16,
                "dropout": predictor_dropout,
                "unfreeze_layers": unfreeze_layers,
                "use_bert": use_bert,
                "include_popularity": include_popularity,
                "use_scheduler": use_scheduler,
                "warmup_epochs": warmup_epochs,
                "min_lr_ratio": min_lr_ratio,
                "seed": seed,
                "rrf_k": rrf_k,
                "rrf_depth": rrf_depth,
                "temperature": temperature,
                "loss_top_k": loss_top_k,
                "backend_names": backend_names,
                "residual_scale": residual_scale,
            },
        }

    @modal.method()
    def predict_batch(
        self,
        questions: List[str],
        popularity: List[float],
        predictor_state: Dict,
        scaler_mean: List[float],
        scaler_scale: List[float],
        model_config: Dict,
        batch_size: int = 64,
    ) -> List[List[List[float]]]:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        use_bert = model_config.get("use_bert", True)
        include_popularity = model_config.get("include_popularity", True)
        pop_normalized = [(p - scaler_mean[0]) / scaler_scale[0] for p in popularity]
        pop_tensor = torch.tensor([[p] for p in pop_normalized], dtype=torch.float32)

        if use_bert:
            tokens = self.tokenizer(questions, padding=True, truncation=True, max_length=128, return_tensors="pt")

        predictor = _RankTransformPredictor(
            input_dim=model_config["input_dim"],
            rrf_depth=model_config["rrf_depth"],
            hidden_dim1=model_config["hidden_dim1"],
            hidden_dim2=model_config["hidden_dim2"],
            dropout=model_config["dropout"],
            residual_scale=model_config.get("residual_scale", 1.0),
        ).to(self.device)
        state_dict_gpu = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in predictor_state.items()}
        predictor.load_state_dict(state_dict_gpu)
        predictor.eval()

        if use_bert:
            dataset = TensorDataset(tokens["input_ids"], tokens["attention_mask"], pop_tensor)
        else:
            dataset = TensorDataset(pop_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_transforms = []
        with torch.no_grad():
            for batch in loader:
                if use_bert:
                    input_ids, attention_mask, pop = [b.to(self.device) for b in batch]
                    outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                    cls_embedding = outputs.last_hidden_state[:, 0, :]
                else:
                    (pop,) = [b.to(self.device) for b in batch]
                    cls_embedding = None

                if use_bert and include_popularity:
                    combined = torch.cat([cls_embedding, pop], dim=1)
                elif use_bert:
                    combined = cls_embedding
                elif include_popularity:
                    combined = pop
                else:
                    combined = torch.ones((pop.shape[0], 1), dtype=torch.float32, device=self.device)
                transforms = predictor(combined)
                all_transforms.extend(transforms.cpu().tolist())

        return all_transforms


class FusionModalService:
    """Client for the Modal-hosted fixed-RRF transformation service."""

    def __init__(self):
        ModelCls = modal.Cls.from_name(APP_NAME, "FusionModel")
        self.service = ModelCls()
        logger.info("Initialized FusionModalService on Modal")

    def train(self, **kwargs) -> dict:
        return self.service.train.remote(**kwargs)

    def predict(self, **kwargs) -> list[list[list[float]]]:
        return self.service.predict_batch.remote(**kwargs)
