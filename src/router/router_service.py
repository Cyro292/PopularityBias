"""Modal GPU Router Training & Inference Service

This module defines a Modal-based service for training and running inference
with a neural router that selects between retrieval backends.

Setup:
    modal deploy src/router/router_service.py

Usage:
    # Use via CLI script (recommended)
    python -m src.router.train_router --help
    
    # Or use directly in retrieval pipeline
    from src.rag import NeuralRouterRagService
    router = NeuralRouterRagService(model_path="models/router.pt", ...)
    
    # Low-level Modal service access
    from src.router.router_service import RouterService
    service = RouterService()
    result = service.train(...)
    predictions = service.predict(...)
"""
from __future__ import annotations

from typing import List, Dict, Tuple
import modal
import logging

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
APP_NAME = "RouterTrainingService"
GPU_CONFIG = "H100"
MAX_CONTAINERS = 4
CONTAINER_TIMEOUT = 600
FUNCTION_TIMEOUT = 3600
MAX_RETRIES = 2

def download_models():
    from transformers import AutoTokenizer, AutoModel
    AutoTokenizer.from_pretrained('bert-base-uncased')
    AutoModel.from_pretrained('bert-base-uncased')

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "pandas",
        "pyarrow",
        "scikit-learn",
        "tqdm"
    )
    .run_function(download_models)
)

app = modal.App(APP_NAME)

# ── Neural Network Definition ─────────────────────────────────────────────────
class RouterClassifier:
    """Classic PyTorch neural network for router classification.
    
    Architecture:
        With popularity:  Input 769d (768d BERT embedding + 1d normalized popularity)
        Without popularity: Input 768d (BERT embedding only)
        Hidden: input_dim → 32 → 16
        Output: num_classes
    """
    def __init__(self, input_dim: int = 769, hidden_dim1: int = 128, hidden_dim2: int = 64, num_classes: int = 2, dropout: float = 0.2, include_popularity: bool = True):
        import torch.nn as nn
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim2, num_classes)
        )
    
    def __call__(self, x):
        return self.network(x)
    
    def to(self, device):
        self.network = self.network.to(device)
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

@app.cls(
    image=image,
    gpu=GPU_CONFIG,
    timeout=FUNCTION_TIMEOUT,
    max_containers=MAX_CONTAINERS,
    scaledown_window=CONTAINER_TIMEOUT,
    retries=MAX_RETRIES
)
class RouterModel:
    @modal.enter()
    def enter(self):
        import torch
        from transformers import AutoTokenizer, AutoModel
        
        print("Loading BERT tokenizer and model...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.bert = AutoModel.from_pretrained('bert-base-uncased').to(self.device)
        
        # Initially freeze all BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False
        
        print(f"Model loaded on {self.device}!")

    @modal.method()
    def train_router(
        self,
        train_questions: List[str],
        train_popularity: List[float],
        train_labels: List[List[float]],
        test_questions: List[str],
        test_popularity: List[float],
        test_labels: List[List[float]],
        num_classes: int = 2,
        epochs: int = 160,
        batch_size: int = 32,
        lr: float = 0.001,
        unfreeze_layers: int = 0,
        bert_lr: float = 2e-5,
        include_popularity: bool = True,
        patience: int = 10,
        dropout: float | None = None,
        use_scheduler: bool = True,
        seed: int = 42,
    ) -> Dict:
        """Train router model on Modal GPU and return trained weights.
        
        Args:
            unfreeze_layers: Number of BERT layers to unfreeze from the end (0=all frozen)
            bert_lr: Learning rate for unfrozen BERT layers
            include_popularity: Whether to concatenate popularity as input feature
            patience: Early stopping patience — stop after this many epochs
                without test_loss improvement (min_delta=1e-4). 0 disables.
            dropout: Optional classifier dropout override. If absent, use the
                historical defaults (0.3 frozen BERT, 0.5 unfrozen BERT).
            use_scheduler: Whether to use cosine learning-rate scheduling.
            seed: Random seed for torch, numpy, and Python random.
        """
        import random

        import numpy as np
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
        from sklearn.preprocessing import StandardScaler
        from tqdm import tqdm

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        print(f"Training on {len(train_questions)} questions, testing on {len(test_questions)}")
        
        # Unfreeze last N BERT layers if requested
        if unfreeze_layers > 0:
            total_layers = len(self.bert.encoder.layer)
            if unfreeze_layers > total_layers:
                print(f"Warning: Requested {unfreeze_layers} layers but BERT only has {total_layers}. Unfreezing all.")
                unfreeze_layers = total_layers
            
            layers_to_unfreeze = self.bert.encoder.layer[-unfreeze_layers:]
            for layer in layers_to_unfreeze:
                for param in layer.parameters():
                    param.requires_grad = True
            
            trainable_params = sum(p.numel() for p in self.bert.parameters() if p.requires_grad)
            print(f"Unfroze last {unfreeze_layers} BERT layer(s) - {trainable_params:,} parameters")
        else:
            print("All BERT layers frozen")
        
        # Tokenize
        train_tokens = self.tokenizer(
            train_questions,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        test_tokens = self.tokenizer(
            test_questions,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        # Normalize popularity
        scaler = StandardScaler()
        train_pop = torch.tensor(
            scaler.fit_transform([[p] for p in train_popularity]),
            dtype=torch.float32
        )
        test_pop = torch.tensor(
            scaler.transform([[p] for p in test_popularity]),
            dtype=torch.float32
        )
        
        # Prepare labels - convert soft labels to hard class indices
        y_train = torch.tensor(train_labels, dtype=torch.float32)
        y_test = torch.tensor(test_labels, dtype=torch.float32)
        y_train_idx = y_train.argmax(dim=1)
        y_test_idx = y_test.argmax(dim=1)
        
        input_dim = 768 + (1 if include_popularity else 0)
        
        # Higher classifier dropout when unfreezing BERT — adds regularization
        # on top of weight decay to combat the overfitting we see with 7M+
        # extra trainable params on a ~4k-example training set. Experiments can
        # override this via CLI to inspect training dynamics under lower noise.
        classifier_dropout = dropout if dropout is not None else (0.5 if unfreeze_layers > 0 else 0.3)
        
        # Define classifier
        classifier = RouterClassifier(
            input_dim=input_dim,
            hidden_dim1=32,
            hidden_dim2=16,
            num_classes=num_classes,
            dropout=classifier_dropout,
            include_popularity=include_popularity
        ).to(self.device)
        
        # Setup training with differential learning rates AND differential
        # weight decay. Per-group wd overrides any global value, so we don't
        # pass one to AdamW. Higher wd on the unfrozen BERT params adds
        # extra regularization for the fine-tuning case.
        optimizer_params = [
            {'params': classifier.parameters(), 'lr': lr, 'weight_decay': 1e-3}
        ]
        
        # Add BERT parameters if any layers are unfrozen
        if unfreeze_layers > 0:
            bert_params = [p for p in self.bert.parameters() if p.requires_grad]
            if bert_params:
                optimizer_params.insert(
                    0, {'params': bert_params, 'lr': bert_lr, 'weight_decay': 1e-2}
                )
                print(f"Using differential LR: BERT={bert_lr} (wd=1e-2), "
                      f"Classifier={lr} (wd=1e-3), dropout={classifier_dropout}")
        
        opt = torch.optim.AdamW(optimizer_params)
        # T_max tracks the actual epoch count so the cosine schedule completes
        # a full cycle by the end of training (was hardcoded to 40, which made
        # LR climb back up for any --epochs > 40 run). Vanilla experiments can
        # disable this to keep a constant LR throughout training.
        scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
            if use_scheduler
            else None
        )
        loss_fn = nn.CrossEntropyLoss()
        
        train_dataset = TensorDataset(
            train_tokens['input_ids'],
            train_tokens['attention_mask'],
            train_pop,
            y_train_idx
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        test_dataset = TensorDataset(
            test_tokens['input_ids'],
            test_tokens['attention_mask'],
            test_pop,
            y_test_idx
        )
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        # Training loop
        history = {'train_loss': [], 'train_acc': [], 'test_acc': [], 'test_loss': []}
        
        # Best-model tracking + early stopping state
        best_test_loss: float = float('inf')
        best_epoch: int = 0
        best_classifier_state: Dict | None = None
        epochs_without_improve: int = 0
        min_delta: float = 1e-4
        stopped_early: bool = False
        
        pbar = tqdm(range(epochs), desc='Training')
        for epoch in pbar:
            # Set models to training mode
            classifier.train()
            if unfreeze_layers > 0:
                self.bert.train()
            
            epoch_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch in train_loader:
                input_ids, attention_mask, pop, labels = [b.to(self.device) for b in batch]
                
                # BERT forward pass (with or without gradients)
                if unfreeze_layers > 0:
                    outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                    cls_embedding = outputs.last_hidden_state[:, 0, :]
                else:
                    with torch.no_grad():
                        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                        cls_embedding = outputs.last_hidden_state[:, 0, :]
                
                if include_popularity:
                    combined = torch.cat([cls_embedding, pop], dim=1)
                else:
                    combined = cls_embedding
                logits = classifier(combined)
                loss = loss_fn(logits, labels)
                
                opt.zero_grad()
                loss.backward()
                
                # Gradient clipping for stability when BERT layers are unfrozen
                if unfreeze_layers > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for group in opt.param_groups for p in group['params']], 
                        max_norm=1.0
                    )
                
                opt.step()
                
                epoch_loss += loss.item()
                
                # Track train accuracy
                preds = logits.argmax(1)
                train_correct += (preds == labels).sum().item()
                train_total += len(labels)
            
            if scheduler is not None:
                scheduler.step()
            
            # ── Always evaluate (was previously gated to every 20 epochs,
            # which hid the overfitting peak). ──────────────────────────────
            classifier.eval()
            self.bert.eval()
            test_correct = 0
            test_total = 0
            test_loss = 0
            
            with torch.no_grad():
                for batch in test_loader:
                    input_ids, attention_mask, pop, labels = [b.to(self.device) for b in batch]
                    
                    outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                    cls_embedding = outputs.last_hidden_state[:, 0, :]
                    if include_popularity:
                        combined = torch.cat([cls_embedding, pop], dim=1)
                    else:
                        combined = cls_embedding
                    logits = classifier(combined)
                    
                    loss = loss_fn(logits, labels)
                    test_loss += loss.item()
                    
                    preds = logits.argmax(1)
                    test_correct += (preds == labels).sum().item()
                    test_total += len(labels)
            
            train_acc = train_correct / train_total
            test_acc = test_correct / test_total
            avg_train_loss = epoch_loss / len(train_loader)
            avg_test_loss = test_loss / len(test_loader)
            
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)
            history['test_loss'].append(avg_test_loss)
            
            pbar.set_postfix({
                'train_loss': f'{avg_train_loss:.4f}', 
                'train_acc': f'{train_acc:.2%}', 
                'test_loss': f'{avg_test_loss:.4f}',
                'test_acc': f'{test_acc:.2%}'
            })
            print(f'Epoch {epoch+1:03d} | Train Loss: {avg_train_loss:.4f} | '
                  f'Train Acc: {train_acc:.2%} | Test Loss: {avg_test_loss:.4f} | '
                  f'Test Acc: {test_acc:.2%}')
            
            # ── Best-model snapshot + early stopping ───────────────────────
            if avg_test_loss < best_test_loss - min_delta:
                best_test_loss = avg_test_loss
                best_epoch = epoch
                best_classifier_state = {
                    k: v.detach().cpu().clone()
                    for k, v in classifier.state_dict().items()
                }
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1
                if patience > 0 and epochs_without_improve >= patience:
                    print(f'Early stopping at epoch {epoch+1} — no test_loss '
                          f'improvement for {patience} epochs '
                          f'(best at epoch {best_epoch+1}: {best_test_loss:.4f})')
                    stopped_early = True
                    break
        
        # Return the latest weights from the final executed epoch. We still
        # record best_epoch / best_test_loss for diagnostics, but the saved
        # checkpoint should reflect the actual end of the requested run.
        state_dict_cpu = {k: v.cpu() for k, v in classifier.state_dict().items()}
        
        return {
            'classifier_state': state_dict_cpu,
            'scaler_mean': scaler.mean_.tolist(),
            'scaler_scale': scaler.scale_.tolist(),
            'history': history,
            'best_epoch': best_epoch,
            'best_test_loss': best_test_loss if best_classifier_state is not None else None,
            'best_test_acc': (
                history['test_acc'][best_epoch]
                if best_classifier_state is not None and best_epoch < len(history['test_acc'])
                else None
            ),
            'stopped_early': stopped_early,
            'model_config': {
                'input_dim': input_dim,
                'hidden_dim1': 32,
                'hidden_dim2': 16,
                'num_classes': num_classes,
                'dropout': classifier_dropout,
                'unfreeze_layers': unfreeze_layers,
                'include_popularity': include_popularity,
                'use_scheduler': use_scheduler,
                'seed': seed,
            }
        }

    @modal.method()
    def predict_batch(
        self,
        questions: List[str],
        popularity: List[float],
        classifier_state: Dict,
        scaler_mean: List[float],
        scaler_scale: List[float],
        model_config: Dict,
        batch_size: int = 64
    ) -> List[int]:
        """Run inference on a batch of questions."""
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
        
        include_popularity = model_config.get('include_popularity', True)
        
        # Tokenize
        tokens = self.tokenizer(
            questions,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        # Normalize popularity using saved scaler
        pop_normalized = [(p - scaler_mean[0]) / scaler_scale[0] for p in popularity]
        pop_tensor = torch.tensor([[p] for p in pop_normalized], dtype=torch.float32)
        
        # Load classifier and move weights to GPU
        classifier = RouterClassifier(
            input_dim=model_config['input_dim'],
            hidden_dim1=model_config['hidden_dim1'],
            hidden_dim2=model_config['hidden_dim2'],
            num_classes=model_config['num_classes'],
            dropout=model_config['dropout'],
            include_popularity=include_popularity
        ).to(self.device)
        
        # classifier_state comes from CPU, need to move to device
        state_dict_gpu = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                          for k, v in classifier_state.items()}
        classifier.load_state_dict(state_dict_gpu)
        classifier.eval()
        
        # Create dataloader
        dataset = TensorDataset(
            tokens['input_ids'],
            tokens['attention_mask'],
            pop_tensor
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Inference
        all_preds = []
        
        with torch.no_grad():
            for batch in loader:
                input_ids, attention_mask, pop = [b.to(self.device) for b in batch]
                
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                cls_embedding = outputs.last_hidden_state[:, 0, :]
                if include_popularity:
                    combined = torch.cat([cls_embedding, pop], dim=1)
                else:
                    combined = cls_embedding
                logits = classifier(combined)
                
                preds = logits.argmax(1).cpu().numpy()
                all_preds.extend(preds.tolist())
        
        return all_preds


# ── Client ────────────────────────────────────────────────────────────────────
class RouterService:
    """Client for Modal-based router training and inference."""
    
    def __init__(self):
        ModelCls = modal.Cls.from_name(APP_NAME, "RouterModel")
        self.service = ModelCls()
        logger.info(f"Initialized RouterService on Modal")
    
    @staticmethod
    def save_model(result: Dict, filepath: str):
        """Save trained model weights and metadata to disk.
        
        Args:
            result: Training result dict containing classifier_state, scaler params, etc.
            filepath: Path to save the model (e.g., 'models/router_v1.pt')
        """
        import torch
        import os
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(result, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath: str) -> Dict:
        """Load trained model weights and metadata from disk.
        
        Args:
            filepath: Path to the saved model file
            
        Returns:
            Dict containing classifier_state, scaler params, model_config, etc.
        """
        import torch
        
        result = torch.load(filepath, map_location='cpu')
        logger.info(f"Model loaded from {filepath}")
        return result
    
    def train(
        self,
        train_questions: List[str],
        train_popularity: List[float],
        train_labels: List[List[float]],
        test_questions: List[str],
        test_popularity: List[float],
        test_labels: List[List[float]],
        num_classes: int = 2,
        epochs: int = 160,
        batch_size: int = 32,
        lr: float = 0.001,
        unfreeze_layers: int = 0,
        bert_lr: float = 2e-5,
        include_popularity: bool = True,
        patience: int = 10,
        dropout: float | None = None,
        use_scheduler: bool = True,
        seed: int = 42,
    ) -> Dict:
        """Train router on Modal GPU.
        
        Args:
            unfreeze_layers: Number of BERT layers to unfreeze from the end (0=all frozen, 1=last layer, etc.)
            bert_lr: Learning rate for unfrozen BERT layers
            include_popularity: Whether to concatenate popularity as input feature
            patience: Early stopping patience (epochs without test_loss improvement)
            dropout: Optional classifier dropout override.
            use_scheduler: Whether to use cosine learning-rate scheduling.
            seed: Random seed for reproducible training.
        """
        return self.service.train_router.remote(
            train_questions,
            train_popularity,
            train_labels,
            test_questions,
            test_popularity,
            test_labels,
            num_classes,
            epochs,
            batch_size,
            lr,
            unfreeze_layers,
            bert_lr,
            include_popularity,
            patience,
            dropout,
            use_scheduler,
            seed,
        )
    
    def predict(
        self,
        questions: List[str],
        popularity: List[float],
        classifier_state: Dict,
        scaler_mean: List[float],
        scaler_scale: List[float],
        model_config: Dict,
        batch_size: int = 64
    ) -> List[int]:
        """Run inference on Modal GPU."""
        return self.service.predict_batch.remote(
            questions,
            popularity,
            classifier_state,
            scaler_mean,
            scaler_scale,
            model_config,
            batch_size
        )
