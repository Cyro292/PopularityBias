"""Query-conditioned backend rank transformation model for fixed RRF.

This model does not learn fusion weights. Instead, it learns how to reshape
each backend's internal ranking distribution before standard RRF is applied.

For a query ``q`` and original backend rank position ``r``:

    bm25_score'(r, q)  = f_theta(r, q)
    faiss_score'(r, q) = g_phi(r, q)

Documents are re-ranked *within each backend* by these transformed scores, and
the final fusion remains the standard unweighted RRF:

    score(d) = 1 / (k + rank'_bm25(d)) + 1 / (k + rank'_faiss(d))

The model outputs one transformed score per original rank position for each
backend. A decreasing base score preserves the original ordering at
initialization, while learned residuals allow the model to promote or demote
positions depending on the query.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BackendTransformPredictor(nn.Module):
    """Predict per-backend transformed position scores for fixed-RRF fusion.

    Args:
        input_dim: Input dimensionality (768 BERT-only, 769 with popularity).
        rrf_depth: Number of positions per backend to transform.
        hidden_dim1: First hidden layer size.
        hidden_dim2: Second hidden layer size.
        dropout: Dropout probability.
        residual_scale: Maximum magnitude of learned position residuals.
    """

    def __init__(
        self,
        input_dim: int = 769,
        rrf_depth: int = 60,
        hidden_dim1: int = 32,
        hidden_dim2: int = 16,
        dropout: float = 0.3,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
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
        self.register_buffer("base_position_scores", base.view(1, 1, rrf_depth))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return transformed backend scores of shape ``(batch, 2, depth)``.

        Output channel 0 corresponds to BM25 positions, channel 1 to FAISS.
        """
        residual = self.network(x).view(-1, 2, self.rrf_depth)
        residual = self.residual_scale * torch.tanh(residual)
        return self.base_position_scores + residual


# Backwards-compatible alias for older imports.
FusionWeightPredictor = BackendTransformPredictor
