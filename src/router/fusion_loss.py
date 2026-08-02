"""Soft-rank MRR loss for backend-transformed fixed-RRF fusion.

The learned component does not weight RRF directly. Instead, it predicts
backend-specific transformed scores for the original rank positions, reorders
documents within each backend, and then applies the unchanged RRF formula.

Training uses soft ranks in two places:

1. Within each backend, transformed scores induce a soft reranking.
2. After fixed-RRF fusion, the gold document's fused rank is approximated with
   a soft-rank MRR loss.
"""
from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


def _build_candidate_matrices(
    bm25_doc_ids: list[list[str]],
    faiss_doc_ids: list[list[str]],
    ground_truth_ids: list[str],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build union-doc lookup matrices for the two backends.

    Returns:
        Tuple of ``(bm25_pos_idx, faiss_pos_idx, bm25_present, faiss_present,
        gold_idx)`` over the per-question union document vocabulary.
    """
    batch_size = len(ground_truth_ids)
    max_union = max(
        len(set(bm25_doc_ids[i]) | set(faiss_doc_ids[i]))
        for i in range(batch_size)
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


def _soft_backend_ranks(
    doc_scores: torch.Tensor,
    present_mask: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """Approximate per-backend ranks from transformed backend scores.

    Args:
        doc_scores: ``(batch, num_docs)`` transformed scores for one backend.
        present_mask: ``(batch, num_docs)`` mask marking docs present in backend.
        tau: Soft-rank temperature.
    """
    batch_size, num_docs = doc_scores.shape
    s_i = doc_scores.unsqueeze(2)
    s_j = doc_scores.unsqueeze(1)
    pairwise = torch.sigmoid((s_i - s_j) / tau)

    valid_pairs = present_mask.unsqueeze(2) & present_mask.unsqueeze(1)
    eye = torch.eye(num_docs, dtype=torch.bool, device=doc_scores.device).unsqueeze(0)
    valid_pairs = valid_pairs & ~eye

    return 1.0 + (pairwise * valid_pairs.to(doc_scores.dtype)).sum(dim=1)


def _fused_scores_from_transforms(
    transformed_scores: torch.Tensor,
    batch_data: list[dict[str, Any]],
    backend_names: list[str],
    rrf_k: int,
    tau: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute soft backend reranks and fixed-RRF fused scores.

    Args:
        transformed_scores: ``(batch, 2, depth)`` transformed position scores.
        batch_data: Per-question data dicts.
        backend_names: ``[bm25_name, faiss_name]``.
        rrf_k: Fixed RRF smoothing constant.
        tau: Soft-rank temperature.

    Returns:
        Tuple of ``(fused_scores, gold_idx, candidate_mask)``.
    """
    device = transformed_scores.device
    depth = transformed_scores.shape[2]
    bm25_key = f"{backend_names[0]}_doc_ids"
    faiss_key = f"{backend_names[1]}_doc_ids"

    bm25_doc_ids = [list(d[bm25_key])[:depth] for d in batch_data]
    faiss_doc_ids = [list(d[faiss_key])[:depth] for d in batch_data]
    ground_truth_ids = [str(d["wikipedia_id"]) for d in batch_data]

    bm25_pos_idx, faiss_pos_idx, bm25_present, faiss_present, gold_idx = (
        _build_candidate_matrices(bm25_doc_ids, faiss_doc_ids, ground_truth_ids, device)
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


class FusionRRFLoss(torch.nn.Module):
    """Soft-rank MRR loss for transformed-rank fixed RRF.

    Args:
        backend_names: Backend names in model order.
        rrf_k: Fixed RRF smoothing constant.
        temperature: Soft-rank temperature ``tau``.
        loss_top_k: Legacy no-op retained for CLI compatibility.
    """

    def __init__(
        self,
        backend_names: list[str] | None = None,
        rrf_k: int = 60,
        temperature: float = 0.02,
        loss_top_k: int = 20,
    ) -> None:
        super().__init__()
        self.backend_names = backend_names or ["bm25_plus", "ivfpq_high"]
        self.rrf_k = rrf_k
        self.temperature = temperature
        self.loss_top_k = loss_top_k

    def forward(
        self,
        transformed_scores: torch.Tensor,
        batch_data: list[dict[str, Any]],
    ) -> torch.Tensor:
        """Compute soft-rank MRR loss on final fixed-RRF fused scores."""
        fused_scores, gold_idx, candidate_mask = _fused_scores_from_transforms(
            transformed_scores,
            batch_data,
            self.backend_names,
            self.rrf_k,
            max(self.temperature, 1e-8),
        )

        valid_mask = gold_idx >= 0
        if not valid_mask.any():
            return torch.tensor(0.0, device=transformed_scores.device, requires_grad=True)

        fused_scores = fused_scores[valid_mask]
        gold_idx = gold_idx[valid_mask]
        candidate_mask = candidate_mask[valid_mask]

        batch_indices = torch.arange(fused_scores.shape[0], device=transformed_scores.device)
        gold_scores = fused_scores[batch_indices, gold_idx].unsqueeze(1)
        pairwise_probs = torch.sigmoid((fused_scores - gold_scores) / max(self.temperature, 1e-8))

        gold_mask = torch.zeros_like(candidate_mask)
        gold_mask[batch_indices, gold_idx] = True
        competitor_mask = candidate_mask & ~gold_mask

        soft_rank = 1.0 + (pairwise_probs * competitor_mask.to(fused_scores.dtype)).sum(dim=1)
        return torch.log(soft_rank).mean()


def compute_mrr(
    transformed_scores: torch.Tensor,
    batch_data: list[dict[str, Any]],
    backend_names: list[str],
    rrf_k: int = 60,
) -> dict[str, float]:
    """Compute hard MRR by reranking backends then applying plain RRF."""
    reciprocal_ranks: list[float] = []
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

        fused: dict[str, float] = {}
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
