"""Router module for neural backend selection.

This module provides:
    - router_service.py: Modal GPU service for training and inference
    - train_router.py: CLI script for training router models

For data handling, use AnalysisDatasetHandler from corpus_handler:
    from src.corpus_handler import AnalysisDatasetHandler

For using trained routers in RAG pipelines:
    from src.rag import NeuralRouterRagService

Quick Start:

1. Train a router:
    python -m src.router.train_router \\
        --collection wiki_full_bil \\
        --dataset-dir all_qa_8k \\
        --model-name my_router \\
        --label-mode retrieval \\
        --retrieval-metric mrr \\
        --retrieval-k 20 \\
        --epochs 80

2. Use in retrieval_runner.py:
    Add to RetrievalConfig.backends:
    
    RetrievalBackend(
        key             = "neural_router_strict",
        label           = "Neural Router (Strict)",
        type            = "neural_router",
        router_sub_keys = ("bm25_plus", "ivfpq_high"),
        service_kwargs  = {
            "model_path": "models/my_router.pt",
            "backend_order": ["bm25_plus", "ivfpq_high"],
            "strict": True,
        },
    )

3. Use directly in code:
    from src.rag import NeuralRouterRagService
    from src.rag.bm25_rag_service import BM25RagService
    from src.rag.faiss_rag_service import FaissRagService
    
    # Load sub-backends
    bm25_service = BM25RagService(...)
    faiss_service = FaissRagService(...)
    
    # Create neural router
    router = NeuralRouterRagService(
        model_path="models/my_router.pt",
        backends={"bm25": bm25_service, "faiss": faiss_service},
        backend_order=["bm25", "faiss"],
        strict=True,  # or False for hybrid mode
    )
    
    # Retrieve
    docs = router.batch_retrieve(
        questions=["What is the capital of France?"],
        popularities=[1000.0],
        top_k=20,
    )
"""
from __future__ import annotations

__all__ = [
    "RouterService",
]

from src.router.router_service import RouterService
