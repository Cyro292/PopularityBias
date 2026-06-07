"""Corpus handler for analysis datasets created by the evaluation pipeline.

This handler loads question-level evaluation data with backend performance.
Unlike ParquetCorpusHandler which wraps raw corpus data, AnalysisDatasetHandler
wraps the *output* of the evaluation pipeline — questions + their retrieval
performance across backends.

The analysis dataset contains:
    - question_id, question_text
    - wikipedia_id, wikipedia_title
    - popularity, decile
    - dataset (source QA dataset), split (train/test)
    - backend (e.g., 'bm25_plus', 'ivfpq_high')
    - performance (binary success/failure)
    - retrieved_doc_ids (list of retrieved wikipedia IDs)
    - llm (which LLM was used for generation)
    - context_size (number of documents in context)

Usage:
    handler = AnalysisDatasetHandler(
        collection_name="wiki_full_bil",
        dataset_dir="all_qa_8k",
    )
    
    # CorpusHandler interface
    docs = handler.get_documents([12345, 67890])
    boundaries_uw, boundaries_cw = handler.get_boundaries()
    
    # Analysis-specific methods
    df = handler.load_analysis_dataset()
    questions = handler.get_questions_by_backend("bm25_plus", split="test")
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from langchain.schema import Document

from config import DATA_DIR
from src.corpus_handler.base import CorpusHandler

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Handler
# ═══════════════════════════════════════════════════════════════════════════════

class AnalysisDatasetHandler(CorpusHandler):
    """Handler for loading and querying analysis datasets.
    
    Implements the CorpusHandler interface for analysis datasets, which are
    the output of the RAG evaluation pipeline. Unlike raw corpus handlers,
    this wraps question-level evaluation data with backend performance metrics.
    
    Args:
        collection_name: Collection folder name under DATA_DIR
        dataset_dir: Dataset subdirectory containing analysis_dataset.parquet
        analysis_file: Name of the analysis parquet file (default: 'analysis_dataset.parquet')
        metadata_file: Name of the metadata JSON file (default: 'metadata.json')
    """
    
    def __init__(
        self,
        collection_name: str = "wiki_full_bil",
        dataset_dir: str = "all_qa_8k",
        analysis_file: str = "analysis_dataset.parquet",
        metadata_file: str = "metadata.json",
    ):
        self.collection_name = collection_name
        self.dataset_dir = dataset_dir
        self.analysis_file = analysis_file
        self.metadata_file = metadata_file
        
        self.collection_path = DATA_DIR / collection_name
        self.dataset_path = self.collection_path / dataset_dir
        self.analysis_path = self.dataset_path / analysis_file
        self.metadata_path = self.dataset_path / metadata_file
        
        if not self.analysis_path.exists():
            raise FileNotFoundError(f"Analysis dataset not found: {self.analysis_path}")
        
        # Cache boundaries
        self._boundaries_uw: np.ndarray | None = None
        self._boundaries_cw: np.ndarray | None = None
        
        logger.info(f"Initialized AnalysisDatasetHandler: {self.analysis_path}")
    
    # ── CorpusHandler Interface ───────────────────────────────────────────────
    
    def get_documents(self, wikipedia_ids: int | list[int]) -> list[Document]:
        """Return Document objects for given Wikipedia IDs.
        
        Since analysis datasets contain questions (not documents), this method
        returns the *questions* that have the given wikipedia_id as their
        ground truth. The page_content is the question_text, and metadata
        includes all question-level fields.
        
        Args:
            wikipedia_ids: Wikipedia ID(s) to look up
        
        Returns:
            List of Document objects representing questions with this ground truth.
        """
        if isinstance(wikipedia_ids, int):
            wikipedia_ids = [wikipedia_ids]
        
        df = self.load_analysis_dataset()
        
        # Filter to questions with matching wikipedia_id
        df = df[df['wikipedia_id'].isin(wikipedia_ids)]
        
        # Deduplicate by question_id (multiple rows per question due to backends)
        df = df.drop_duplicates(subset=['question_id'])
        
        # Convert to Documents
        results = []
        for _, row in df.iterrows():
            metadata = row.to_dict()
            question_text = str(metadata.pop('question_text', ''))
            
            results.append(Document(
                page_content=question_text,
                metadata=metadata,
            ))
        
        return results
    
    def get_boundaries(self) -> tuple[np.ndarray, np.ndarray]:
        """Return popularity decile boundaries.
        
        Loads from metadata.json if available, otherwise raises an error.
        Analysis datasets should have metadata created by the evaluation pipeline.
        
        Returns:
            (boundaries_uw, boundaries_cw) tuple of numpy arrays
        """
        if self._boundaries_uw is not None and self._boundaries_cw is not None:
            return self._boundaries_uw, self._boundaries_cw
        
        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {self.metadata_path}. "
                f"Analysis datasets should have metadata created by the evaluation pipeline."
            )
        
        logger.info(f"Loading decile boundaries from {self.metadata_path}")
        
        from src.metrics.decile_utils import load_boundaries_from_metadata
        boundaries_uw, boundaries_cw, _ = load_boundaries_from_metadata(self.metadata_path)
        
        self._boundaries_uw = boundaries_uw
        self._boundaries_cw = boundaries_cw
        
        return boundaries_uw, boundaries_cw
    
    def create_metadata(self, output_path: Path) -> None:
        """Create metadata file with decile boundaries.
        
        For analysis datasets, metadata should be created by the evaluation pipeline.
        This method is not typically used, but provided for interface compatibility.
        
        Args:
            output_path: Path where metadata.json should be written
        """
        raise NotImplementedError(
            "Analysis datasets should have metadata created by the evaluation pipeline. "
            "Use the corpus-level ParquetCorpusHandler.create_metadata() instead."
        )
    
    # ── Analysis-Specific Methods ─────────────────────────────────────────────
    
    def load_analysis_dataset(self) -> pd.DataFrame:
        """Load the full analysis dataset.
        
        Returns:
            DataFrame with all columns from the analysis dataset.
        """
        logger.info(f"Loading analysis dataset from {self.analysis_path}...")
        df = pd.read_parquet(self.analysis_path)
        logger.info(f"Loaded {len(df):,} rows with {len(df.columns)} columns")
        return df
    
    def get_questions_by_backend(
        self,
        backend: str,
        split: Literal["train", "test"] | None = None,
        dataset: str | None = None,
    ) -> pd.DataFrame:
        """Get questions for a specific backend.
        
        Args:
            backend: Backend name (e.g., 'bm25_plus', 'ivfpq_high')
            split: Optional data split filter ('train' or 'test')
            dataset: Optional dataset name filter (e.g., 'natural_questions')
        
        Returns:
            DataFrame filtered to the specified backend (and split/dataset if provided).
        """
        df = self.load_analysis_dataset()
        
        # Filter by backend
        df = df[df['backend'] == backend]
        
        # Optional filters
        if split:
            df = df[df['split'] == split]
        if dataset:
            df = df[df['dataset'] == dataset]
        
        logger.info(f"Filtered to backend={backend}, split={split}, dataset={dataset}: {len(df):,} rows")
        return df
    
    def get_unique_questions(
        self,
        split: Literal["train", "test"] | None = None,
        dataset: str | None = None,
    ) -> pd.DataFrame:
        """Get unique questions (one row per question_id).
        
        Since the analysis dataset has one row per (question, backend) pair,
        this method deduplicates to get one row per question.
        
        Args:
            split: Optional data split filter
            dataset: Optional dataset name filter
        
        Returns:
            DataFrame with unique questions (question_id, question_text, popularity, etc.)
        """
        df = self.load_analysis_dataset()
        
        # Optional filters
        if split:
            df = df[df['split'] == split]
        if dataset:
            df = df[df['dataset'] == dataset]
        
        # Deduplicate by question_id
        df = df.drop_duplicates(subset=['question_id']).copy()
        
        logger.info(f"Unique questions (split={split}, dataset={dataset}): {len(df):,}")
        return df
    
    def get_backend_performance_summary(
        self,
        backend: str,
        split: Literal["train", "test"] | None = None,
        group_by: str | None = None,
    ) -> pd.DataFrame:
        """Get performance summary for a backend.
        
        Args:
            backend: Backend name
            split: Optional split filter
            group_by: Optional column to group by (e.g., 'dataset', 'decile')
        
        Returns:
            DataFrame with performance metrics (success rate, count, etc.)
        """
        df = self.get_questions_by_backend(backend, split=split)
        
        if group_by:
            # Group by specified column
            grouped = df.groupby(group_by).agg({
                'performance': ['mean', 'sum', 'count'],
                'question_id': 'count',
            })
            grouped.columns = ['success_rate', 'successes', 'total_performance', 'total_questions']
            return grouped.reset_index()
        else:
            # Overall summary
            summary = {
                'backend': backend,
                'split': split or 'all',
                'total_questions': len(df),
                'successes': df['performance'].sum(),
                'success_rate': df['performance'].mean(),
            }
            return pd.DataFrame([summary])
    
    def pivot_to_question_centric(
        self,
        backends: list[str],
        split: Literal["train", "test"] | None = None,
        metric: str = 'performance',
    ) -> pd.DataFrame:
        """Pivot dataset to question-centric format.
        
        Transform from (question_id, backend, performance) to
        (question_id, bm25_performance, faiss_performance, ...).
        
        Args:
            backends: List of backend names to include as columns
            split: Optional split filter
            metric: Column to pivot (default: 'performance')
        
        Returns:
            DataFrame with one row per question, columns for each backend.
        """
        df = self.load_analysis_dataset()
        
        # Filter to requested backends
        df = df[df['backend'].isin(backends)]
        
        # Optional split filter
        if split:
            df = df[df['split'] == split]
        
        # Pivot
        pivoted = df.pivot_table(
            index='question_id',
            columns='backend',
            values=metric,
            aggfunc='first'
        ).reset_index()
        
        # Merge metadata
        meta = df.groupby('question_id').agg({
            'question_text': 'first',
            'popularity': 'first',
            'dataset': 'first',
            'split': 'first',
            'decile': 'first',
            'wikipedia_id': 'first',
            'wikipedia_title': 'first',
        }).reset_index()
        
        result = meta.merge(pivoted, on='question_id')
        
        logger.info(f"Pivoted to question-centric format: {len(result):,} questions × {len(backends)} backends")
        return result
    
    def get_backend_names(self) -> list[str]:
        """Get list of all backend names in the dataset."""
        df = self.load_analysis_dataset()
        backends = sorted(df['backend'].unique())
        return backends
    
    def get_dataset_names(self) -> list[str]:
        """Get list of all source dataset names."""
        df = self.load_analysis_dataset()
        datasets = sorted(df['dataset'].unique())
        return datasets
    
    def compute_retrieval_metric(
        self,
        retrieved_doc_ids: list[int] | None,
        ground_truth_id: int | None,
        metric: Literal["mrr", "recall"] = "mrr",
        k: int = 20,
    ) -> float:
        """Compute retrieval metric for a single question.
        
        Args:
            retrieved_doc_ids: List of retrieved document IDs (in rank order)
            ground_truth_id: Ground truth document ID
            metric: Metric to compute ('mrr' or 'recall')
            k: Cutoff for metric computation (only consider top-k results)
        
        Returns:
            Metric value (MRR: 0 to 1, Recall: 0 or 1)
        """
        # Handle missing data
        if retrieved_doc_ids is None or ground_truth_id is None:
            return 0.0
        
        if not isinstance(retrieved_doc_ids, (list, tuple)):
            retrieved_doc_ids = list(retrieved_doc_ids)

        if len(retrieved_doc_ids) == 0:
            return 0.0
        
        # Convert to strings for comparison
        ground_truth_str = str(ground_truth_id)
        retrieved_strs = [str(x) for x in retrieved_doc_ids[:k]]
        
        # Check if ground truth is in top-k
        if ground_truth_str not in retrieved_strs:
            return 0.0
        
        # Compute metric
        if metric == "recall":
            return 1.0  # Binary: found or not found
        elif metric == "mrr":
            rank = retrieved_strs.index(ground_truth_str) + 1  # 1-indexed
            return 1.0 / rank
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def compute_retrieval_metrics_for_backends(
        self,
        backends: list[str],
        metric: Literal["mrr", "recall"] = "mrr",
        k: int = 20,
        split: Literal["train", "test"] | None = None,
    ) -> pd.DataFrame:
        """Compute retrieval metrics for all backends and pivot to question-centric format.
        
        This is used for training the router on retrieval metrics instead of answer metrics.
        
        Args:
            backends: List of backend names to compute metrics for
            metric: Metric to compute ('mrr' or 'recall')
            k: Cutoff for metric computation (top-k)
            split: Optional split filter
        
        Returns:
            DataFrame with one row per question, columns for each backend's metric value.
            Example columns: question_id, question_text, popularity, bm25_plus, ivfpq_high
        """
        df = self.load_analysis_dataset()
        
        # Filter to requested backends and split
        df = df[df['backend'].isin(backends)]
        if split:
            df = df[df['split'] == split]
        
        # Compute metric for each row
        logger.info(f"Computing {metric}@{k} for {len(df):,} rows...")
        df[f'{metric}@{k}'] = df.apply(
            lambda row: self.compute_retrieval_metric(
                retrieved_doc_ids=row.get('retrieved_doc_ids'),
                ground_truth_id=row.get('wikipedia_id'),
                metric=metric,
                k=k,
            ),
            axis=1
        )
        
        # Pivot to question-centric format
        pivoted = df.pivot_table(
            index='question_id',
            columns='backend',
            values=f'{metric}@{k}',
            aggfunc='first'
        ).reset_index()
        
        # Merge metadata
        meta = df.groupby('question_id').agg({
            'question_text': 'first',
            'popularity': 'first',
            'dataset': 'first',
            'split': 'first',
            'decile': 'first',
            'wikipedia_id': 'first',
        }).reset_index()
        
        result = meta.merge(pivoted, on='question_id')
        
        logger.info(f"Computed {metric}@{k}: {len(result):,} questions × {len(backends)} backends")
        return result
