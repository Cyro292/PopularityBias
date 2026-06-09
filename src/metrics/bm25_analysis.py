"""BM25+ retrieval analysis per popularity decile.

Provides functions to diagnose why BM25+ retrieval performance varies
across popularity deciles.  Each function operates on the enriched
results DataFrame produced by ``shared_setup`` and returns per-decile
statistics ready for plotting.

Public API
----------
``compute_score_gap_by_decile``
    Compare BM25 scores of the gold document vs the top-1 retrieved doc.

``compute_popularity_displacement``
    Analyse whether retrieved docs are systematically more/less popular
    than the gold document.

``compute_rank_distribution_by_decile``
    Distribution of gold-document ranks within the retrieved top-k.

``compute_competition_stats``
    Corpus-level competition metrics (chunks/doc, chunks/question).

``compute_score_vs_length_by_decile``
    Relationship between BM25 score, document length, and decile.

``compute_retrieved_composition_by_decile``
    Where retrieved documents come from (decile composition, overlap).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_NUM_DECILES = 10


# === Score Gap Analysis ===


def compute_score_gap_by_decile(
    df: pd.DataFrame,
    decile_col: str,
    *,
    top_k: int = 10,
) -> pd.DataFrame:
    """Compute per-decile BM25 score-gap between gold and top-1 retrieved.

    For each question where the gold document appears in the top-k,
    measures how much lower the gold score is compared to the highest-
    scoring retrieved document.  A large gap means BM25 assigns much
    higher scores to competing documents.

    Args:
        df: Enriched results DataFrame with ``topk_ids``, ``topk_scores``,
            ``wikipedia_id``, and the decile column.
        decile_col: Name of the 0-based decile column.
        top_k: Number of retrieved documents to consider.

    Returns:
        DataFrame with one row per decile and columns:
        ``decile``, ``count``, ``hit_rate``,
        ``mean_gold_score``, ``mean_top1_score``,
        ``mean_score_gap``, ``mean_score_ratio``.
    """
    rows: list[dict[str, Any]] = []
    for d in range(_NUM_DECILES):
        sub = df[df[decile_col] == d]
        n = len(sub)
        if n == 0:
            continue

        gold_scores: list[float] = []
        top1_scores: list[float] = []
        score_gaps: list[float] = []
        score_ratios: list[float] = []
        hits = 0

        for _, row in sub.iterrows():
            topk_ids = list(row["topk_ids"]) if hasattr(row["topk_ids"], "__iter__") else []
            topk_scores = list(row["topk_scores"]) if hasattr(row["topk_scores"], "__iter__") else []
            gold_id = str(row["wikipedia_id"])

            if not topk_scores or all(pd.isna(s) for s in topk_scores):
                continue

            top1 = float(topk_scores[0]) if not pd.isna(topk_scores[0]) else float("nan")

            if gold_id in topk_ids[:top_k]:
                idx = topk_ids.index(gold_id)
                gold_s = float(topk_scores[idx]) if idx < len(topk_scores) and not pd.isna(topk_scores[idx]) else float("nan")
                hits += 1
                if not np.isnan(top1) and not np.isnan(gold_s):
                    gold_scores.append(gold_s)
                    top1_scores.append(top1)
                    gap = top1 - gold_s
                    score_gaps.append(gap)
                    ratio = gold_s / top1 if top1 > 0 else float("nan")
                    score_ratios.append(ratio)

        hit_rate = hits / n if n > 0 else 0.0

        def _mean(lst: list[float]) -> float:
            return float(np.mean(lst)) if lst else float("nan")

        rows.append({
            "decile": d,
            "count": n,
            "hit_rate": hit_rate,
            "mean_gold_score": _mean(gold_scores),
            "mean_top1_score": _mean(top1_scores),
            "mean_score_gap": _mean(score_gaps),
            "mean_score_ratio": _mean(score_ratios),
        })

    return pd.DataFrame(rows)


# === Popularity Displacement Analysis ===


def compute_popularity_displacement(
    df: pd.DataFrame,
    decile_col: str,
    *,
    top_k: int = 10,
) -> pd.DataFrame:
    """Analyse whether BM25+ retrieves docs more/less popular than gold.

    For each question, compares the gold document's popularity to the
    mean popularity of retrieved documents.  When gold is NOT in the
    top-k, this shows what types of documents displaced it.

    Args:
        df: Enriched results DataFrame with ``topk_popularities``,
            ``popularity_avg``, and the decile column.
        decile_col: Name of the 0-based decile column.
        top_k: Number of retrieved documents to consider.

    Returns:
        DataFrame with one row per decile and columns:
        ``decile``, ``count``,
        ``mean_gold_pop``, ``mean_retrieved_pop``,
        ``mean_pop_ratio``, ``mean_n_more_popular``,
        ``mean_n_less_popular``.
    """
    rows: list[dict[str, Any]] = []
    for d in range(_NUM_DECILES):
        sub = df[df[decile_col] == d]
        n = len(sub)
        if n == 0:
            continue

        gold_pops: list[float] = []
        retrieved_pops: list[float] = []
        pop_ratios: list[float] = []
        n_more: list[int] = []
        n_less: list[int] = []

        for _, row in sub.iterrows():
            topk_pops = list(row["topk_popularities"]) if hasattr(row["topk_popularities"], "__iter__") else []
            topk_pops = topk_pops[:top_k]
            if not topk_pops:
                continue
            gold_pop = float(row["popularity_avg"])
            topk_pops_f = [float(p) for p in topk_pops if p is not None and not pd.isna(p)]
            if not topk_pops_f:
                continue

            mean_retrieved = float(np.mean(topk_pops_f))
            gold_pops.append(gold_pop)
            retrieved_pops.append(mean_retrieved)
            pop_ratios.append(mean_retrieved / gold_pop if gold_pop > 0 else float("nan"))
            n_more.append(sum(1 for p in topk_pops_f if p > gold_pop))
            n_less.append(sum(1 for p in topk_pops_f if p < gold_pop))

        def _mean(lst: list[float]) -> float:
            return float(np.mean(lst)) if lst else float("nan")

        rows.append({
            "decile": d,
            "count": n,
            "mean_gold_pop": _mean(gold_pops),
            "mean_retrieved_pop": _mean(retrieved_pops),
            "mean_pop_ratio": _mean(pop_ratios),
            "mean_n_more_popular": _mean(n_more),
            "mean_n_less_popular": _mean(n_less),
        })

    return pd.DataFrame(rows)


# === Rank Distribution Analysis ===


def compute_rank_distribution_by_decile(
    df: pd.DataFrame,
    decile_col: str,
    *,
    top_k: int = 10,
) -> pd.DataFrame:
    """Compute the distribution of gold-document ranks per decile.

    Shows what fraction of gold documents fall at rank 1, 2, ..., top_k
    vs not found at all.

    Args:
        df: Enriched results DataFrame with ``rank`` and the decile column.
        decile_col: Name of the 0-based decile column.
        top_k: Maximum rank to break out individually.

    Returns:
        DataFrame with one row per decile and columns:
        ``decile``, ``count``, ``frac_not_found``,
        ``frac_rank_1``, ``frac_rank_2``, ... ``frac_rank_{top_k}``,
        ``mean_rank_found``.
    """
    rows: list[dict[str, Any]] = []
    for d in range(_NUM_DECILES):
        sub = df[df[decile_col] == d]
        n = len(sub)
        if n == 0:
            continue

        row_data: dict[str, Any] = {"decile": d, "count": n}

        ranks = sub["rank"].dropna()
        n_found = len(ranks)
        n_not_found = n - n_found
        row_data["frac_not_found"] = n_not_found / n if n > 0 else 0.0

        for k in range(1, top_k + 1):
            row_data[f"frac_rank_{k}"] = int((ranks == k).sum()) / n if n > 0 else 0.0

        row_data["mean_rank_found"] = float(ranks.mean()) if n_found > 0 else float("nan")
        row_data["median_rank_found"] = float(ranks.median()) if n_found > 0 else float("nan")

        rows.append(row_data)

    return pd.DataFrame(rows)


# === Competition Stats ===


def compute_competition_stats(
    corpus_docs: np.ndarray,
    corpus_chunks: np.ndarray,
    questions_per_decile: np.ndarray,
) -> pd.DataFrame:
    """Compute corpus-level competition metrics per decile.

    Args:
        corpus_docs: Document count per decile (length 10).
        corpus_chunks: Chunk count per decile (length 10).
        questions_per_decile: Question count per decile (length 10).

    Returns:
        DataFrame with columns: ``decile``, ``n_docs``, ``n_chunks``,
        ``n_questions``, ``chunks_per_doc``, ``chunks_per_question``,
        ``random_doc_hit_rate``, ``random_chunk_hit_rate``,
        ``competition_index``.
    """
    rows: list[dict[str, Any]] = []
    for d in range(_NUM_DECILES):
        nd = int(corpus_docs[d]) if d < len(corpus_docs) else 0
        nc = int(corpus_chunks[d]) if d < len(corpus_chunks) else 0
        nq = int(questions_per_decile[d]) if d < len(questions_per_decile) else 0
        rows.append({
            "decile": d,
            "n_docs": nd,
            "n_chunks": nc,
            "n_questions": nq,
            "chunks_per_doc": nc / nd if nd > 0 else 0.0,
            "chunks_per_question": nc / nq if nq > 0 else float("inf"),
            "random_doc_hit_rate": 1.0 / nd if nd > 0 else 0.0,
            "random_chunk_hit_rate": 1.0 / nc if nc > 0 else 0.0,
            "competition_index": nc / nq if nq > 0 else float("inf"),
        })
    return pd.DataFrame(rows)


# === Score vs Length Analysis ===


def compute_score_vs_length_by_decile(
    df: pd.DataFrame,
    decile_col: str,
    *,
    top_k: int = 10,
) -> pd.DataFrame:
    """Analyse BM25 score vs document length per decile.

    Compares the BM25 score of the top-1 retrieved document and the
    gold document's length across deciles to see whether length
    normalisation systematically penalises popular (longer) docs.

    Args:
        df: Enriched results DataFrame with ``topk_scores``, ``doc_length``,
            and the decile column.
        decile_col: Name of the 0-based decile column.
        top_k: Number of retrieved documents to consider.

    Returns:
        DataFrame with columns: ``decile``, ``count``,
        ``mean_top1_score``, ``mean_gold_doc_length``,
        ``mean_gold_score_if_found``.
    """
    rows: list[dict[str, Any]] = []
    for d in range(_NUM_DECILES):
        sub = df[df[decile_col] == d]
        n = len(sub)
        if n == 0:
            continue

        top1_scores: list[float] = []
        gold_scores: list[float] = []
        doc_lengths: list[float] = []

        for _, row in sub.iterrows():
            topk_ids = list(row["topk_ids"]) if hasattr(row["topk_ids"], "__iter__") else []
            topk_scores = list(row["topk_scores"]) if hasattr(row["topk_scores"], "__iter__") else []
            gold_id = str(row["wikipedia_id"])
            dl = row.get("doc_length")
            if dl is not None and not pd.isna(dl):
                doc_lengths.append(float(dl))

            if topk_scores and not pd.isna(topk_scores[0]):
                top1_scores.append(float(topk_scores[0]))

            if gold_id in topk_ids[:top_k]:
                idx = topk_ids.index(gold_id)
                if idx < len(topk_scores) and not pd.isna(topk_scores[idx]):
                    gold_scores.append(float(topk_scores[idx]))

        def _mean(lst: list[float]) -> float:
            return float(np.mean(lst)) if lst else float("nan")

        rows.append({
            "decile": d,
            "count": n,
            "mean_top1_score": _mean(top1_scores),
            "mean_gold_score_if_found": _mean(gold_scores),
            "mean_gold_doc_length": _mean(doc_lengths),
        })

    return pd.DataFrame(rows)


# === Retrieved Composition Analysis ===


def compute_retrieved_composition_by_decile(
    df: pd.DataFrame,
    decile_col: str,
    boundaries: np.ndarray,
    *,
    top_k: int = 10,
) -> pd.DataFrame:
    """Analyse the popularity composition of BM25+ retrieved sets.

    Assigns deciles to each retrieved document using its popularity and
    then computes, per gold-document decile, what fraction of retrieved
    docs come from the same or different deciles.

    Args:
        df: Enriched results DataFrame with ``topk_popularities`` and
            the decile column.
        decile_col: Name of the 0-based decile column.
        boundaries: Decile boundary array (length 11) for assigning
            retrieved docs to deciles.
        top_k: Number of retrieved documents to consider.

    Returns:
        DataFrame with columns: ``decile``, ``count``,
        ``mean_frac_same_decile``, ``mean_frac_higher_decile``,
        ``mean_frac_lower_decile``,
        ``mean_retrieved_decile``.
    """
    from src.metrics.decile_utils import assign_decile

    rows: list[dict[str, Any]] = []
    for d in range(_NUM_DECILES):
        sub = df[df[decile_col] == d]
        n = len(sub)
        if n == 0:
            continue

        frac_same: list[float] = []
        frac_higher: list[float] = []
        frac_lower: list[float] = []
        mean_decile: list[float] = []

        for _, row in sub.iterrows():
            topk_pops = list(row["topk_popularities"]) if hasattr(row["topk_popularities"], "__iter__") else []
            topk_pops = topk_pops[:top_k]
            pops_valid = [float(p) for p in topk_pops if p is not None and not pd.isna(p) and p >= 0]
            if not pops_valid:
                continue

            pop_series = pd.Series(pops_valid)
            ret_deciles = assign_decile(pop_series, boundaries).astype(int)
            ret_deciles_list = ret_deciles.tolist() if hasattr(ret_deciles, "tolist") else list(ret_deciles)

            n_ret = len(ret_deciles_list)
            if n_ret == 0:
                continue

            same = sum(1 for rd in ret_deciles_list if rd == d) / n_ret
            higher = sum(1 for rd in ret_deciles_list if rd > d) / n_ret
            lower = sum(1 for rd in ret_deciles_list if rd < d) / n_ret
            frac_same.append(same)
            frac_higher.append(higher)
            frac_lower.append(lower)
            mean_decile.append(float(np.mean(ret_deciles_list)))

        def _mean(lst: list[float]) -> float:
            return float(np.mean(lst)) if lst else float("nan")

        rows.append({
            "decile": d,
            "count": n,
            "mean_frac_same_decile": _mean(frac_same),
            "mean_frac_higher_decile": _mean(frac_higher),
            "mean_frac_lower_decile": _mean(frac_lower),
            "mean_retrieved_decile": _mean(mean_decile),
        })

    return pd.DataFrame(rows)


# === Master Table ===


def compute_bm25_diagnostic_table(
    df: pd.DataFrame,
    decile_col: str,
    corpus_docs: np.ndarray,
    corpus_chunks: np.ndarray,
    boundaries: np.ndarray,
    *,
    top_k: int = 10,
) -> pd.DataFrame:
    """Compute a single diagnostic table combining all analyses.

    Convenience function that merges the output of
    :func:`compute_score_gap_by_decile`,
    :func:`compute_popularity_displacement`,
    :func:`compute_competition_stats`, and
    :func:`compute_score_vs_length_by_decile` into one DataFrame.

    Args:
        df: Enriched results DataFrame.
        decile_col: Name of the 0-based decile column.
        corpus_docs: Document count per decile (length 10).
        corpus_chunks: Chunk count per decile (length 10).
        boundaries: Decile boundary array for retrieved-doc decile assignment.
        top_k: Number of retrieved documents to consider.

    Returns:
        Wide DataFrame with one row per decile and all diagnostic columns.
    """
    questions_per_decile = np.array([
        int((df[decile_col] == d).sum()) for d in range(_NUM_DECILES)
    ], dtype=float)

    gap = compute_score_gap_by_decile(df, decile_col, top_k=top_k)
    disp = compute_popularity_displacement(df, decile_col, top_k=top_k)
    comp = compute_competition_stats(corpus_docs, corpus_chunks, questions_per_decile)
    slen = compute_score_vs_length_by_decile(df, decile_col, top_k=top_k)

    merged = comp.merge(gap, on="decile", how="outer")
    merged = merged.merge(disp, on="decile", how="outer", suffixes=("", "_disp"))
    merged = merged.merge(slen, on="decile", how="outer", suffixes=("", "_slen"))
    return merged.sort_values("decile").reset_index(drop=True)
