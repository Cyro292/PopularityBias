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

``compute_bm25_lexical_factor_rows``
    Per-query lexical factors: query IDF, query-target overlap, target length,
    and target chunk fragmentation.

``compute_ranked_score_competition_by_decile``
    Direct score-based competition diagnostics from ranked BM25 results.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
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

            if gold_id in topk_ids[:top_k]:
                idx = topk_ids.index(gold_id)
                hits += 1

                if not topk_scores or all(pd.isna(s) for s in topk_scores):
                    continue

                top1 = float(topk_scores[0]) if not pd.isna(topk_scores[0]) else float("nan")
                gold_s = float(topk_scores[idx]) if idx < len(topk_scores) and not pd.isna(topk_scores[idx]) else float("nan")
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


def compute_ranked_score_competition_by_decile(
    df: pd.DataFrame,
    decile_col: str,
    *,
    depth: int = 50,
    epsilon: float = 0.1,
) -> pd.DataFrame:
    """Measure direct lexical competition from scored BM25 rankings.

    The diagnostic compares the highest-scoring chunk from the target article
    with the strongest non-target chunk in each query's ranked candidate set.
    It also counts non-target chunks whose scores are within ``epsilon`` of the
    best target score. Results are exact only when the supplied ranking contains
    every scored candidate; with a top-``depth`` ranking they are explicitly
    lower-bound, candidate-set diagnostics.

    Args:
        df: Results DataFrame containing ``topk_ids``, ``topk_scores``, the
            target ``wikipedia_id``, and ``decile_col``.
        decile_col: Name of the 0-based popularity decile column.
        depth: Number of ranked candidates per query to inspect.
        epsilon: Score tolerance used to classify a non-target chunk as a
            near-tie with the best target chunk.

    Returns:
        One row per decile with candidate coverage, the mean best-target and
        best-non-target scores, their margin, the mean near-tie count, and the
        fraction of queries whose best target chunk is outscored.

    Raises:
        ValueError: If ``depth`` is not positive or ``epsilon`` is negative.
        KeyError: If required result columns are absent.
    """
    if depth <= 0:
        raise ValueError("depth must be positive")
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")

    required = {"topk_ids", "topk_scores", "wikipedia_id", decile_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"df missing required columns: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    for decile in range(_NUM_DECILES):
        sub = df[df[decile_col] == decile]
        candidate_queries = 0
        target_present = 0
        comparable_queries = 0
        best_target_scores: list[float] = []
        best_non_target_scores: list[float] = []
        margins: list[float] = []
        near_tie_counts: list[int] = []
        target_outscored: list[float] = []

        for _, row in sub.iterrows():
            ids_value = row["topk_ids"]
            scores_value = row["topk_scores"]
            if not hasattr(ids_value, "__iter__") or not hasattr(scores_value, "__iter__"):
                continue

            candidates = [
                (str(doc_id), float(score))
                for doc_id, score in zip(list(ids_value)[:depth], list(scores_value)[:depth])
                if score is not None and not pd.isna(score)
            ]
            if not candidates:
                continue

            candidate_queries += 1
            gold_id = str(row["wikipedia_id"])
            target_scores = [score for doc_id, score in candidates if doc_id == gold_id]
            non_target_scores = [score for doc_id, score in candidates if doc_id != gold_id]
            if not target_scores:
                continue

            target_present += 1
            if not non_target_scores:
                continue

            comparable_queries += 1
            best_target = max(target_scores)
            best_non_target = max(non_target_scores)
            best_target_scores.append(best_target)
            best_non_target_scores.append(best_non_target)
            margins.append(best_target - best_non_target)
            near_tie_counts.append(
                sum(score >= best_target - epsilon for score in non_target_scores)
            )
            target_outscored.append(float(best_non_target > best_target))

        rows.append({
            "decile": decile,
            "query_count": len(sub),
            "scored_candidate_coverage": candidate_queries / len(sub) if len(sub) else float("nan"),
            "target_candidate_coverage": target_present / len(sub) if len(sub) else float("nan"),
            "comparable_candidate_coverage": comparable_queries / len(sub) if len(sub) else float("nan"),
            "mean_best_target_score": _mean_or_nan(best_target_scores),
            "mean_best_non_target_score": _mean_or_nan(best_non_target_scores),
            "mean_target_margin": _mean_or_nan(margins),
            "mean_near_tie_count": _mean_or_nan([float(v) for v in near_tie_counts]),
            "fraction_target_outscored": _mean_or_nan(target_outscored),
            "depth": depth,
            "epsilon": epsilon,
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


# === Lexical Factor Analysis ===


def _split_chunks(text: str, chunk_size: int | None, chunk_overlap: int) -> list[str]:
    """Split text into the same fixed-width chunks used by local BM25 indexing."""
    if not text:
        return [""]
    if not chunk_size:
        return [text]
    step = max(1, chunk_size - chunk_overlap)
    return [text[i : i + chunk_size] for i in range(0, len(text), step)]


def _mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _se_or_zero(values: list[float]) -> float:
    return float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0


def compute_bm25_lexical_factor_rows(
    query_df: pd.DataFrame,
    corpus_path: str | Path,
    *,
    decile_col: str,
    corpus_vectorizer: Any,
    chunk_size: int | None = 1000,
    chunk_overlap: int = 100,
    top_k: int = 10,
) -> pd.DataFrame:
    """Compute per-query lexical factors that can explain BM25 degradation.

    The returned rows describe the ingredients BM25 has available for the gold
    document: query-term IDF, query vocabulary coverage, query-target lexical
    overlap, target document length, and how concentrated the query evidence is
    in the best matching target chunk.

    Args:
        query_df: Retrieval results with ``question``, ``wikipedia_id``,
            ``topk_ids``, rank/recall columns, and the selected decile column.
        corpus_path: Path to the corpus Parquet with ``wikipedia_id`` and
            ``text`` columns.
        decile_col: Name of the 0-based popularity decile column.
        corpus_vectorizer: Fitted ``TfidfVectorizer`` whose vocabulary and IDF
            values define term rarity.
        chunk_size: Character chunk size matching the BM25 index. ``None`` uses
            full documents.
        chunk_overlap: Character overlap between chunks.
        top_k: Cutoff used to derive ``hit@k`` when recall columns are absent.

    Returns:
        DataFrame with one row per query/document pair and BM25 factor columns.
    """
    import pyarrow.parquet as pq
    from tqdm.auto import tqdm

    required = {"question", "wikipedia_id", decile_col}
    missing = required - set(query_df.columns)
    if missing:
        raise KeyError(f"query_df missing required columns: {sorted(missing)}")

    corpus_path = Path(corpus_path)
    needed_ids = set(str(wid).strip() for wid in query_df["wikipedia_id"].dropna())
    id_to_text: dict[str, str] = {}

    pf = pq.ParquetFile(str(corpus_path))
    for batch in tqdm(
        pf.iter_batches(batch_size=100_000, columns=["wikipedia_id", "text"]),
        desc="Scanning corpus for BM25 factors",
    ):
        batch_df = batch.to_pandas()
        batch_df["wikipedia_id"] = batch_df["wikipedia_id"].astype(str).str.strip()
        batch_df = batch_df[batch_df["wikipedia_id"].isin(needed_ids)]
        for wid, text in zip(batch_df["wikipedia_id"], batch_df["text"]):
            if wid not in id_to_text and isinstance(text, str):
                id_to_text[wid] = text
        if len(id_to_text) >= len(needed_ids):
            break

    vocabulary = corpus_vectorizer.vocabulary_
    idf_values = corpus_vectorizer.idf_
    analyzer = corpus_vectorizer.build_analyzer()
    rows: list[dict[str, Any]] = []

    for _, row in tqdm(query_df.iterrows(), total=len(query_df), desc="Computing BM25 factors"):
        gold_id = str(row["wikipedia_id"]).strip()
        text = id_to_text.get(gold_id, "")
        query_terms_all = analyzer(str(row["question"] or ""))
        query_terms = sorted({t for t in query_terms_all if t in vocabulary})
        query_oov_terms = sorted({t for t in query_terms_all if t not in vocabulary})
        query_idfs = [float(idf_values[vocabulary[t]]) for t in query_terms]
        query_idf_by_term = {t: float(idf_values[vocabulary[t]]) for t in query_terms}

        chunks = _split_chunks(text, chunk_size, chunk_overlap)
        chunk_term_counts = [Counter(analyzer(chunk)) for chunk in chunks]
        chunk_term_sets = [set(counts) for counts in chunk_term_counts]
        matching_chunks = [terms for terms in chunk_term_sets if any(t in terms for t in query_terms)]

        best_overlap_count = 0
        best_overlap_idf = 0.0
        best_overlap_frac = 0.0
        best_chunk_unique_terms = 0
        best_chunk_query_tf_sum = 0.0
        best_chunk_query_log_tf_sum = 0.0
        best_chunk_query_tfidf_sum = 0.0
        if query_terms:
            for terms, counts in zip(chunk_term_sets, chunk_term_counts):
                overlap = [t for t in query_terms if t in terms]
                overlap_count = len(overlap)
                overlap_idf = float(sum(query_idf_by_term[t] for t in overlap))
                if overlap_idf > best_overlap_idf or (
                    overlap_idf == best_overlap_idf and overlap_count > best_overlap_count
                ):
                    best_overlap_count = overlap_count
                    best_overlap_idf = overlap_idf
                    best_overlap_frac = overlap_count / len(query_terms)
                    best_chunk_unique_terms = len(terms)
                    best_chunk_query_tf_sum = float(sum(counts[t] for t in overlap))
                    best_chunk_query_log_tf_sum = float(sum(1.0 + np.log(counts[t]) for t in overlap))
                    best_chunk_query_tfidf_sum = float(
                        sum((1.0 + np.log(counts[t])) * query_idf_by_term[t] for t in overlap)
                    )

        doc_terms = set().union(*chunk_term_sets) if chunk_term_sets else set()
        doc_counts: Counter[str] = Counter()
        for counts in chunk_term_counts:
            doc_counts.update(counts)
        doc_overlap = [t for t in query_terms if t in doc_terms]
        doc_query_tf_sum = float(sum(doc_counts[t] for t in doc_overlap))
        doc_query_log_tf_sum = float(sum(1.0 + np.log(doc_counts[t]) for t in doc_overlap))
        doc_query_tfidf_sum = float(
            sum((1.0 + np.log(doc_counts[t])) * query_idf_by_term[t] for t in doc_overlap)
        )
        topk_ids = list(row.get("topk_ids", [])) if hasattr(row.get("topk_ids", []), "__iter__") else []
        topk_ids_str = [str(x) for x in topk_ids]
        hit_at_k = float(gold_id in topk_ids_str[:top_k]) if topk_ids else float("nan")
        rank = row.get("rank", float("nan"))

        rows.append({
            "question_id": row.get("question_id", None),
            "wikipedia_id": gold_id,
            "decile": int(row[decile_col]),
            "dataset": row.get("dataset", None),
            "hit_at_1": float(row.get("recall@1", topk_ids_str[:1] == [gold_id])),
            "hit_at_k": float(row.get(f"recall@{top_k}", hit_at_k)),
            "rank": float(rank) if rank is not None and not pd.isna(rank) else float("nan"),
            "query_terms_total": len(query_terms_all),
            "query_terms_in_vocab": len(query_terms),
            "query_oov_rate": len(query_oov_terms) / len(query_terms_all) if query_terms_all else float("nan"),
            "query_idf_mean": _mean_or_nan(query_idfs),
            "query_idf_sum": float(sum(query_idfs)) if query_idfs else 0.0,
            "query_idf_max": max(query_idfs) if query_idfs else float("nan"),
            "doc_length_chars": len(text),
            "n_chunks": len(chunks),
            "doc_query_overlap_count": len(doc_overlap),
            "doc_query_overlap_frac": len(doc_overlap) / len(query_terms) if query_terms else float("nan"),
            "doc_query_tf_sum": doc_query_tf_sum,
            "doc_query_log_tf_sum": doc_query_log_tf_sum,
            "doc_query_tfidf_sum": doc_query_tfidf_sum,
            "best_chunk_query_overlap_count": best_overlap_count,
            "best_chunk_query_overlap_frac": best_overlap_frac,
            "best_chunk_query_idf_sum": best_overlap_idf,
            "best_chunk_query_tf_sum": best_chunk_query_tf_sum,
            "best_chunk_query_log_tf_sum": best_chunk_query_log_tf_sum,
            "best_chunk_query_tfidf_sum": best_chunk_query_tfidf_sum,
            "matching_chunk_count": len(matching_chunks),
            "matching_chunk_frac": len(matching_chunks) / len(chunks) if chunks else float("nan"),
            "best_chunk_unique_terms": best_chunk_unique_terms,
        })

    return pd.DataFrame(rows)


def aggregate_bm25_lexical_factors(
    factor_df: pd.DataFrame,
    *,
    decile_col: str = "decile",
    success_col: str = "hit_at_1",
) -> pd.DataFrame:
    """Aggregate BM25 lexical factors by decile and retrieval outcome.

    Args:
        factor_df: Output of :func:`compute_bm25_lexical_factor_rows`.
        decile_col: Decile column in ``factor_df``.
        success_col: Binary outcome column, usually ``hit_at_1`` or
            ``hit_at_k``.

    Returns:
        Long DataFrame with one row per ``(decile, outcome)`` and means/SEs for
        each diagnostic factor.
    """
    metric_cols = [
        "query_idf_mean",
        "query_idf_sum",
        "query_oov_rate",
        "doc_length_chars",
        "n_chunks",
        "doc_query_overlap_frac",
        "doc_query_tf_sum",
        "doc_query_log_tf_sum",
        "doc_query_tfidf_sum",
        "best_chunk_query_overlap_frac",
        "best_chunk_query_idf_sum",
        "best_chunk_query_tf_sum",
        "best_chunk_query_log_tf_sum",
        "best_chunk_query_tfidf_sum",
        "matching_chunk_count",
        "matching_chunk_frac",
        "best_chunk_unique_terms",
    ]
    rows: list[dict[str, Any]] = []
    df = factor_df.copy()
    df["outcome"] = np.where(df[success_col].astype(float) >= 1.0, "hit", "miss")
    for (decile, outcome), sub in df.groupby([decile_col, "outcome"], sort=True):
        row: dict[str, Any] = {"decile": int(decile), "outcome": outcome, "count": len(sub)}
        for col in metric_cols:
            vals = sub[col].replace([np.inf, -np.inf], np.nan).dropna().astype(float).tolist()
            row[f"mean_{col}"] = _mean_or_nan(vals)
            row[f"se_{col}"] = _se_or_zero(vals)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["decile", "outcome"]).reset_index(drop=True)
