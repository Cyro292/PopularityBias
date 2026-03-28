"""helpers/metrics.py — Retrieval evaluation metric helpers.

Provides reusable, stateless functions for computing retrieval metrics
(Recall@K, MRR, rank) and for binning/aggregating per-decile statistics.
These functions are intentionally dataset-agnostic: they operate on
DataFrames produced by any RAG backend.

Typical usage::

    from src.metrics.metrics import (
        compute_metrics,
        get_found_rank,
        get_wrong_pops,
        pick_group_col,
        binned_stats,
        decile_stats,
    )
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# ── Retrieval metric functions ────────────────────────────────────────────────


def compute_metrics(
    results_df: pd.DataFrame,
    k_values: list[int],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Compute Recall@K, MRR, median_rank and mean_rank, mutating *results_df*.

    New columns added to *results_df*: ``recall@{k}`` for each k,
    ``reciprocal_rank``, and ``rank``.

    Args:
        results_df: DataFrame with at least ``wikipedia_id`` and ``topk_ids``
            columns. ``topk_ids`` must be an iterable of retrieved doc IDs.
        k_values: List of cutoff values, e.g. ``[1, 3, 5, 10]``.

    Returns:
        A 2-tuple ``(metrics, results_df)`` where ``metrics`` is a dict with
        keys ``recall@{k}``, ``mrr``, ``median_rank``, ``mean_rank``.
    """
    metrics: dict[str, Any] = {}

    for k in k_values:
        def recall_at_k(row, _k: int = k) -> float:
            topk = row["topk_ids"][:_k] if len(row["topk_ids"]) >= _k else row["topk_ids"]
            return 1.0 if row["wikipedia_id"] in topk else 0.0

        results_df[f"recall@{k}"] = results_df.apply(recall_at_k, axis=1)
        metrics[f"recall@{k}"] = results_df[f"recall@{k}"].mean()

    def get_reciprocal_rank(row) -> float:
        try:
            topk = list(row["topk_ids"]) if hasattr(row["topk_ids"], "__iter__") else []
            rank = topk.index(row["wikipedia_id"]) + 1
            return 1.0 / rank
        except (ValueError, AttributeError, TypeError):
            return 0.0

    results_df["reciprocal_rank"] = results_df.apply(get_reciprocal_rank, axis=1)
    metrics["mrr"] = results_df["reciprocal_rank"].mean()

    def get_rank(row) -> int | None:
        try:
            topk = list(row["topk_ids"]) if hasattr(row["topk_ids"], "__iter__") else []
            return topk.index(row["wikipedia_id"]) + 1
        except (ValueError, AttributeError, TypeError):
            return None

    results_df["rank"] = results_df.apply(get_rank, axis=1)
    found_ranks = results_df["rank"].dropna()
    metrics["median_rank"] = found_ranks.median() if len(found_ranks) > 0 else None
    metrics["mean_rank"]   = found_ranks.mean()   if len(found_ranks) > 0 else None

    return metrics, results_df


def get_found_rank(row) -> int | None:
    """Return the 1-based rank of the target document in the top-K list.

    Args:
        row: A DataFrame row with ``topk_ids`` and ``wikipedia_id`` fields.

    Returns:
        1-based rank if found, ``None`` otherwise.
    """
    try:
        topk = list(row["topk_ids"]) if hasattr(row["topk_ids"], "__iter__") else []
        return topk.index(row["wikipedia_id"]) + 1
    except (ValueError, AttributeError, TypeError):
        return None


def get_wrong_pops(row) -> list[float] | None:
    """Return the popularities of all *wrong* documents in the top-K list.

    A document is wrong if its ID differs from ``row["wikipedia_id"]``.

    Args:
        row: A DataFrame row with ``topk_ids``, ``topk_popularities``, and
            ``wikipedia_id`` fields.

    Returns:
        List of popularity scores for wrong docs, or ``None`` if empty.
    """
    wrong_pops: list[float] = []
    topk_ids   = list(row["topk_ids"])          if hasattr(row["topk_ids"],          "__iter__") else []
    topk_pops  = list(row["topk_popularities"]) if hasattr(row["topk_popularities"], "__iter__") else []
    for i, doc_id in enumerate(topk_ids):
        if doc_id != row["wikipedia_id"] and i < len(topk_pops):
            if topk_pops[i] is not None:
                wrong_pops.append(topk_pops[i])
    return wrong_pops if wrong_pops else None


# ── DataFrame grouping helpers ────────────────────────────────────────────────


def pick_group_col(df: pd.DataFrame) -> str:
    """Return the first recognised dataset/source column present in *df*.

    Checks a fixed list of common column names in priority order.

    Args:
        df: DataFrame to inspect.

    Returns:
        Column name if found, empty string otherwise.
    """
    for candidate in [
        "company", "provider", "dataset_company", "dataset_source",
        "dataset_name", "dataset", "qa_dataset", "source_dataset",
    ]:
        if candidate in df.columns:
            return candidate
    return ""


# ── Binning / aggregation helpers ─────────────────────────────────────────────


def binned_stats(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    *,
    n_bins: int = 35,
    min_pts: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin (x, y) data into equal-width bins and compute mean ± 95% CI per bin.

    Bins with fewer than *min_pts* points are omitted from the output.

    Args:
        x_vals: 1-D array of x values (e.g. ``ln(popularity)``).
        y_vals: 1-D array of corresponding y values (e.g. ``recall@K``).
        n_bins: Number of equal-width bins to create over the x range.
        min_pts: Minimum number of points required to include a bin.

    Returns:
        A 3-tuple ``(centers, means, ci95)`` — all 1-D arrays of the same
        length, containing bin centre x values, per-bin means, and 95%
        confidence intervals (``1.96 * SE``).
    """
    bins = np.linspace(x_vals.min(), x_vals.max(), n_bins + 1)
    centers: list[float] = []
    means:   list[float] = []
    ci:      list[float] = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (x_vals >= lo) & (x_vals <= hi)
        n = int(mask.sum())
        if n < min_pts:
            continue
        y = y_vals[mask]
        m  = float(y.mean())
        se = float(y.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        centers.append((lo + hi) / 2)
        means.append(m)
        ci.append(1.96 * se)
    return np.array(centers), np.array(means), np.array(ci)


def decile_stats(
    df: pd.DataFrame,
    metric_col: str,
    decile_col: str,
    *,
    group_col: str = "",
    n_deciles: int = 10,
) -> dict[str | None, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Compute per-decile mean ± 95% CI for a metric, optionally split by group.

    Args:
        df: Results DataFrame.
        metric_col: Name of the metric column (e.g. ``"recall@10"``).
        decile_col: Name of the decile column (0-indexed, values 0–9).
        group_col: Optional column to split results by (e.g. ``"dataset"``).
            If empty or not in ``df``, the whole DataFrame is treated as a
            single group keyed by ``None``.
        n_deciles: Number of deciles (default 10).

    Returns:
        Dict mapping group label → ``(deciles_1based, means, ci95)`` arrays.
        Deciles with no data have ``np.nan`` in means/ci95.
    """
    has_group = bool(group_col) and group_col in df.columns and df[group_col].nunique() > 1
    groups = sorted(df[group_col].dropna().unique()) if has_group else [None]

    out: dict[str | None, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for grp in groups:
        sub = df[df[group_col] == grp] if has_group else df
        decile_arr: list[int]   = []
        mean_arr:   list[float] = []
        ci_arr:     list[float] = []
        for d in range(n_deciles):
            vals = sub.loc[sub[decile_col] == d, metric_col].dropna()
            n = len(vals)
            decile_arr.append(d + 1)
            if n == 0:
                mean_arr.append(float("nan"))
                ci_arr.append(float("nan"))
            else:
                m  = float(vals.mean())
                se = float(vals.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
                mean_arr.append(m)
                ci_arr.append(1.96 * se)
        out[grp] = (np.array(decile_arr), np.array(mean_arr), np.array(ci_arr))
    return out
