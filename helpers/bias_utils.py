"""helpers/bias_utils.py — Popularity-bias analysis utilities.

Reusable helpers for analysing and visualising the popularity bias of
retrieval results across Wikipedia popularity deciles. These functions
are intentionally decoupled from any specific notebook layout and operate
only on standard NumPy arrays, Pandas DataFrames, and Matplotlib Axes.

Typical usage::

    from helpers.bias_utils import (
        build_heatmap_pct,
        pref_curve,
        render_heatmap,
        apply_transform,
        TRANSFORMS,
    )
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.axes


# ── Score-to-distance transform ───────────────────────────────────────────────

# Maps strategy name → transform mode.
# "similarity" : proximity = 1 - score  (score already in [0, 1])
# "inverse"    : proximity = 1 / (1 + score)  (score ≥ 0, lower = closer)
# "auto"       : infer from value range at runtime
TRANSFORMS: dict[str, str] = {
    "approximation": "auto",
    "bm25":          "inverse",
}


def apply_transform(strategy: str, value: Any) -> float:
    """Convert a raw retrieval score to a [0, 1] proximity value.

    The transform is chosen based on ``TRANSFORMS[strategy]``. Unknown
    strategies fall back to ``"auto"`` mode.

    Args:
        strategy: Name of the retrieval strategy (e.g. ``"bm25"``).
        value: Raw retrieval score. Must be castable to ``float``.

    Returns:
        Proximity in ``[0, 1]``, or ``np.nan`` if conversion fails.
    """
    try:
        v = float(value)
    except Exception:
        return float("nan")

    mode = TRANSFORMS.get(strategy, "auto")
    if mode == "similarity":
        return 1.0 - v
    if mode == "inverse":
        return 1.0 / (1.0 + v)
    # auto: infer from value range
    if 0.0 <= v <= 1.0:
        return 1.0 - v
    return 1.0 / (1.0 + v) if v >= 0 else float("nan")


# ── Wrong-document popularity heatmap ─────────────────────────────────────────


def build_heatmap_pct(
    df: pd.DataFrame,
    decile_col: str,
    boundaries: np.ndarray,
) -> np.ndarray:
    """Build a row-normalised 10×10 popularity-confusion heatmap.

    Each cell ``[i, j]`` represents the percentage of wrong retrieved
    documents that belong to decile *j* when the expected document
    belongs to decile *i*.

    Args:
        df: Results DataFrame. Must contain *decile_col* (0-indexed int),
            and ``wrong_docs_popularities`` (list of popularity values or
            ``None``).
        decile_col: Name of the decile column.
        boundaries: Array of decile boundary values (as returned by
            :func:`helpers.decile_utils.boundaries_for`).

    Returns:
        A ``(10, 10)`` float array where rows sum to 100 (or 0 if no wrong
        docs for that expected decile).
    """
    from helpers.decile_utils import assign_decile

    hm = np.zeros((10, 10), dtype=float)
    for _, row in df.iterrows():
        expected_dec = int(row[decile_col])
        wrong_pops = row.get("wrong_docs_popularities")
        if wrong_pops is not None and len(wrong_pops) > 0:
            for wp in wrong_pops:
                wd = int(assign_decile(wp, boundaries))
                hm[expected_dec, wd] += 1

    row_sums = hm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return hm / row_sums * 100.0


# ── Popularity preference curve ───────────────────────────────────────────────


def pref_curve(
    df: pd.DataFrame,
    decile_col: str,
    *,
    n_deciles: int = 10,
) -> tuple[list[float], list[float]]:
    """Compute the popularity preference curve with 95% CI per decile.

    For each decile, measures the fraction of wrong retrieved documents
    whose popularity is *higher* than the expected document's popularity.
    A value > 0.5 indicates a bias towards more popular documents.

    Args:
        df: Results DataFrame with *decile_col*, ``wrong_docs_popularities``,
            and the popularity column (``COL_POPULARITY`` from
            :mod:`helpers.decile_utils`).
        decile_col: Name of the decile column (0-indexed, values 0–9).
        n_deciles: Number of deciles (default 10).

    Returns:
        A 2-tuple ``(bias, ci95)`` — lists of length *n_deciles* containing
        the preference fraction and its 95% CI. Deciles with no data
        contain ``float("nan")``.
    """
    from helpers.decile_utils import COL_POPULARITY

    bias: list[float] = []
    ci95: list[float] = []

    for d in range(n_deciles):
        subset = df[df[decile_col] == d]
        n_higher = 0
        n_total  = 0

        for wp_list, pop in zip(subset["wrong_docs_popularities"], subset[COL_POPULARITY]):
            if wp_list is None or len(wp_list) == 0 or pop is None:
                continue
            for wp in wp_list:
                n_total  += 1
                if wp > pop:
                    n_higher += 1

        if n_total > 0:
            p = n_higher / n_total
            bias.append(p)
            ci95.append(1.96 * np.sqrt(p * (1 - p) / n_total))
        else:
            bias.append(float("nan"))
            ci95.append(float("nan"))

    return bias, ci95


# ── Heatmap rendering helper ──────────────────────────────────────────────────


def render_heatmap(
    ax: matplotlib.axes.Axes,
    hm_pct: np.ndarray,
    title: str,
    *,
    cmap: str = "Reds",
    vmin: float = 0.0,
    vmax: float = 60.0,
    annotate_threshold: float = 1.0,
    highlight_diagonal: bool = True,
) -> None:
    """Draw a single row-normalised popularity heatmap panel on *ax*.

    Args:
        ax: Matplotlib ``Axes`` to draw on.
        hm_pct: ``(10, 10)`` array of row-normalised percentages (as
            returned by :func:`build_heatmap_pct`).
        title: Panel title text.
        cmap: Matplotlib colormap name.
        vmin: Colormap minimum value.
        vmax: Colormap maximum value.
        annotate_threshold: Cells with absolute value above this threshold
            are annotated with their numeric value.
        highlight_diagonal: If ``True``, draw a green rectangle around each
            diagonal cell.
    """
    im = ax.imshow(hm_pct, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax, label="%", pad=0.02, shrink=0.8)
    cbar.ax.tick_params(labelsize=7)

    for i in range(10):
        for j in range(10):
            val = hm_pct[i, j]
            if abs(val) > annotate_threshold:
                colour = "white" if abs(val) > vmax * 0.5 else "black"
                ax.text(
                    j, i, f"{val:.0f}",
                    ha="center", va="center",
                    color=colour, fontsize=6, fontweight="bold",
                )

    if highlight_diagonal:
        for i in range(10):
            ax.add_patch(
                plt.Rectangle(
                    (i - 0.5, i - 0.5), 1, 1,
                    fill=False, edgecolor="limegreen", linewidth=1.8,
                )
            )

    ax.set_xticks(range(10))
    ax.set_xticklabels(range(1, 11), fontsize=7)
    ax.set_yticks(range(10))
    ax.set_yticklabels(range(1, 11), fontsize=7)
    ax.set_xlabel("Retrieved Decile",  fontsize=9, fontweight="bold")
    ax.set_ylabel("Expected Decile",   fontsize=9, fontweight="bold")
    ax.set_title(title, fontsize=9, fontweight="bold")
