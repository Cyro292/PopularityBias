"""Analyse similarity scores parquet and visualise discrepancies between
new (2026) and known (old corpus) articles across BM25 and dense retrieval.

Usage:
    python scripts/analyse_similarity_scores.py
    python scripts/analyse_similarity_scores.py --input data/similarity_scores.parquet
    python scripts/analyse_similarity_scores.py --output plots/similarity_analysis.png
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

ROOT         = Path(__file__).parent.parent
DEFAULT_IN   = ROOT / "data" / "similarity_scores.parquet"
DEFAULT_OUT  = ROOT / "data" / "similarity_analysis.png"

NEW_LABEL  = "New (2026)"
OLD_LABEL  = "Known (old corpus)"

PALETTE = {NEW_LABEL: "#E05C5C", OLD_LABEL: "#4C8BBF"}


# ── Data preparation ──────────────────────────────────────────────────────────

def load_long(path: Path) -> pd.DataFrame:
    """Load the wide parquet and reshape to long format for seaborn plotting."""
    df = pd.read_parquet(path)

    new = pd.DataFrame({
        "pair_index":    range(len(df)),
        "question":      df["question_1"],
        "wiki_title":    df["wiki_title_1"],
        "bm25_score":    df["bm25_score_1"],
        "cosine_score":  df["cosine_score_1"],
        "source":        NEW_LABEL,
    })
    old = pd.DataFrame({
        "pair_index":    range(len(df)),
        "question":      df["question_2"],
        "wiki_title":    df["wiki_title_2"],
        "bm25_score":    df["bm25_score_2"],
        "cosine_score":  df["cosine_score_2"],
        "source":        OLD_LABEL,
    })
    long = pd.concat([new, old], ignore_index=True)

    # Replace 0.0 cosine/bm25 scores (article not found in top-k) with NaN
    # so they are excluded from distribution plots rather than dragging means down.
    long["bm25_score"]   = long["bm25_score"].replace(0.0, np.nan)
    long["cosine_score"] = long["cosine_score"].replace(0.0, np.nan)

    return long


def compute_deltas(path: Path) -> pd.DataFrame:
    """Return per-pair delta dataframe: score_1 - score_2 (new minus known)."""
    df = pd.read_parquet(path)
    deltas = pd.DataFrame({
        "pair_index":    range(len(df)),
        "label":         df["wiki_title_1"] + " vs " + df["wiki_title_2"],
        "delta_bm25":    df["bm25_score_1"]   - df["bm25_score_2"],
        "delta_cosine":  df["cosine_score_1"] - df["cosine_score_2"],
    })
    # Pairs where either side was 0 (not found) produce unreliable deltas — mark NaN
    raw = pd.read_parquet(path)
    mask_bm25   = (raw["bm25_score_1"] == 0) | (raw["bm25_score_2"] == 0)
    mask_cosine = (raw["cosine_score_1"] == 0) | (raw["cosine_score_2"] == 0)
    deltas.loc[mask_bm25,   "delta_bm25"]   = np.nan
    deltas.loc[mask_cosine, "delta_cosine"] = np.nan
    return deltas


# ── Statistical annotation helpers ───────────────────────────────────────────

def _ttest_annotation(a: pd.Series, b: pd.Series) -> str:
    """Return a short string with t-test p-value and effect size (Cohen's d)."""
    a = a.dropna()
    b = b.dropna()
    if len(a) < 2 or len(b) < 2:
        return "n/a"
    t, p = stats.ttest_ind(a, b)
    pooled_std = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2)
    d = (a.mean() - b.mean()) / pooled_std if pooled_std > 0 else 0.0
    p_str = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
    return f"{p_str}  |d|={abs(d):.2f}"


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot(long: pd.DataFrame, deltas: pd.DataFrame, out_path: Path) -> None:
    sns.set_theme(style="whitegrid", font_scale=1.05)
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        "Popularity Bias in RAG Retrieval\n"
        "New 2026 articles vs. Known articles — BM25 & Dense similarity scores",
        fontsize=14, fontweight="bold", y=0.98,
    )

    gs = fig.add_gridspec(3, 2, hspace=0.52, wspace=0.35)

    ax_box_bm25    = fig.add_subplot(gs[0, 0])
    ax_box_cosine  = fig.add_subplot(gs[0, 1])
    ax_kde_bm25    = fig.add_subplot(gs[1, 0])
    ax_kde_cosine  = fig.add_subplot(gs[1, 1])
    ax_delta_bm25  = fig.add_subplot(gs[2, 0])
    ax_delta_cos   = fig.add_subplot(gs[2, 1])

    # ── Row 0: paired box + strip plots ──────────────────────────────────────

    for ax, score_col, title in [
        (ax_box_bm25,   "bm25_score",   "BM25 Score Distribution"),
        (ax_box_cosine, "cosine_score", "Dense (Cosine) Score Distribution"),
    ]:
        sns.boxplot(
            data=long, x="source", y=score_col,
            hue="source", palette=PALETTE, width=0.45, linewidth=1.2,
            fliersize=0, ax=ax, legend=False,
            order=[NEW_LABEL, OLD_LABEL],
        )
        sns.stripplot(
            data=long, x="source", y=score_col,
            hue="source", palette=PALETTE, size=5, alpha=0.55, jitter=True,
            ax=ax, legend=False, order=[NEW_LABEL, OLD_LABEL],
        )

        new_vals = long.loc[long["source"] == NEW_LABEL, score_col]
        old_vals = long.loc[long["source"] == OLD_LABEL, score_col]
        ann = _ttest_annotation(new_vals, old_vals)
        ax.set_title(f"{title}\n({ann})", fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel(score_col.replace("_", " ").title())

        # Annotate means
        for i, (vals, label) in enumerate([(new_vals, NEW_LABEL), (old_vals, OLD_LABEL)]):
            m = vals.mean()
            ax.annotate(
                f"μ={m:.3f}",
                xy=(i, m), xytext=(i + 0.28, m),
                fontsize=8.5, color="black",
                va="center",
                arrowprops=dict(arrowstyle="-", color="grey", lw=0.8),
            )

    # ── Row 1: KDE overlays ───────────────────────────────────────────────────

    for ax, score_col, title in [
        (ax_kde_bm25,   "bm25_score",   "BM25 Score Density"),
        (ax_kde_cosine, "cosine_score", "Dense (Cosine) Score Density"),
    ]:
        for label, colour in PALETTE.items():
            vals = long.loc[long["source"] == label, score_col].dropna()
            if len(vals) < 2:
                continue
            sns.kdeplot(vals, ax=ax, label=label, color=colour, fill=True, alpha=0.25, linewidth=1.8)
            ax.axvline(vals.mean(), color=colour, linestyle="--", linewidth=1.2, alpha=0.8)

        ax.set_title(title, fontsize=11)
        ax.set_xlabel(score_col.replace("_", " ").title())
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)

    # ── Row 2: per-pair delta bar charts ─────────────────────────────────────

    for ax, delta_col, title, zero_line_label in [
        (ax_delta_bm25, "delta_bm25",   "Per-pair BM25 Δ  (New − Known)",   "No difference"),
        (ax_delta_cos,  "delta_cosine", "Per-pair Dense Δ (New − Known)",    "No difference"),
    ]:
        d = deltas[["label", delta_col]].dropna().sort_values(delta_col)
        colours = [PALETTE[NEW_LABEL] if v >= 0 else PALETTE[OLD_LABEL] for v in d[delta_col]]

        ax.barh(range(len(d)), d[delta_col], color=colours, edgecolor="white", linewidth=0.4)
        ax.set_yticks(range(len(d)))
        ax.set_yticklabels(d["label"], fontsize=7)
        ax.axvline(0, color="black", linewidth=0.9, linestyle="-")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Score difference")

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=PALETTE[NEW_LABEL], label="New article scores higher"),
            Patch(facecolor=PALETTE[OLD_LABEL], label="Known article scores higher"),
        ]
        ax.legend(handles=legend_elements, fontsize=8, loc="lower right")

        # Mean delta annotation
        mean_d = d[delta_col].mean()
        ax.axvline(mean_d, color="black", linewidth=1.0, linestyle=":", alpha=0.7)
        ax.annotate(
            f"mean Δ={mean_d:+.3f}",
            xy=(mean_d, len(d) - 1),
            xytext=(mean_d + (0.3 if mean_d >= 0 else -0.3), len(d) * 0.85),
            fontsize=8, ha="left" if mean_d >= 0 else "right",
            arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
        )

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved plot to {out_path}")
    print(f"Plot saved → {out_path}")


# ── Summary stats to stdout ───────────────────────────────────────────────────

def print_summary(long: pd.DataFrame, deltas: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    for score_col in ("bm25_score", "cosine_score"):
        label = "BM25" if "bm25" in score_col else "Dense (Cosine)"
        new_vals = long.loc[long["source"] == NEW_LABEL, score_col].dropna()
        old_vals = long.loc[long["source"] == OLD_LABEL, score_col].dropna()
        print(f"\n{label}  (zeros excluded as 'not found')")
        print(f"  {NEW_LABEL:<22} n={len(new_vals):2d}  mean={new_vals.mean():.4f}  std={new_vals.std():.4f}")
        print(f"  {OLD_LABEL:<22} n={len(old_vals):2d}  mean={old_vals.mean():.4f}  std={old_vals.std():.4f}")
        ann = _ttest_annotation(new_vals, old_vals)
        print(f"  t-test: {ann}")

    print(f"\nPer-pair deltas (New − Known, NaN where either side not found):")
    print(f"  BM25   mean Δ = {deltas['delta_bm25'].mean():+.4f}  "
          f"(positive = new article scored higher)")
    print(f"  Dense  mean Δ = {deltas['delta_cosine'].mean():+.4f}  "
          f"(positive = new article scored higher)")
    print("=" * 60 + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",  type=Path, default=DEFAULT_IN,  help="Path to similarity_scores.parquet")
    p.add_argument("--output", type=Path, default=DEFAULT_OUT, help="Output PNG path")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    long   = load_long(args.input)
    deltas = compute_deltas(args.input)

    print_summary(long, deltas)
    plot(long, deltas, args.output)


if __name__ == "__main__":
    main()
