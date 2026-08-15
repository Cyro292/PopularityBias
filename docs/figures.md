# Figure Reproduction Manifest

This manifest covers every external analytical graphic referenced by the thesis
under `data/thesis_latex_tum_updated`. Run commands from the repository root.

## Synchronization

Copy existing canonical artifacts into the thesis:

```bash
venv/bin/python -m src.process.analysis.sync_thesis_figures --strict
```

Regenerate maintained non-expensive figures before copying:

```bash
venv/bin/python -m src.process.analysis.sync_thesis_figures --generate --strict
```

Add `--include-expensive` to rerun live BM25 depth-100 diagnostics.

## Thesis Figures

| Thesis file | Generator | Status |
|---|---|---|
| `study_pipeline.pdf` | `latexmk -pdf -cd data/thesis_latex_tum_updated/figures/study_pipeline.tex` | Exact TikZ source |
| `mean_chunks_per_article_by_chunk_weighted_decile.png` | `notebooks/retrieval_gold_document_eval/05_bm25_analysis.ipynb` | Notebook-generated |
| `mrr_bm25_by_dataset_and_decile_60k_balanced.png` | `plot_dataset_recall_by_decile` | Script-generated |
| `mrr_faiss_high_by_dataset_and_decile_60k_balanced.png` | `plot_dataset_recall_by_decile` | Script-generated |
| `delta_vs_bm25_retrieved_docs_ivfpq_high_mrr_60k_balanced.png` | `plot_dataset_recall_by_decile` | Script-generated |
| `bm25_candidate_recall_by_decile_60k_balanced.png` | `plot_bm25_60k_diagnostics` | Script-generated, expensive without cache |
| `bm25_lexical_competition_by_decile_60k_balanced.png` | `plot_bm25_60k_diagnostics` | Script-generated, expensive without cache |
| `similarity_score_distribution.png` | `plot_analogue_similarity`, synchronized from `analogue_similarity_score_distribution.png` | Reproducible from existing scores; upstream 2026 index provenance is incomplete |
| `pref-curve.png` | `plot_wrong_retrieval_preference_curve`, synchronized from `wrong_retrieval_preference_equal_article_60k.png` | Script-generated; thesis name is legacy |
| `qwen_generation_retrieval_accuracy_by_decile.png` | `notebooks/full_pipe_eval/03_popularity_bias.ipynb` | Notebook-generated; currently configured for 8k cohort |
| `qwen_retrieval_hit_lift_by_decile.png` | same notebook | Notebook-generated; currently configured for 8k cohort |

## Exact Retrieval Commands

The synchronization module runs the maintained commands. For individual MRR
figures, see [`../paper_figures/README.md`](../paper_figures/README.md).

Preference curve:

```bash
venv/bin/python -m src.process.analysis.plot_wrong_retrieval_preference_curve
```

Analogue plots from existing scores:

```bash
venv/bin/python -m src.process.analysis.plot_analogue_similarity \
  --scores-path data/similarity_scores.parquet \
  --output-dir paper_figures
```

BM25 diagnostics:

```bash
venv/bin/python -m src.process.analysis.plot_bm25_60k_diagnostics
```

## Known Gaps

- The two Qwen figures are reproducible from a notebook, but its shared setup
  currently points to `all_qa_8k`; do not label them as 60k outputs.
- `similarity_score_distribution.png` was a legacy filename. The maintained
  plotter writes `analogue_similarity_score_distribution.png`, which the sync
  module renames for the thesis.
- `pref-curve.png` is also a legacy thesis filename. Its maintained source is
  `wrong_retrieval_preference_equal_article_60k.png`.
- Logos are static TUM template assets, not analysis outputs.
