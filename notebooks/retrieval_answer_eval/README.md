# Retrieval Answer Evaluation

This analysis scores retrieval by whether a retrieved chunk contains any gold
answer alias. It does not require the retrieved chunk to come from the annotated
gold Wikipedia document.

## Notebook workflow

1. `01_paper_figures.ipynb`: generate checkpoints and concise paper figures.
2. `02_setup_and_distribution.ipynb`: inspect evaluability and the shared cohort.
3. `03_retrieval_performance.ipynb`: compare Answer Recall@K, Answer MRR, decile trends, and paired backend deltas.
4. `04_answer_match_bias.ipynb`: inspect answer rank, backend disagreement, and answer-alias length effects.
5. `05_answer_chunk_preference.ipynb`: count answer-bearing chunks across popularity and plot the substring-defined non-answer preference curve.
6. `06_gold_document_vs_substring_recall.ipynb`: compare BM25+ and FAISS-high Recall@K under gold-document and answer-substring relevance definitions.

The analysis notebooks exclude non-evaluable questions plus `hotpot_qa` and
`trex`, matching the legacy gold-document cohort. Outputs are isolated under
`data/wiki_full_bil/all_qa_8k/answer_eval_results`.

Generate the per-question evaluation checkpoints:

```bash
python -m src.process.pipeline.retrieval_answer_eval_runner \
  --only-keys bm25_plus ivfpq_high
```

Generate the concise figures used for paper analysis:

```bash
python -m src.process.analysis.plot_answer_retrieval_by_decile
```

Use `--metric recall --k 5` for answer Recall@5 instead of answer MRR. The
notebook in this directory is intentionally a thin wrapper around these scripts
so evaluation and plotting logic remain reproducible outside Jupyter.

Retrieved-document popularity and score diagnostics are intentionally omitted:
the answer checkpoints do not contain per-rank IDs, scores, or popularities, and
a non-gold document can be correct under answer containment.
