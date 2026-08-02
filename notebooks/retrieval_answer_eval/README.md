# Retrieval Answer Evaluation

This analysis scores retrieval by whether a retrieved chunk contains any gold
answer alias. It does not require the retrieved chunk to come from the annotated
gold Wikipedia document.

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
