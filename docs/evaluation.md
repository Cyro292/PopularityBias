# Retrieval Evaluation

## Immutable Cohort

Use the checked QA cache directly to preserve the 60k study cohort:

```text
data/wiki_full_bil/all_qa_60k_balanced/cyro_qa_cache.parquet
```

## Retrieval

```bash
venv/bin/python -m src.process.pipeline.retrieval_runner \
  --collection wiki_full_bil \
  --output-dir all_qa_60k_balanced \
  --qa-file data/wiki_full_bil/all_qa_60k_balanced/cyro_qa_cache.parquet \
  --only-keys bm25_plus ivfpq_high \
  --top-k 10
```

Existing checkpoints resume by missing question ID. Add
`--restart-keys bm25_plus` after rebuilding BM25.

## Relevance Definitions

Gold-document recall requires a retrieved `metadata_wikipedia_id` to equal the
annotated target article ID. Answer-substring recall requires any normalized
answer alias to occur literally in a retrieved chunk. The latter can accept
useful evidence outside the gold article, but also counts incidental common
answer strings.

Generate answer-containment checkpoints:

```bash
venv/bin/python -m src.process.pipeline.retrieval_answer_eval_runner \
  --collection wiki_full_bil \
  --output-dir all_qa_60k_balanced \
  --qa-path data/wiki_full_bil/all_qa_60k_balanced/cyro_qa_cache.parquet \
  --only-keys bm25_plus ivfpq_high \
  --k-values 1 3 5 10 \
  --restart
```

Compare both definitions and their paired disagreement:

```bash
venv/bin/python -m src.process.analysis.compare_gold_and_substring_recall
```

Outputs are under
`data/wiki_full_bil/all_qa_60k_balanced/answer_eval_results/`.
