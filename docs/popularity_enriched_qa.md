# Popularity-Enriched QA Datasets

This notebook joins PopQA, Natural Questions, and TriviaQA with Wikipedia popularity metrics
and serializes the enriched QA pairs for downstream retrieval experiments.

## Outputs

- popqa_with_popularity.parquet
- nq_with_popularity.parquet
- triviaqa_with_popularity.parquet

## Subsets

- PopQA (test split of akariasai/PopQA)
- Natural Questions (validation split 'nq' of facebook/kilt_tasks)
- TriviaQA (validation split 'triviaqa_support_only' of facebook/kilt_tasks)

## HuggingFace Upload

The generated parquet files are pushed to Cyro1/popularity-enriched-qa-datasets using
`HUGGINGFACE_TOKEN` credentials provided to this notebook.
