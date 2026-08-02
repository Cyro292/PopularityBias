# PopularityBias

Research code for measuring popularity bias in retrieval-augmented generation.
Wikipedia articles are grouped by pageview decile, then BM25, FAISS,
Elasticsearch, hybrid retrieval, and learned routers are compared across those
groups.

## Repository Layout

```text
PopularityBias/
├── data/                         # corpora, indices, checkpoints, and results
├── src/
│   ├── rag/                      # retrieval service implementations
│   ├── router/                   # learned router and fusion models
│   ├── metrics/                  # reusable evaluation and scoring logic
│   └── process/                  # runnable, domain-oriented workflows
│       ├── analysis/             # tables, diagnostics, and paper figures
│       ├── corpus/               # corpus and pageview preparation
│       ├── indexing/             # BM25, FAISS, and Elasticsearch indexing
│       ├── migrations/           # explicit one-off legacy data migrations
│       ├── pipeline/             # retrieval, generation, and evaluation
│       └── qa_datasets/          # QA dataset preparation
├── notebooks/                    # exploratory and evaluation notebooks
├── paper_figures/                # selected publication figures
├── tests/                        # unit and integration tests
├── config.py                     # project paths and environment configuration
└── docs/                         # detailed workflow documentation
```

There is intentionally no generic `scripts/` package. Run workflows as Python
modules from the repository root, for example
`python -m src.process.analysis.plot_analogue_similarity`.

## Setup

Python 3.11 or newer is required.

```bash
python -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Fill only the credentials needed by the backend you use. Local BM25 and FAISS
analysis does not require Elasticsearch credentials.

## Core Workflows

Build local retrieval indices:

```bash
python -m src.process.indexing.run_bm25 \
  --parquet data/wiki_full_bil/wiki_corpus.parquet \
  --output-dir data/wiki_full_bil/bm25_bm25plus \
  --method bm25+

python -m src.process.indexing.run_faiss \
  --parquet data/wiki_full_bil/wiki_corpus.parquet \
  --output-dir data/wiki_full_bil/faiss_high \
  --strategy ivfpq \
  --distance cosine
```

Prepare a balanced QA cohort:

```bash
python -m src.process.qa_datasets.prepare_qa \
  --qa-datasets natural_questions hotpot_qa trivia_qa pop_qa fever trex \
  --corpus data/wiki_full_bil/wiki_corpus.parquet \
  --output data/wiki_full_bil/all_qa_8k.parquet \
  --balance \
  --target-per-decile 800
```

Run retrieval, generation, and evaluation:

```bash
python -m src.process.pipeline.full_pipeline \
  --collection wiki_full_bil \
  --output-dir all_qa_8k \
  --only-keys bm25_plus ivfpq_high faiss_hybrid \
  --models neo qwen \
  --context-sizes 3 \
  --top-k 10
```

Build the flat analysis dataset consumed by router training and notebooks:

```bash
python -m src.process.analysis.build_analysis_dataset \
  --results-dir data/wiki_full_bil/all_qa_8k \
  --qa-parquet data/wiki_full_bil/all_qa_8k/cyro_qa_cache.parquet \
  --output data/wiki_full_bil/all_qa_8k/analysis_dataset.parquet
```

See [`docs/pipeline.md`](docs/pipeline.md) for stage-specific commands,
checkpoint behavior, indexing, and router training.

## Generate Figures

The commands below use the current 60k balanced cohort and write directly to
`paper_figures/`. Run them from the repository root.

Retrieval Recall@10 overview:

```bash
python -m src.process.analysis.plot_dataset_recall_by_decile \
  --results-dir data/wiki_full_bil/all_qa_60k_balanced \
  --qa-path data/wiki_full_bil/all_qa_60k_balanced/cyro_qa_cache.parquet \
  --metric recall --k 10 --panel both \
  --cohort-label "60k Balanced Question Pool" \
  --exclude-datasets hotpot_qa \
  --output-path paper_figures/recall10_pop.png
```

BM25+ and FAISS-high MRR, plus the FAISS-minus-BM25 delta:

```bash
python -m src.process.analysis.plot_dataset_recall_by_decile \
  --results-dir data/wiki_full_bil/all_qa_60k_balanced \
  --qa-path data/wiki_full_bil/all_qa_60k_balanced/cyro_qa_cache.parquet \
  --metric mrr --panel bm25 --exclude-datasets hotpot_qa \
  --output-path paper_figures/mrr_bm25_by_dataset_and_decile_60k_balanced.png

python -m src.process.analysis.plot_dataset_recall_by_decile \
  --results-dir data/wiki_full_bil/all_qa_60k_balanced \
  --qa-path data/wiki_full_bil/all_qa_60k_balanced/cyro_qa_cache.parquet \
  --metric mrr --panel faiss-high --exclude-datasets hotpot_qa \
  --output-path paper_figures/mrr_faiss_high_by_dataset_and_decile_60k_balanced.png

python -m src.process.analysis.plot_dataset_recall_by_decile \
  --results-dir data/wiki_full_bil/all_qa_60k_balanced \
  --qa-path data/wiki_full_bil/all_qa_60k_balanced/cyro_qa_cache.parquet \
  --figure delta --metric mrr --exclude-datasets hotpot_qa \
  --output-path paper_figures/delta_vs_bm25_retrieved_docs_ivfpq_high_mrr_60k_balanced.png
```

BM25 diagnostics, dense-retrieval heatmaps, and preference curve:

```bash
python -m src.process.analysis.plot_bm25_60k_diagnostics
python -m src.process.analysis.plot_dense_retrieval_decile_heatmaps
python -m src.process.analysis.plot_wrong_retrieval_preference_curve
```

BM25 ranking failure and analogue-pair figures:

```bash
python -m src.process.analysis.analyse_bm25_competition
python -m src.process.analysis.plot_bm25_ranking_failure
python -m src.process.analysis.plot_analogue_similarity
```

The analogue plot command requires `data/similarity_scores.parquet`. Rebuild it
after both corpora have matching BM25 and FAISS indices with:

```bash
python -m src.process.analysis.build_analogue_similarity_scores
```

Figure provenance, required inputs, and output filenames are documented in
[`paper_figures/README.md`](paper_figures/README.md).

## Tests

```bash
pytest -m "not integration"
pytest
```

Integration tests require the corresponding external services and credentials.

## Additional Documentation

- [`docs/pipeline.md`](docs/pipeline.md): complete pipeline and indexing guide
- [`docs/router_training.md`](docs/router_training.md): recorded router experiments
- [`docs/popularity_enriched_qa.md`](docs/popularity_enriched_qa.md): legacy enriched QA dataset notes
- [`AGENTS.md`](AGENTS.md): repository conventions for coding assistants
