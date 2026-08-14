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

## Reproduce the Retrieval Study

Run all commands from the repository root. The main study expects this artifact
layout:

```text
data/wiki_full_bil/
├── wiki_corpus.parquet
├── metadata.json
├── bm25_bm25plus_recursive/
├── faiss_high/
└── all_qa_60k_balanced/
    └── cyro_qa_cache.parquet
```

`wiki_corpus.parquet` must contain `text`, `wikipedia_id`,
`wikipedia_title`, and `popularity_avg`. The checked 60k cohort should be
treated as an immutable input: rebuilding a balanced cohort can select a
different set of questions when source datasets or cache state change.

### 1. Build BM25+

BM25 and FAISS now use the same LangChain `RecursiveCharacterTextSplitter`
configuration. The current paper configuration is 1,000 characters with 100
characters of overlap:

```bash
venv/bin/python -m src.process.indexing.run_bm25 \
  --collection wiki_full_bil \
  --output-dir data/wiki_full_bil/bm25_bm25plus_recursive \
  --method bm25+ \
  --chunk-size 1000 \
  --chunk-overlap 100
```

The full corpus build is expensive. It writes `corpus.jsonl`, an mmap index,
and index parameters under `bm25_bm25plus_recursive/`. A completed build can
be loaded without rebuilding.

### 2. Build FAISS

To create a fresh local IVF-PQ index with matching chunk boundaries:

```bash
venv/bin/python -m src.process.indexing.run_faiss \
  --collection wiki_full_bil \
  --parquet data/wiki_full_bil/wiki_corpus.parquet \
  --output-dir data/wiki_full_bil/faiss_high \
  --strategy ivfpq \
  --distance cosine \
  --chunk-size 1000 \
  --chunk-overlap 100
```

The existing `faiss_high` study artifact was migrated from Elasticsearch and
contains IVF training-vector mappings in addition to document mappings.
Consequently, FAISS `index.ntotal` is not directly comparable to the unique
document count in `docstore.sqlite`. Use the document store when auditing
corpus chunk counts. Migration and resume details are in
[`docs/pipeline.md`](docs/pipeline.md).

### 3. Generate Retrieval Checkpoints

Use the local QA file to preserve the exact cohort. Always specify
`--only-keys`; otherwise the runner attempts every configured experimental
backend.

```bash
venv/bin/python -m src.process.pipeline.retrieval_runner \
  --collection wiki_full_bil \
  --output-dir all_qa_60k_balanced \
  --qa-file data/wiki_full_bil/all_qa_60k_balanced/cyro_qa_cache.parquet \
  --only-keys bm25_plus ivfpq_high \
  --top-k 10
```

Outputs:

```text
data/wiki_full_bil/all_qa_60k_balanced/
├── retrieved_docs_bm25_plus.csv
├── retrieved_docs_ivfpq_high.csv
├── latency_retrieval_bm25_plus.json
└── latency_retrieval_ivfpq_high.json
```

Existing checkpoints resume by missing question ID. To overwrite one backend
after rebuilding its index, add `--restart-keys bm25_plus` or
`--restart-keys ivfpq_high`.

### 4. Evaluate Answer Containment

Gold-answer substring metrics are computed separately from gold-document ID
metrics:

```bash
venv/bin/python -m src.process.pipeline.retrieval_answer_eval_runner \
  --collection wiki_full_bil \
  --output-dir all_qa_60k_balanced \
  --qa-path data/wiki_full_bil/all_qa_60k_balanced/cyro_qa_cache.parquet \
  --only-keys bm25_plus ivfpq_high \
  --k-values 1 3 5 10 \
  --restart
```

This writes `retrieval_answer_eval_<backend>.parquet` and
`retrieval_answer_eval_summary.csv`. Substring recall means that any answer
alias occurs literally in a retrieved chunk; it does not require the chunk to
come from the annotated article and is sensitive to common answer strings.

### 5. Build the Analysis Dataset

```bash
venv/bin/python -m src.process.analysis.build_analysis_dataset \
  --results-dir data/wiki_full_bil/all_qa_60k_balanced \
  --qa-parquet data/wiki_full_bil/all_qa_60k_balanced/cyro_qa_cache.parquet \
  --output data/wiki_full_bil/all_qa_60k_balanced/analysis_dataset.parquet
```

## Generate Paper Figures

The following commands use the 60k cohort. Most write to `paper_figures/`; the
gold-document versus substring comparison writes to the cohort's
`answer_eval_results/` directory.

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

Gold-document versus answer-substring Recall@5 and Recall@10:

```bash
venv/bin/python -m src.process.analysis.compare_gold_and_substring_recall
```

Outputs:

```text
data/wiki_full_bil/all_qa_60k_balanced/answer_eval_results/
├── gold_document_vs_substring_disagreement_by_decile.csv
├── gold_document_vs_substring_disagreement_by_decile.png
├── gold_document_vs_substring_recall_by_decile.csv
└── gold_document_vs_substring_recall_by_decile.png
```

The matching reproducible notebook is
[`notebooks/retrieval_answer_eval/06_gold_document_vs_substring_recall.ipynb`](notebooks/retrieval_answer_eval/06_gold_document_vs_substring_recall.ipynb).

Answer-containment Recall or MRR by dataset and decile:

```bash
venv/bin/python -m src.process.analysis.plot_answer_retrieval_by_decile \
  --results-dir data/wiki_full_bil/all_qa_60k_balanced \
  --output-dir paper_figures \
  --backends bm25_plus ivfpq_high \
  --metric recall --k 10 \
  --decile-col pop_decile_unweighted \
  --exclude-datasets fever hotpot_qa trex trivia_qa \
  --cohort-label "60k Balanced Question Pool"
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

Some diagnostics have expensive producer stages. In particular,
`analyse_retrieval_neighborhood_density` performs live top-100 retrieval, and
`build_analogue_similarity_scores` requires matching indices for both corpus
versions. Existing parquet checkpoints are reused where supported.

## Optional Generation

Generation is not required for retrieval recall, MRR, gold-document recall, or
answer-substring recall. It requires the configured Modal/API model services.
See [`docs/pipeline.md`](docs/pipeline.md) before running
`src.process.pipeline.full_pipeline`; the orchestrator loads the cached cohort
from the selected output directory and does not accept `--qa-file`.

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
