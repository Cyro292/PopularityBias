# Indexing

Run commands from the repository root with Python 3.11 or newer. The main
corpus is `data/wiki_full_bil/wiki_corpus.parquet`; index metadata and decile
boundaries are stored in `data/wiki_full_bil/metadata.json`.

## BM25+

BM25 and FAISS use `RecursiveCharacterTextSplitter` with the same 1000/100
configuration:

```bash
venv/bin/python -m src.process.indexing.run_bm25 \
  --collection wiki_full_bil \
  --output-dir data/wiki_full_bil/bm25_bm25plus_recursive \
  --method bm25+ \
  --chunk-size 1000 \
  --chunk-overlap 100
```

The full build creates about 24.7 million chunks and requires substantial disk
space and several hours. The active retrieval backend points to
`bm25_bm25plus_recursive`.

## FAISS

Fresh local construction:

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

The checked study artifact was migrated from Elasticsearch. Its FAISS total
includes IVF training positions without corresponding document rows, so audit
the SQLite `docs` table rather than comparing `index.ntotal` directly with the
BM25 chunk count.

## Resume Behavior

- BM25 indexing rewrites its output; use a new directory until validation.
- FAISS fresh indexing accepts `--skip-rows`, but the caller must supply the
  correct source-row offset.
- Elasticsearch migration has a separate `--resume` mode and reconciliation
  logic documented in `src/process/migrations/elasticsearch_to_faiss.py`.

See [`pipeline.md`](pipeline.md) for Elasticsearch and legacy workflows.
