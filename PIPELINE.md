# PopularityBias Pipeline

End-to-end pipeline for evaluating popularity bias in RAG systems. Wikipedia articles are
stratified by pageview popularity (deciles), and retrieval quality is measured across
backends (BM25, FAISS, Elasticsearch, neural router).

## Quick Start

```bash
# 1. Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in API keys

# 2. Build indices (one-time)
python -m src.rag.bm25_rag_service    # see indexing section below
python -m src.rag.faiss_rag_service   # see indexing section below

# 3. Run the full pipeline (retrieval -> generation -> evaluation)
python -m src.process.pipeline.full_pipeline -c wiki_full_bil -o all_qa_8k

# 4. Build analysis dataset from results
python -m scripts.build_analysis_dataset

# 5. (Optional) Train a neural router
python -m src.router.train_router --label-mode retrieval --retrieval-metric mrr
```

---

## Environment Setup

### Requirements

- **Python 3.11+**
- **16 GB+ RAM** (corpus is streamed in batches; indices use memory-mapped files)

### Install

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Environment Variables (`.env`)

```
OPENAI_API_KEY=...
HUGGINGFACE_TOKEN=...
GOOGLE_API_KEY=...               # optional, for Google embeddings
ELASTICSEARCH_ENDPOINT=...       # optional, for ES backends
ELASTICSEARCH_USERNAME=...
ELASTICSEARCH_PASSWORD=...
ELASTICSEARCH_API_KEY=...
```

### Local Elasticsearch (optional)

```bash
cd elastic-start-local && bash start.sh
```

---

## Data Layout

All data lives under `data/` (gitignored). Key paths:

```
data/
  wiki_full_bil/                  # Collection directory
    wiki_corpus.parquet           # Source corpus (wikipedia_id, text, popularity_avg, ...)
    metadata.json                 # Decile boundaries, TF-IDF stats
    all_qa_8k/                    # Output directory for pipeline runs
      cyro_qa_cache.parquet      # Cached/balanced QA dataset
      retrieved_docs_*.csv       # Stage 1 output (per backend)
      answer_checkpoint_*.csv    # Stage 2 output (per model x backend x ctx)
      results_*.parquet          # Stage 3 output (final eval results)
      latency_*.json             # Per-stage latency measurements
      analysis_dataset.parquet   # Flat analysis dataset for notebooks/router training
    bm25_lucene/                 # BM25 index (lucene method)
    bm25_bm25plus/               # BM25 index (bm25+ method)
    faiss_high/                  # FAISS IVF-PQ index (high popularity)
    faiss_low/                   # FAISS IVF-PQ index (low popularity)
  wiki_2026/                     # 2026 Wikipedia corpus (for drift analysis)
models/                          # Trained router models (*.pt)
```

---

## Pipeline Stages

### Stage 1: Retrieval

Retrieves documents for every question using configured backends. Writes
`retrieved_docs_<key>.csv` checkpoints.

```bash
python -m src.process.pipeline.retrieval_runner
```

#### CLI Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `-c`, `--collection` | `wiki_full_bil` | Collection folder name under `data/` |
| `-o`, `--output-dir` | `all_qa_8k` | Output subdirectory |
| `--top-k` | `10` | Documents retrieved per question |
| `--questions-per-decile` | `800` | Questions sampled per decile (`-1` = all) |
| `--restart` | `False` | Overwrite all existing checkpoints |
| `--restart-keys` | `[]` | Overwrite only these backend keys |
| `--only-keys` | `[]` | Run only these backends |
| `--datasets` | all 6 | QA dataset names from HuggingFace |
| `--qa-file` | `None` | Local QA parquet (skips HuggingFace download) |

#### Example

```bash
# Run only BM25 and FAISS backends
python -m src.process.pipeline.retrieval_runner \
    --only-keys bm25_plus ivfpq_high \
    --top-k 20

# Restart only a specific backend
python -m src.process.pipeline.retrieval_runner \
    --restart-keys bm25_plus

# Use a local QA file instead of HuggingFace
python -m src.process.pipeline.retrieval_runner \
    --qa-file data/wiki_full_bil/all_qa_8k/cyro_qa_cache.parquet
```

#### Available Backend Types

| Type | Description | Required Fields |
|------|-------------|-----------------|
| `zero_shot` | No retrieval (baseline) | — |
| `elasticsearch` | ES dense/BM25/hybrid | `es_strategy` |
| `faiss` | Local FAISS vector index | `index_path` |
| `bm25` | Local bm25s index | `index_path` |
| `router` | TorchScript popularity router | `router_sub_keys` |
| `hybrid_faiss` | Dense + sparse RRF fusion | `router_sub_keys` |
| `neural_router` | BERT neural router | `router_sub_keys`, `service_kwargs` |

#### Configuring Backends

Backends are declared in the `RetrievalConfig.backends` list inside
`retrieval_runner.py`. Each entry is a `RetrievalBackend`:

```python
RetrievalBackend(
    key             = "bm25_plus",
    label           = "Sparse Retrieval (BM25 plus)",
    type            = "bm25",
    index_path      = DATA_DIR / "wiki_full_bil" / "bm25_bm25plus",
)

RetrievalBackend(
    key             = "ivfpq_high",
    label           = "Dense Retrieval (FAISS high-pop ivfpq)",
    type            = "faiss",
    index_path      = DATA_DIR / "wiki_full_bil" / "faiss_high",
    service_kwargs  = {"ivfpq_nprobe": 256},
)

RetrievalBackend(
    key             = "neural_router_strict",
    label           = "Neural Router (Strict - Argmax)",
    type            = "neural_router",
    router_sub_keys = ("bm25_plus", "ivfpq_high"),
    service_kwargs  = {
        "model_path":    "models/router_mrr20.pt",
        "backend_order": ["bm25_plus", "ivfpq_high"],
        "strict":        True,
    },
)

RetrievalBackend(
    key             = "neural_router_hybrid",
    label           = "Neural Router (Hybrid - Probability Weighted RRF)",
    type            = "neural_router",
    router_sub_keys = ("bm25_plus", "ivfpq_high"),
    service_kwargs  = {
        "model_path":    "models/router_mrr20.pt",
        "backend_order": ["bm25_plus", "ivfpq_high"],
        "strict":        False,
        "rrf_k":         60,
        "rrf_depth":     60,
    },
)
```

**Important:** For `router`, `hybrid_faiss`, and `neural_router` types, sub-backends
referenced in `router_sub_keys` must appear **earlier** in the backends list.

#### Checkpoint & Resume

- Existing `retrieved_docs_<key>.csv` files are reused automatically.
- The runner diffs against existing checkpoints and only retrieves missing question IDs.
- Use `--restart` to wipe all, or `--restart-keys` for specific backends.

---

### Stage 2: Generation

Generates answers using LLMs, given retrieved documents from Stage 1.
Writes `answer_checkpoint_<llm>_<key>_top<n>.csv`.

```bash
python -m src.process.pipeline.generating_runner
```

#### CLI Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `-c`, `--collection` | `wiki_full_bil` | Collection folder |
| `-o`, `--output-dir` | `all_qa_8k` | Output subdirectory |
| `--top-k` | `10` | Docs retrieved per question |
| `--questions-per-decile` | `800` | Questions per decile |
| `--restart` | `False` | Delete existing checkpoints and regenerate |

#### LLM Backends

Configured in `GeneratingConfig.models` inside `generating_runner.py`:

| Key | Type | Model | Description |
|-----|------|-------|-------------|
| `neo` | `neo` | EleutherAI/gpt-neo-2.7B | Modal GPU (H100) |
| `qwen` | `qwen` | Qwen2.5-7B-Instruct | Modal GPU (H100) |
| `mistral` | `mistral` | Mistral-7B-Instruct-v0.2 | Modal GPU (H100) |
| `openai` | `openai` | (configurable) | OpenAI API |

#### Prompt Template

```python
prompt_template = "Documents: {documents}\n \n \n Question: {question}"
```

Placeholders: `{question}`, `{documents}`, `{dataset}`.

`context_sizes` controls how many retrieved documents are fed to the LLM (default: `[3]`).

---

### Stage 3: Evaluation

Evaluates generated answers against ground truth. Writes
`results_<llm>_<key>_<ctx_label>_<evaluator>.parquet`.

```bash
python -m src.process.pipeline.llm_eval_runner
```

#### CLI Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `-c`, `--collection` | `wiki_full_bil` | Collection folder |
| `-o`, `--output-dir` | `all_qa_8k` | Output subdirectory |
| `--top-k` | `10` | Docs per question |
| `--questions-per-decile` | `800` | Questions per decile |
| `--restart` | `False` | Re-evaluate everything |

#### Evaluator Types

| Type | Description | LLM Required? |
|------|-------------|---------------|
| `substring` | Exact substring match | No |
| `binary` | LLM judge (yes/no) | Yes |

Example evaluator configuration:

```python
EvalBackend(key="substring", type="substring"),
EvalBackend(key="binary_mistral", type="binary", llm_type="mistral"),
EvalBackend(key="binary_gpt4", type="binary", llm_type="openai", llm_model_name="gpt-4o"),
```

---

### Full Pipeline (Orchestrator)

Runs all three stages in sequence:

```bash
python -m src.process.pipeline.full_pipeline
```

#### CLI Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--only-keys` | all default backends | Backend keys to run through all stages |
| `--models` | `neo qwen` | LLM keys for generation and eval |
| `--context-sizes` | `3` | Number of docs fed to LLM |
| `-c`, `--collection` | `wiki_full_bil` | Collection name |
| `-o`, `--output-dir` | `all_qa_8k` | Output subdirectory |
| `--top-k` | `10` | Docs per question |
| `--questions-per-decile` | `800` | Questions per decile |
| `--skip-retrieval` | `False` | Skip Stage 1 |
| `--skip-generation` | `False` | Skip Stage 2 |
| `--skip-eval` | `False` | Skip Stage 3 |
| `--restart` | `False` | Wipe all three stages |
| `--restart-retrieval` | `False` | Wipe Stage 1 only |
| `--restart-generation` | `False` | Wipe Stage 2 only |
| `--restart-eval` | `False` | Wipe Stage 3 only |
| `--restart-retrieval-keys` | `[]` | Wipe specific backend checkpoints |

#### Example

```bash
# Run everything for BM25 + FAISS with Qwen
python -m src.process.pipeline.full_pipeline \
    --only-keys bm25_plus ivfpq_high \
    --models qwen

# Skip retrieval (already done), regenerate answers
python -m src.process.pipeline.full_pipeline \
    --skip-retrieval \
    --restart-generation

# Re-run only one retrieval backend, then generate + eval
python -m src.process.pipeline.full_pipeline \
    --restart-retrieval-keys bm25_plus \
    --only-keys bm25_plus
```

---

## Building Indices

### BM25 Index (local bm25s)

```python
from src.rag.bm25_rag_service import BM25RagService

service = BM25RagService(chunk=True, chunk_size=1000, chunk_overlap=100, method="lucene")
service.index_from_parquet(
    parquet_path="data/wiki_full_bil/wiki_corpus.parquet",
    output_dir="data/wiki_full_bil/bm25_lucene",
)
```

**BM25 methods:** `"lucene"` (default), `"bm25+"`, `"atire"`, `"robertson"`, `"bm25l"`

The index is built with a two-pass streaming strategy (minimal RAM). After building,
it is reloaded with `mmap=True` so score arrays stay on disk.

### FAISS Index (local)

```python
from src.rag.faiss_rag_service import FaissRagService
from src.rag.utils import IndexingConfig

config = IndexingConfig(
    embedding_provider="huggingface",
    embedding_model="Lajavaness/bilingual-embedding-small",
    chunk_size=1000,
    chunk_overlap=200,
)
service = FaissRagService(config=config, strategy="ivfpq", ivfpq_nprobe=256)
service.index_from_parquet(
    parquet_path="data/wiki_full_bil/wiki_corpus.parquet",
    output_dir="data/wiki_full_bil/faiss_high",
)
```

**FAISS strategies:** `"ivfpq"` (default), `"vector"` (flat), `"hnsw"`, `"opq_ivfpq"`, `"ivfpq_disk"`

**Key FAISS parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ivfpq_nlist` | 4096 | IVF Voronoi cells |
| `ivfpq_m` | 48 | PQ sub-quantisers |
| `ivfpq_nbits` | 8 | Bits per PQ code |
| `ivfpq_nprobe` | 64 | Cells probed per query |
| `hnsw_m` | 32 | HNSW graph connectivity |
| `hnsw_ef_construction` | 200 | HNSW build-time search width |
| `hnsw_ef_search` | 128 | HNSW query-time search width |

### Elasticsearch Index

Requires a live Elasticsearch instance. Indexing is done via
`ElasticsearchRagService.index_from_parquet()`.

### Tuning FAISS Index

```bash
# Change nprobe (instant, no rebuild)
python -m scripts.tune_faiss_index --nprobe 256

# Rebuild with different PQ parameters (slow)
python -m scripts.tune_faiss_index --m 48 --nbits 12 --rebuild

# Benchmark recall@k
python -m scripts.tune_faiss_index --benchmark --k 10
```

---

## Neural Router

The neural router is a BERT-based classifier that routes queries to the best
retrieval backend based on query text and document popularity.

### Prerequisites

Before training, you need an **analysis dataset** — a flat parquet with per-question
retrieval/generation performance across backends. This is produced by running the
full pipeline (Stages 1-3) and then building the analysis dataset:

```bash
python -m scripts.build_analysis_dataset
```

### Training

```bash
python -m src.router.train_router
```

#### CLI Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--collection` | `wiki_full_bil` | Collection folder |
| `--dataset-dir` | `all_qa_8k` | Dataset directory |
| `--model-name` | `router_v1` | Output: `models/<name>.pt` |
| `--backends` | `bm25_plus ivfpq_high` | Backends to train on |
| `--eval-backends` | (same + zero_shot, faiss_hybrid) | Extra backends for eval |
| `--exclude-datasets` | `[]` | Datasets to exclude (e.g., `fever`) |
| `--llm` | `neo` | LLM filter (answer mode only) |
| `--label-mode` | `answer` | **`answer`** (binary, needs generation) or **`retrieval`** (MRR/Recall, no generation needed) |
| `--retrieval-metric` | `mrr` | `mrr` or `recall` (retrieval mode only) |
| `--retrieval-k` | `20` | Top-k for retrieval metric |
| `--epochs` | `80` | Training epochs |
| `--batch-size` | `32` | Batch size |
| `--lr` | `0.001` | Classifier learning rate |
| `--unfreeze-layers` | `0` | BERT layers to unfreeze from end (0 = all frozen) |
| `--bert-lr` | `2e-5` | BERT learning rate (if layers unfrozen) |

#### Label Modes

| Mode | Labels | Requires | Description |
|------|--------|----------|-------------|
| `answer` | Binary 0/1 from generation performance | Full pipeline (Stages 1-3) | `1.0` if backend produced correct answer, `0.0` if not |
| `retrieval` | Continuous from MRR/Recall@k | Stage 1 only | Soft labels normalized to sum to 1. E.g., BM25 MRR=0.33, FAISS MRR=1.0 → `[0.248, 0.752]` |

**Recommendation:** Use `--label-mode retrieval` to train before running expensive generation.

#### Training Examples

```bash
# Train with retrieval labels (fastest — no generation needed)
python -m src.router.train_router \
    --label-mode retrieval \
    --retrieval-metric mrr \
    --retrieval-k 20 \
    --backends bm25_plus ivfpq_high \
    --model-name router_mrr20 \
    --epochs 80

# Train with answer labels (needs full pipeline results)
python -m src.router.train_router \
    --label-mode answer \
    --llm neo \
    --backends bm25_plus ivfpq_high \
    --model-name router_answer_neo \
    --epochs 80

# Fine-tune BERT layers
python -m src.router.train_router \
    --label-mode retrieval \
    --unfreeze-layers 2 \
    --bert-lr 2e-5 \
    --epochs 40

# Train on 3 backends
python -m src.router.train_router \
    --label-mode retrieval \
    --backends bm25_plus ivfpq_high es_approx \
    --model-name router_3way
```

Training runs on **Modal GPU (H100)**. The trained model is saved to `models/<name>.pt`.

### Using the Neural Router in Retrieval

Add a `neural_router` backend to the `RetrievalConfig.backends` list in
`retrieval_runner.py`:

```python
# Strict mode: argmax selects single backend
RetrievalBackend(
    key             = "neural_router_strict",
    label           = "Neural Router (Strict)",
    type            = "neural_router",
    router_sub_keys = ("bm25_plus", "ivfpq_high"),
    service_kwargs  = {
        "model_path":    "models/router_mrr20.pt",
        "backend_order": ["bm25_plus", "ivvpq_high"],
        "strict":        True,
    },
)

# Hybrid mode: probability-weighted RRF fusion
RetrievalBackend(
    key             = "neural_router_hybrid",
    label           = "Neural Router (Hybrid RRF)",
    type            = "neural_router",
    router_sub_keys = ("bm25_plus", "ivvpq_high"),
    service_kwargs  = {
        "model_path":    "models/router_mrr20.pt",
        "backend_order": ["bm25_plus", "ivvpq_high"],
        "strict":        False,
        "rrf_k":         60,
        "rrf_depth":     60,
    },
)
```

**NeuralRouterRagService parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `backends` | required | Dict mapping backend names to loaded `RagService` instances |
| `backend_order` | required | Backend names in model-training order |
| `model_path` | required | Path to trained `.pt` model |
| `strict` | `True` | `True` = argmax (single backend), `False` = probability-weighted RRF |
| `rrf_k` | `60` | RRF smoothing constant (hybrid mode) |
| `rrf_depth` | `60` | Candidates per backend before fusion (hybrid mode) |
| `predict_batch_size` | `32` | BERT inference batch size |
| `device` | `"cpu"` | Torch device for inference |

**Strict mode:** Routes each query to a single backend (argmax of logits). Groups
queries by predicted backend for efficient batch retrieval.

**Hybrid mode:** Retrieves `rrf_depth` candidates from **all** backends, then fuses
with probability-weighted RRF: `score(doc) = sum(prob_backend * 1/(rrf_k + rank))`.

### Using Programmatically

```python
from src.rag import NeuralRouterRagService
from src.rag.bm25_rag_service import BM25RagService
from src.rag.faiss_rag_service import FaissRagService

# Load sub-backends
bm25 = BM25RagService()
bm25.load_index("data/wiki_full_bil/bm25_bm25plus")

faiss = FaissRagService(config=...)
faiss.load_index("data/wiki_full_bil/faiss_high")

# Create router
router = NeuralRouterRagService(
    backends={"bm25_plus": bm25, "ivfpq_high": faiss},
    backend_order=["bm25_plus", "ivvpq_high"],
    model_path="models/router_mrr20.pt",
    strict=True,
)

# Retrieve
docs = router.retrieve_documents(
    "What is the capital of France?",
    top_k=10,
    popularity=1500.0,
)
```

---

## QA Datasets

### Source Datasets

Questions are loaded from the HuggingFace repository
`Cyro1/popularity-enriched-qa-datasets` with 6 sub-datasets:

| Name | Source | Split |
|------|--------|-------|
| `natural_questions` | Facebook KILT NQ | validation |
| `hotpot_qa` | Facebook KILT HotpotQA | validation |
| `trivia_qa` | Facebook KILT TriviaQA | validation |
| `pop_qa` | Akariasai/PopQA | test |
| `fever` | Facebook KILT FEVER | validation |
| `trex` | Facebook KILT T-REX | validation |

### Decile Balancing

Questions are balanced across popularity deciles (0-9) so each decile has equal
representation. Two decile flavours:

- **Unweighted:** 1 document = 1 vote (equal document distribution)
- **Chunk-weighted:** 1 chunk = 1 vote (popular long articles get more chunks)

Controlled by `--questions-per-decile` and `balance_decile_mode` in RetrievalConfig.

---

## Analysis & Utilities

### Build Analysis Dataset

Aggregates all pipeline result parquets into a single flat analysis dataset for
notebooks and router training:

```bash
python -m scripts.build_analysis_dataset
```

| Flag | Default | Description |
|------|---------|-------------|
| `--results-dir` | auto | Override results directory |
| `--output` | auto | Override output parquet path |
| `--qa-parquet` | auto | QA parquet for split reconstruction |
| `--llms` | all | Filter to specific LLMs |
| `--backends` | all | Filter to specific backends |
| `--ctx-labels` | all | Filter to specific context labels |
| `--eval-types` | all | Filter to specific evaluator types |

Output: `data/wiki_full_bil/analysis_dataset.parquet`

### FAISS Index Tuning

```bash
# Change nprobe (instant)
python -m scripts.tune_faiss_index --nprobe 256

# Rebuild index structure (slow)
python -m scripts.tune_faiss_index --m 48 --nbits 12 --nlist 8192 --rebuild

# Dry run (preview changes)
python -m scripts.tune_faiss_index --m 48 --rebuild --dry-run

# Benchmark recall
python -m scripts.tune_faiss_index --benchmark --k 10

# Benchmark with real QA questions
python -m scripts.tune_faiss_index --benchmark-qa --n-questions 500
```

### Export Elasticsearch to Parquet

```bash
python -m scripts.export_es_to_parquet \
    --output data/wiki_full_bil/chunks_with_vectors \
    --batch-size 2000 \
    --shard-size 500000
```

### Backfill Decile Columns

```bash
# Into parquet files
python -m scripts.backfill_decile_columns
python -m scripts.backfill_decile_columns --dry-run

# Into CSV metadata blobs
python -m scripts.backfill_csv_decile_columns
```

### 2026 Wikipedia Corpus

```bash
# Build from XML dump + pageviews
python -m scripts.build_wiki_2026_corpus

# Clean MediaWiki markup
python -m scripts.clean_wiki_2026_corpus

# Convert raw pageviews to parquet
python -m scripts.popularity_parquet_script

# Find top new articles absent from old corpus
python -m scripts.top_new_2026_articles

# Analyze similarity scores (new vs old articles)
python -m scripts.build_similarity_scores
python -m scripts.analyse_similarity_scores
```

---

## RAG Service Reference

### BM25RagService

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk` | `True` | Split articles into chunks |
| `chunk_size` | `1000` | Max chars per chunk |
| `chunk_overlap` | `100` | Overlap between chunks |
| `k1` | `1.5` | BM25 term-saturation |
| `b` | `0.75` | BM25 length-normalisation |
| `method` | `"lucene"` | Scoring: `lucene`, `bm25+`, `atire`, `robertson`, `bm25l` |

### FaissRagService

| Parameter | Default | Description |
|-----------|---------|-------------|
| `config` | required | `IndexingConfig` for embeddings |
| `strategy` | `"ivfpq"` | Index type: `vector`, `hnsw`, `ivfpq`, `opq_ivfpq`, `ivfpq_disk` |
| `distance_strategy` | `"cosine"` | `cosine`, `dot_product`, `euclidean` |
| `ivfpq_nprobe` | `64` | Cells probed per query |
| `ivfpq_nlist` | `4096` | IVF Voronoi cells |
| `ivfpq_m` | `48` | PQ sub-quantisers |
| `ivfpq_nbits` | `8` | Bits per PQ code |
| `hnsw_m` | `32` | HNSW connectivity |
| `hnsw_ef_construction` | `200` | HNSW build-time width |
| `hnsw_ef_search` | `128` | HNSW query-time width |

### ElasticsearchRagService

| Parameter | Default | Description |
|-----------|---------|-------------|
| `config` | `None` | `IndexingConfig` (required for non-BM25) |
| `strategy` | `"vector"` | `vector`, `approximation`, `bm25`, `hybrid` |
| `distance_function` | `None` | `COSINE`, `DOT_PRODUCT`, `EUCLIDEAN_DISTANCE` |
| `bm25_b` | `None` | BM25 b param (index creation only) |
| `bm25_k1` | `None` | BM25 k1 param (index creation only) |

### HybridFaissRagService

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dense_service` | required | FAISS dense backend |
| `sparse_service` | required | BM25 sparse backend |
| `rrf_k` | `60` | RRF smoothing constant |
| `rrf_depth` | `60` | Candidates per backend before fusion |

### RouterRagService (TorchScript)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dense_service` | required | Dense backend |
| `sparse_service` | required | Sparse backend |
| `device` | `"cpu"` | Torch device |

### NeuralRouterRagService

| Parameter | Default | Description |
|-----------|---------|-------------|
| `backends` | required | Dict of backend name → `RagService` |
| `backend_order` | required | Names in model-training order |
| `model_path` | required | Path to `.pt` model |
| `strict` | `True` | Argmax vs. probability-weighted RRF |
| `rrf_k` | `60` | RRF smoothing (hybrid mode) |
| `rrf_depth` | `60` | Candidates per backend (hybrid mode) |
| `predict_batch_size` | `32` | BERT inference batch size |
| `device` | `"cpu"` | Torch device |

---

## LLM Service Reference

| Service | Model | Platform | Key Parameters |
|---------|-------|----------|----------------|
| `GPTNeo27bLLMService` | EleutherAI/gpt-neo-2.7B | Modal H100 | `temperature`, `request_batch_size=128`, `gpu_batch_size=32` |
| `QwenLLMService` | Qwen2.5-7B-Instruct | Modal H100 | `temperature`, `request_batch_size=128`, `gpu_batch_size=32` |
| `MistralLLMService` | Mistral-7B-Instruct-v0.2 | Modal H100 | `temperature`, `request_batch_size=64`, `gpu_batch_size=4` |
| `OpenAIService` | (configurable) | OpenAI API | `model_name`, `temperature`, `requests_per_second=30` |
| `ModalLLMService` | (configurable) | Modal T4 | `model_name`, `temperature`, `max_new_tokens=512` |

---

## Embedding Providers

| Provider | Model (default) | Description |
|----------|-----------------|-------------|
| `huggingface` | `Lajavaness/bilingual-embedding-small` | Local sentence-transformer |
| `modal` | (same) | Modal GPU service for embeddings |
| `openai` | `text-embedding-3-large` | OpenAI embeddings API |
| `google` | — | Google Generative AI embeddings |

Configured via `IndexingConfig`:

```python
from src.rag.utils import IndexingConfig

config = IndexingConfig(
    embedding_provider="huggingface",
    embedding_model="Lajavaness/bilingual-embedding-small",
    chunk_size=1000,
    chunk_overlap=200,
)
```

---

## End-to-End Workflow Summary

```
1. Build indices (one-time)
   ├─ BM25:  BM25RagService.index_from_parquet()
   ├─ FAISS: FaissRagService.index_from_parquet()
   └─ ES:    ElasticsearchRagService.index_from_parquet()

2. Run pipeline
   ├─ Stage 1: retrieval_runner  → retrieved_docs_<key>.csv
   ├─ Stage 2: generating_runner → answer_checkpoint_<llm>_<key>_top<n>.csv
   └─ Stage 3: llm_eval_runner   → results_<llm>_<key>_<ctx>_<eval>.parquet

3. Build analysis dataset
   └─ scripts.build_analysis_dataset → analysis_dataset.parquet

4. (Optional) Train neural router
   ├─ train_router.py → models/router.pt
   └─ Add neural_router backend to retrieval_runner.py

5. (Optional) Re-run pipeline with neural router
   └─ retrieval_runner --only-keys neural_router_strict
```

---

## Testing

```bash
# Run all tests
pytest

# Skip integration tests (need live ES / API keys)
pytest -m "not integration"

# Single test file
pytest tests/bm25_parquet_vs_es.py

# Verbose, stop on first failure
pytest -xv
```
