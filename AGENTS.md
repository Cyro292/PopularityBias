# AGENTS.md — PopularityBias

Guidelines for agentic coding assistants working in this repository.

---

## Project Overview

Python research codebase investigating popularity bias in RAG (Retrieval-Augmented Generation)
systems. It evaluates retrieval quality across Wikipedia articles stratified by pageview
popularity (deciles). Key components: RAG service backends (BM25, FAISS, Elasticsearch,
Chroma), a QA dataset pipeline, async Wikipedia pageview fetching, and OpenAI/Google LLM
wrappers.

**Python 3.11+** (uses `X | Y` union syntax and `match` patterns; Modal cloud workers target 3.11).

---

## Build / Environment Setup

There is no Makefile or pyproject.toml. Use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` → `.env` (if it exists) or create `.env` with:
```
OPENAI_API_KEY=...
HUGGINGFACE_TOKEN=...
GOOGLE_API_KEY=...
ELASTICSEARCH_ENDPOINT=...
ELASTICSEARCH_USERNAME=...
ELASTICSEARCH_PASSWORD=...
ELASTICSEARCH_API_KEY=...
```

Local Elasticsearch (via Docker Compose):
```bash
cd elastic-start-local && bash start.sh
```

---

## Test Commands

```bash
# Run all tests
pytest

# Run all tests, skipping integration tests (those requiring live ES / API keys)
pytest -m "not integration"

# Run a single test file
pytest tests/test_bm25_analysis.py

# Run a single test by name
pytest tests/test_bm25_analysis.py::test_compute_per_query_retrieval_metrics

# Verbose output
pytest -v

# Stop on first failure
pytest -x
```

`pytest.ini` sets `asyncio_mode = auto` — all `async def test_*` functions run automatically
without needing `@pytest.mark.asyncio`.

Integration tests (marked `@pytest.mark.integration`) require a live Elasticsearch instance
and valid API keys in `.env`.

---

## Key Workflows

```bash
# Build local or Elasticsearch indices
python -m src.process.indexing.run_bm25 --help
python -m src.process.indexing.run_faiss --help
python -m src.process.indexing.run_es --jobs data/jobs/job1.json

# Prepare a balanced QA evaluation dataset
python -m src.process.qa_datasets.prepare_qa \
    --qa-datasets natural_questions hotpot_qa \
    --corpus data/wiki_full/wiki_corpus.parquet \
    --output data/wiki_full/all_qa_8k.parquet \
    --balance --target-per-decile 800

# Deploy Modal GPU embedding service
modal deploy src/embeddings/modal_embedding.py
```

---

## Code Style

### Imports

Always start every module with `from __future__ import annotations`.
Group imports in this order with one blank line between groups:

```python
from __future__ import annotations

# 1. Standard library
import gc
import logging
from pathlib import Path
from typing import Any, Sequence

# 2. Third-party
import pandas as pd
from langchain.schema import Document
from tqdm import tqdm

# 3. Internal (relative)
from .base import RagService
from .utils import IndexingConfig
```

Use lazy imports inside functions when importing heavy optional dependencies (e.g.,
`faiss`, `modal`) to avoid import-time errors when a backend is not installed.

### Naming Conventions

| Kind | Style | Example |
|---|---|---|
| Classes | `PascalCase` | `BM25RagService`, `SqliteDocstore` |
| Functions / methods | `snake_case` | `retrieve_documents`, `index_from_parquet` |
| Module-level constants | `UPPER_SNAKE_CASE` | `COL_POPULARITY`, `MAX_CONCURRENT_REQUESTS` |
| Private helpers | leading `_` | `_rate_limiter`, `_cache_key` |
| Module files | `snake_case` | `bm25_rag_service.py`, `decile_utils.py` |

Avoid `camelCase` for modules or functions (some legacy files use it; do not replicate).

### Type Annotations

- Annotate all public function signatures (parameters and return types).
- Prefer PEP 604 `X | Y` union over `Union[X, Y]` (requires `from __future__ import annotations`).
- Use `T | None` over `Optional[T]` for nullable types.
- Use `Sequence[T]` for read-only parameters, `list[T]` for mutable/concrete lists.
- Use `@dataclass` for plain configuration objects (see `IndexingConfig` in `rag/utils.py`).
- Use `Literal[...]` for fixed-choice string parameters (e.g., retrieval strategy).

```python
def retrieve(
    self,
    query: str,
    k: int = 10,
    strategy: Literal["vector", "bm25", "hybrid"] = "vector",
    filters: dict[str, Any] | None = None,
) -> list[Document]:
```

### Formatting

- **Indentation**: 4 spaces (no tabs).
- **Line length**: no hard limit enforced, but keep lines readable (aim for ≤ 100 chars).
- **Strings**: f-strings for all interpolation; double quotes preferred.
- **Paths**: always use `pathlib.Path`, never `os.path.join`.
- **Section separators**: use `# === Section Name ===` or `# ── Section Name ──` comment
  headers to visually group large files into logical sections.
- **Keyword-only args**: use `*` separator for optional configuration parameters to prevent
  accidental positional misuse.

### Logging

Every module must define a module-level logger:

```python
logger = logging.getLogger(__name__)
```

Use `logger.info` for pipeline progress, `logger.warning` for non-fatal skips,
`logger.error` for caught exceptions before re-raising. Never use `print` in library
code (scripts and notebooks may use `print`).

### Error Handling

- Raise `ValueError` for invalid arguments or missing required configuration.
- Raise `NotImplementedError` for unimplemented abstract paths (even inside `@abstractmethod`).
- In batch / async loops: catch `Exception`, log with `logger.error(f"...: {e}")`, then
  `raise` — do not silently swallow errors.
- Use `logger.warning` + `continue` for non-fatal per-item failures (e.g., a single doc
  failing to embed).

```python
try:
    embeddings = embed_documents(batch)
except Exception as e:
    logger.error(f"Error embedding batch starting at index {i}: {e}")
    raise
```

### Docstrings

Use **Google-style docstrings** for all public classes and functions:

```python
def compute_corpus_boundaries(
    parquet_path: Path,
    column: str = COL_POPULARITY,
) -> list[float]:
    """Compute decile boundary values from a parquet corpus.

    Args:
        parquet_path: Path to the parquet file containing the corpus.
        column: Name of the numeric column to compute boundaries for.

    Returns:
        List of 9 boundary values (the 10th, 20th, … 90th percentiles).

    Raises:
        FileNotFoundError: If parquet_path does not exist.
        KeyError: If column is not present in the parquet file.
    """
```

All `.py` files must have a module-level docstring describing their purpose.

### Memory Management

Call `gc.collect()` explicitly after deleting large DataFrames in batch loops to avoid
OOM in long-running indexing scripts.

---

## Architecture Notes

- **`rag/base.py`**: `RagService` ABC and `VectorStoreLike` Protocol — all backends implement these.
- **`rag/utils.py`**: Shared utilities: `build_embeddings()`, `IndexingConfig` dataclass,
  `BaseRateLimiter`, batch retrieval helpers. Import from here rather than duplicating.
- **`config.py`**: Single source of truth for paths (`ROOT_DIR`, `DATA_DIR`) and environment
  variables. Always import paths from here, never hard-code.
- **`src/metrics/decile_utils.py`**: Popularity decile logic. `COL_POPULARITY` constant defines
  the canonical column name used across the pipeline.
- Data files (parquet, indices, caches) live under `data/` which is gitignored.
- Notebooks under the root are for experimentation; production logic belongs in `.py` modules.

---

## No Linting / Formatting Tooling Configured

There is currently no `black`, `ruff`, `flake8`, `mypy`, or `isort` configuration.
Follow the style conventions above manually. If you add a formatter, do so via
`pyproject.toml` and update this file.
