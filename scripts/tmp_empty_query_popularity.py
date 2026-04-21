"""Probe what the RAG retrieves for empty queries, broken down by popularity decile.

Sends a configurable number of empty-string queries to each retrieval strategy
(bm25, approximation) and records the popularity distribution of the returned
documents.  Mirrors the setup in ``src/process/rag_pipeline.py`` so the same
index / service config is used.

Usage::

    python scripts/tmp_empty_query_popularity.py

Outputs a summary table to stdout showing, per strategy:
  - mean / median / std of retrieved document popularity
  - document count per decile bucket
  - decile bias ratio (decile 9 count / decile 0 count)
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm.auto import tqdm

# ── Repo root on path ──────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

load_dotenv()

from config import DATA_DIR
from src.rag.elasticsearch_rag_service import ElasticsearchRagService
from src.rag.utils import IndexingConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("elastic_transport.transport").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# ── Configuration (mirrors PipelineConfig defaults) ────────────────────────
COLLECTION_NAME     = "wiki_full_bil"
STRATEGIES          = ["bm25", "approximation"]
NUM_QUERIES         = 500        # number of empty queries to fire per strategy
TOP_K               = 5         # docs returned per query
NUM_CANDIDATES      = 1_000     # kNN num_candidates (approximation only)
SEARCH_WORKERS      = 6
MSEARCH_BATCH_SIZE  = 50
NUM_DECILES         = 10

CHUNK_SIZE          = 1_000
CHUNK_OVERLAP       = 100
EMBEDDING_MODEL     = "Lajavaness/bilingual-embedding-small"
EMBEDDING_PROVIDER  = "huggingface"
EMBED_BATCH_SIZE    = 254
GPU_BATCH_SIZE      = 254

ES_URL      = os.getenv("ELASTICSEARCH_ENDPOINT", "")
ES_USER     = os.getenv("ELASTICSEARCH_USERNAME", "")
ES_PASSWORD = os.getenv("ELASTICSEARCH_PASSWORD", "")

CORPUS_PATH = DATA_DIR / COLLECTION_NAME / "wiki_corpus.parquet"


# === Helpers ===

def build_popularity_lookup(corpus_path: Path) -> dict[int, float]:
    """Build a wikipedia_id -> popularity_avg mapping from the corpus parquet.

    Args:
        corpus_path: Path to the wiki_corpus.parquet file.

    Returns:
        Dict mapping wikipedia_id (int) to its mean daily pageview count.

    Raises:
        FileNotFoundError: If corpus_path does not exist.
    """
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    logger.info("Loading popularity lookup from %s …", corpus_path)
    df = pd.read_parquet(corpus_path, columns=["wikipedia_id", "popularity_avg"])
    lookup: dict[int, float] = {
        int(row.wikipedia_id): float(row.popularity_avg)
        for row in df.itertuples(index=False)
        if pd.notna(row.popularity_avg)
    }
    logger.info("  %d entries loaded", len(lookup))
    return lookup


def assign_decile(popularity: float, boundaries: list[float]) -> int:
    """Assign a 0-indexed decile given sorted boundary values.

    Args:
        popularity: The popularity_avg value for a document.
        boundaries: Sorted list of boundary values (len = NUM_DECILES - 1).

    Returns:
        Integer decile 0–9 (inclusive).
    """
    for i, b in enumerate(boundaries):
        if popularity <= b:
            return i
    return NUM_DECILES - 1


def compute_boundaries(lookup: dict[int, float]) -> list[float]:
    """Compute decile boundary values from the popularity lookup.

    Args:
        lookup: wikipedia_id -> popularity_avg mapping.

    Returns:
        List of 9 boundary floats (10th through 90th percentile).
    """
    values = sorted(lookup.values())
    n = len(values)
    boundaries: list[float] = []
    for i in range(1, NUM_DECILES):
        idx = max(0, int(n * i / NUM_DECILES) - 1)
        boundaries.append(values[idx])
    return boundaries


def summarise(
    strategy: str,
    all_popularities: list[float],
    boundaries: list[float],
) -> None:
    """Print a popularity + decile summary table for one strategy.

    Args:
        strategy: Strategy name (e.g. 'bm25').
        all_popularities: Flat list of popularity_avg values for all retrieved docs.
        boundaries: Decile boundary values from the corpus.
    """
    if not all_popularities:
        logger.warning("No popularity data for strategy '%s'", strategy)
        return

    s = pd.Series(all_popularities)
    deciles = s.apply(lambda p: assign_decile(p, boundaries))

    print(f"\n{'=' * 60}")
    print(f"  Strategy: {strategy.upper()}")
    print(f"  Queries: {NUM_QUERIES}  |  Top-K: {TOP_K}  |  Total docs: {len(s)}")
    print(f"{'=' * 60}")
    print(f"  Popularity (pageviews/day)")
    print(f"    mean   : {s.mean():>12.2f}")
    print(f"    median : {s.median():>12.2f}")
    print(f"    std    : {s.std():>12.2f}")
    print(f"    min    : {s.min():>12.2f}")
    print(f"    max    : {s.max():>12.2f}")
    print()
    print(f"  Docs per decile (0 = least popular, {NUM_DECILES-1} = most popular)")
    counts = deciles.value_counts().sort_index()
    for d in range(NUM_DECILES):
        count = counts.get(d, 0)
        bar = "#" * int(count / max(counts.values) * 30) if counts.max() > 0 else ""
        print(f"    decile {d}: {count:4d}  {bar}")

    d0 = counts.get(0, 0)
    d9 = counts.get(NUM_DECILES - 1, 0)
    if d0 > 0:
        ratio = d9 / d0
        print(f"\n  Bias ratio (decile {NUM_DECILES-1} / decile 0): {ratio:.2f}x")
    else:
        print(f"\n  Bias ratio: decile 0 has 0 docs, cannot compute ratio")
    print()


# === Main ===

def main() -> None:
    """Run empty-query retrieval probes and print popularity distribution."""

    # ── Build corpus popularity lookup ─────────────────────────────────────
    lookup = build_popularity_lookup(CORPUS_PATH)
    boundaries = compute_boundaries(lookup)
    logger.info(
        "Decile boundaries (10th–90th pct): %s",
        [f"{b:.1f}" for b in boundaries],
    )

    # ── Initialise RAG service (mirrors rag_pipeline.py) ───────────────────
    rag_service = ElasticsearchRagService(
        config=IndexingConfig(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            embedding_model=EMBEDDING_MODEL,
            embedding_provider=EMBEDDING_PROVIDER,
            request_batch_size=EMBED_BATCH_SIZE,
            gpu_batch_size=GPU_BATCH_SIZE,
            normalise_embeddings=True,
            trust_remote_code=True,
        ),
        es_url=ES_URL,
        es_user=ES_USER,
        es_password=ES_PASSWORD,
    )

    logger.info("Loading index '%s' …", COLLECTION_NAME)
    rag_service.load_index(COLLECTION_NAME)

    # ── Fire empty queries per strategy ────────────────────────────────────
    empty_queries = [""] * NUM_QUERIES

    for strategy in STRATEGIES:
        logger.info("Probing strategy '%s' with %d empty queries …", strategy, NUM_QUERIES)

        results_with_scores = rag_service.batch_retrieve_with_scores(
            empty_queries,
            top_k=TOP_K,
            strategy=strategy,
            search_workers=SEARCH_WORKERS,
            msearch_batch_size=MSEARCH_BATCH_SIZE,
            num_candidates=NUM_CANDIDATES,
        )

        all_popularities: list[float] = []
        missing = 0

        for docs_with_scores in tqdm(results_with_scores, desc=f"  Collecting [{strategy}]", leave=False):
            for doc, _score in docs_with_scores:
                # Prefer popularity stored in doc metadata, fall back to corpus lookup
                pop = doc.metadata.get("popularity") or doc.metadata.get("popularity_avg")
                if pop is None:
                    wiki_id = doc.metadata.get("wikipedia_id")
                    if wiki_id is not None:
                        pop = lookup.get(int(wiki_id))
                if pop is not None:
                    all_popularities.append(float(pop))
                else:
                    missing += 1

        if missing:
            logger.warning(
                "  [%s] %d retrieved docs had no popularity data", strategy, missing
            )

        summarise(strategy, all_popularities, boundaries)

    logger.info("Done.")


if __name__ == "__main__":
    main()
