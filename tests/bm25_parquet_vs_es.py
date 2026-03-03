"""
bm25_parquet_vs_es.py
---------------------
Compares Okapi BM25 retrieval computed directly from wiki_corpus.parquet
against the results returned by the Elasticsearch wiki_full index.

Algorithm
---------
Pass 1: Stream all row groups → compute global corpus statistics
        (N: total chunks, avgdl: average chunk length, df: per-query-term
         document frequency across all chunks).
Pass 2: Stream all row groups → score every chunk with Okapi BM25 and
        keep only a top-K min-heap.
Then:   Compare the top-K parquet ranking with the top-K ES ranking.

Chunking matches the indexed collection:
  RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

Retrieval unit matches Elasticsearch:
  Both sides score individual *chunks* (not whole pages), keep top-K chunks,
  then collapse to unique pages by max-score per wikipedia_id.

BM25 formula (Lucene/ES default):
  score(t, d) = IDF(t) * TF_norm(t, d)
  IDF(t)      = ln(1 + (N - df + 0.5) / (df + 0.5))
  TF_norm     = freq * (k1 + 1) / (freq + k1 * (1 - b + b * dl/avgdl))
  k1          = 1.2, b = 0.75

Memory footprint: one row group at a time  (~50k docs, ~150k chunks).
"""

from __future__ import annotations

import heapq
import logging
import math
import os
import re
import sys
from collections import Counter

import dotenv
import pyarrow.parquet as pq
from elasticsearch import Elasticsearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
try:
    dotenv.load_dotenv(dotenv_path=".env", override=False)
except Exception:
    # Fallback: best-effort auto-discovery
    dotenv.load_dotenv()

PARQUET_PATH = "data/wiki_full/wiki_corpus.parquet"
ES_INDEX     = "wiki_full"
ES_URL       = "http://localhost:9200"
CHUNK_SIZE   = 1000
CHUNK_OVERLAP = 100   # must match the value in data/wiki_full/metadata.json
TOP_K        = 200       # compare top-K results
BM25_K1      = 1.2
BM25_B       = 0.75

# One representative query
QUERY = "Who is the president of the United States?"

# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.WARNING, format="%(message)s")
log = logging.getLogger(__name__)


def get_es_client() -> Elasticsearch:
    user = os.getenv("ELASTICSEARCH_USERNAME")
    pwd  = os.getenv("ELASTICSEARCH_PASSWORD")
    return Elasticsearch(
        ES_URL,
        basic_auth=(user, pwd) if (user and pwd) else None,
        request_timeout=30,
    )


def tokenize(text: str) -> list[str]:
    """Simple whitespace/alphanumeric tokeniser – lowercase, matching ES default."""
    if not isinstance(text, str):
        return []
    return re.findall(r"[a-z0-9]+", text.lower())


def make_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )


# ---------------------------------------------------------------------------
# PASS 1  – corpus statistics
# ---------------------------------------------------------------------------
def pass1_stats(parquet_file: pq.ParquetFile, query_terms: set[str]) -> tuple[int, float, dict]:
    """
    Returns:
        N       – total chunk count
        avgdl   – average chunk length (tokens)
        df      – {term: chunk_doc_freq}
    """
    splitter  = make_splitter()

    N         = 0
    total_len = 0
    df        = {t: 0 for t in query_terms}

    for i in tqdm(range(parquet_file.num_row_groups), desc="Pass 1 – stats"):
        table = parquet_file.read_row_group(i, columns=["text"])
        texts = table["text"].to_pylist()

        for raw_text in texts:
            if not isinstance(raw_text, str):
                continue
            chunks = splitter.split_text(raw_text)
            for chunk in chunks:
                tokens     = tokenize(chunk)
                chunk_set  = set(tokens)
                length     = len(tokens)
                N         += 1
                total_len += length
                for t in query_terms:
                    if t in chunk_set:
                        df[t] += 1

    avgdl = total_len / N if N > 0 else 1.0
    return N, avgdl, df


# ---------------------------------------------------------------------------
# PASS 2  – BM25 scoring, top-K heap  (chunk-level, mirrors Elasticsearch)
# ---------------------------------------------------------------------------
def pass2_score(
    parquet_file: pq.ParquetFile,
    query_terms:  list[str],
    idf:          dict[str, float],
    avgdl:        float,
    top_k:        int,
) -> list[tuple[float, int, str, str]]:
    """
    Score every individual *chunk* (not whole pages) and return the top_k
    highest-scoring chunks as (score, wikipedia_id, title, snippet), sorted desc.

    This mirrors Elasticsearch exactly:
      ES stores each chunk as a separate document, runs BM25 per chunk, and
      returns the top-K chunk hits.  We then deduplicate to unique pages in
      dedupe_parquet_by_wiki_id(), exactly as dedupe_es_by_wiki_id() does.

    Min-heap of positive scores: heap[0] is always the *lowest* score in the
    current top-K, so we only evict when a new chunk scores higher.
    """
    splitter = make_splitter()
    heap: list[tuple[float, int, str, str]] = []  # (score, wiki_id, title, snippet)

    for i in tqdm(range(parquet_file.num_row_groups), desc="Pass 2 – scoring"):
        table  = parquet_file.read_row_group(
            i, columns=["wikipedia_id", "wikipedia_title", "text"]
        )
        rows = table.to_pylist()

        for row in rows:
            raw_text = row["text"]
            wiki_id  = row["wikipedia_id"]
            title    = row["wikipedia_title"]

            if not isinstance(raw_text, str):
                continue

            for chunk in splitter.split_text(raw_text):
                tokens = tokenize(chunk)
                dl     = len(tokens)
                if dl == 0:
                    continue
                tf = Counter(t for t in tokens if t in idf)
                if not tf:
                    continue

                score = sum(
                    idf[term] * freq * (BM25_K1 + 1) / (
                        freq + BM25_K1 * (1 - BM25_B + BM25_B * dl / avgdl)
                    )
                    for term, freq in tf.items()
                )
                if score <= 0:
                    continue

                entry = (score, wiki_id, title, chunk[:120])
                if len(heap) < top_k:
                    heapq.heappush(heap, entry)
                elif score > heap[0][0]:
                    heapq.heapreplace(heap, entry)

    # sort descending by score
    heap.sort(key=lambda r: r[0], reverse=True)
    return heap


def dedupe_parquet_by_wiki_id(chunk_hits: list[tuple[float, int, str, str]], top_k: int) -> list[dict]:
    """Collapse parquet chunk-hits into unique wikipedia pages by max score.
    Mirrors dedupe_es_by_wiki_id so both sides are compared identically."""
    best: dict[str, dict] = {}
    for rank, (score, wiki_id, title, snippet) in enumerate(chunk_hits, 1):
        key = str(wiki_id)
        prev = best.get(key)
        if prev is None or score > prev["pq_score"]:
            best[key] = {
                "wiki_id":         wiki_id,
                "title":           title,
                "snippet":         snippet,
                "pq_score":        score,
                "best_chunk_rank": rank,
            }
    pages = sorted(best.values(), key=lambda r: (-r["pq_score"], r["best_chunk_rank"]))
    return pages[:top_k]


# ---------------------------------------------------------------------------
# FETCH ES results
# ---------------------------------------------------------------------------
def fetch_es_results(es: Elasticsearch, query: str, top_k: int) -> list[dict]:
    resp = es.search(
        index=ES_INDEX,
        body={
            "query": {"match": {"text": query}},
            "size":  top_k,
            "_source": ["metadata", "text"],
        },
    )
    results = []
    for hit in resp["hits"]["hits"]:
        src  = hit["_source"]
        meta = src.get("metadata") or {}
        results.append(
            {
                "es_score":  hit["_score"],
                "wiki_id":   meta.get("wikipedia_id"),
                "title":     meta.get("wikipedia_title", ""),
                "snippet":   src.get("text", "")[:120],
            }
        )
    return results


def dedupe_es_by_wiki_id(es_hits: list[dict], top_k: int) -> list[dict]:
    """Collapse ES chunk-hits into unique wikipedia pages by max score."""
    best: dict[str, dict] = {}
    for rank, hit in enumerate(es_hits, 1):
        wid = hit.get("wiki_id")
        if wid is None:
            continue
        key = str(wid)
        prev = best.get(key)
        if prev is None or hit["es_score"] > prev["es_score"]:
            best[key] = {
                "wiki_id": wid,
                "title": hit.get("title", ""),
                "snippet": hit.get("snippet", ""),
                "es_score": hit["es_score"],
                "best_chunk_rank": rank,
            }
        else:
            # keep smallest rank where this page appeared
            if rank < prev.get("best_chunk_rank", rank):
                prev["best_chunk_rank"] = rank

    # Sort by ES score descending (tie-break: earliest appearance)
    pages = sorted(best.values(), key=lambda r: (-r["es_score"], r["best_chunk_rank"]))
    return pages[:top_k]


# ---------------------------------------------------------------------------
# COMPARISON REPORT
# ---------------------------------------------------------------------------
def print_comparison(parquet_pages: list[dict], es_pages: list[dict]) -> None:
    """Both parquet_pages and es_pages are lists of dicts with keys:
    wiki_id, title, pq_score / es_score (respectively)."""
    parquet_ids = [str(r["wiki_id"]) for r in parquet_pages]
    es_ids      = [str(r["wiki_id"]) for r in es_pages]

    parquet_set  = set(parquet_ids)
    es_set       = set(es_ids)
    overlap      = parquet_set & es_set
    overlap_pct  = 100 * len(overlap) / TOP_K if TOP_K else 0

    parquet_score_map = {str(r["wiki_id"]): r["pq_score"] for r in parquet_pages}
    parquet_title_map = {str(r["wiki_id"]): r["title"]    for r in parquet_pages}

    print("\n" + "=" * 100)
    print(f"QUERY : {QUERY}")
    print(f"TOP-K : {TOP_K}  (chunk-level, then deduped to unique pages)")
    print("=" * 100)

    print("\n--- Side-by-side: ES page rank vs Parquet BM25 score ---")
    print(f"{'Rank':<5} {'Wiki ID':<12} {'ES Score':<12} {'Parquet BM25':<14} {'Δ Score':<10} Title")
    print("-" * 100)
    for rank, r in enumerate(es_pages[:50], 1):
        wid   = str(r["wiki_id"])
        es_sc = r["es_score"]
        if wid in parquet_score_map:
            pq_sc = parquet_score_map[wid]
            delta = es_sc - pq_sc
            flag  = ""
        else:
            pq_sc = float("nan")
            delta = float("nan")
            flag  = "  ← NOT in parquet top-K"
        print(f"{rank:<5} {wid:<12} {es_sc:<12.4f} {pq_sc:<14.4f} {delta:<10.4f} {r['title']}{flag}")

    print("\n--- Parquet top-10 (after dedupe, by BM25 score) ---")
    print(f"{'Rank':<5} {'Wiki ID':<12} {'BM25 Score':<14} Title")
    print("-" * 80)
    for rank, r in enumerate(parquet_pages[:10], 1):
        in_es = "✓ in ES" if str(r["wiki_id"]) in es_set else "✗ NOT in ES top-K"
        print(f"{rank:<5} {str(r['wiki_id']):<12} {r['pq_score']:<14.4f} {r['title']}  [{in_es}]")

    print(f"\n{'='*60}")
    print(f"Overlap  (docs in BOTH top-{TOP_K}): {len(overlap)}/{TOP_K}  ({overlap_pct:.1f}%)")
    print(f"Only in ES     : {len(es_set - parquet_set)}")
    print(f"Only in Parquet: {len(parquet_set - es_set)}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"Query : {QUERY!r}")
    print(f"Index : {ES_INDEX}  |  Parquet: {PARQUET_PATH}")
    print(f"Top-K : {TOP_K}  |  Chunk: {CHUNK_SIZE}/{CHUNK_OVERLAP}  |  k1={BM25_K1}  b={BM25_B}\n")

    # 1. ES results
    es = get_es_client()
    assert es.ping(), "Cannot reach Elasticsearch!"
    print("Fetching ES results…")
    es_chunk_hits = fetch_es_results(es, QUERY, TOP_K)
    print(f"  Got {len(es_chunk_hits)} ES chunk hits.\n")
    es_pages = dedupe_es_by_wiki_id(es_chunk_hits, TOP_K)
    print(f"  Collapsed to {len(es_pages)} unique pages.\n")

    # 2. Parquet BM25
    parquet_file = pq.ParquetFile(PARQUET_PATH)
    print(f"Parquet file: {parquet_file.num_row_groups} row groups\n")

    # Keep query as bag-of-words (duplicates allowed) for scoring; use unique for df/idf
    query_tokens = tokenize(QUERY)
    query_terms_unique = set(query_tokens)
    print(f"Query tokens: {query_tokens}\n")

    # Pass 1
    N, avgdl, df = pass1_stats(parquet_file, query_terms_unique)
    print(f"\nCorpus stats → N={N:,}  avgdl={avgdl:.1f} tokens")
    print(f"Doc freqs    → {df}\n")

    # Compute IDF (Lucene formula)
    idf = {
        t: math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5))
        for t in query_terms_unique
    }
    print(f"IDF values   → { {t: round(v, 4) for t, v in idf.items()} }\n")

    # Pass 2 – top-K individual chunks
    parquet_chunks = pass2_score(parquet_file, query_tokens, idf, avgdl, TOP_K)
    print(f"  Top chunk score: {parquet_chunks[0][0]:.4f}" if parquet_chunks else "  (no chunks scored)")

    # Dedupe chunks → unique pages (mirrors dedupe_es_by_wiki_id)
    parquet_pages = dedupe_parquet_by_wiki_id(parquet_chunks, TOP_K)
    print(f"  Collapsed to {len(parquet_pages)} unique pages.\n")

    # 3. Compare
    print_comparison(parquet_pages, es_pages)


if __name__ == "__main__":
    main()
