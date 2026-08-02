"""Build BM25 and FAISS similarity scores parquet from question/wiki pairs.

question_1 + wiki_id_1 → text fetched from wiki_2026_corpus (new articles)
question_2 + wiki_id_2 → text fetched from wiki_full_bil/wiki_corpus (old corpus)

Dense scores use the matching ``faiss_high`` index; BM25 scores use the
matching ``bm25_bm25plus`` index for each corpus.
Output: data/similarity_scores.parquet
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from dotenv import load_dotenv
load_dotenv()

from config import DATA_DIR
from src.rag.bm25_rag_service import BM25RagService
from src.rag.faiss_rag_service import FaissRagService
from src.rag.utils import IndexingConfig
from src.metrics.similarity_scorer import SimilarityScorer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Paths / config ─────────────────────────────────────────────────────────────

PAIRS_JSON = DATA_DIR / "similarity_pairs.json"
CORPUS_2026 = DATA_DIR / "wiki_2026" / "wiki_2026_corpus_clean.parquet"
CORPUS_BIL = DATA_DIR / "wiki_full_bil" / "wiki_corpus.parquet"
OUTPUT_PATH = DATA_DIR / "similarity_scores.parquet"

FAISS_2026 = DATA_DIR / "wiki_2026" / "faiss_high"
FAISS_BIL = DATA_DIR / "wiki_full_bil" / "faiss_high"
BM25_2026 = DATA_DIR / "wiki_2026" / "bm25_bm25plus"
BM25_BIL = DATA_DIR / "wiki_full_bil" / "bm25_bm25plus"
TOP_K       = 50


# ── Article fetching ───────────────────────────────────────────────────────────

def _fetch_articles(
    parquet_path: Path,
    ids: list[int],
    id_col: str,
    title_col: str,
    text_col: str = "text",
    batch_size: int = 50_000,
) -> dict[int, dict]:
    """Stream a parquet and return {id: {title, text}} for requested ids only."""
    id_set = set(ids)
    result: dict[int, dict] = {}
    pf = pq.ParquetFile(parquet_path)

    for batch in pf.iter_batches(batch_size=batch_size, columns=[id_col, title_col, text_col]):
        df = batch.to_pandas()
        matches = df[df[id_col].isin(id_set)]
        for _, row in matches.iterrows():
            result[int(row[id_col])] = {
                "title": row[title_col],
                "text": row[text_col],
            }
        if len(result) == len(id_set):
            break

    missing = id_set - set(result.keys())
    if missing:
        logger.warning(f"IDs not found in {parquet_path.name}: {missing}")
    return result


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """Score every analogue pair against its own corpus's FAISS index."""
    required_paths = [
        PAIRS_JSON, CORPUS_2026, CORPUS_BIL,
        FAISS_2026, FAISS_BIL, BM25_2026, BM25_BIL,
    ]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Required pair, corpus, FAISS, or BM25 index artifact(s) not found: "
            f"{missing_paths}. Build matching 2026 FAISS and BM25 indices before scoring."
        )
    pairs: list[dict] = json.loads(PAIRS_JSON.read_text())
    logger.info(f"Loaded {len(pairs)} pair(s)")

    # Collect ids per corpus — skip nulls
    ids_2026 = [p["wiki_id_1"] for p in pairs if p.get("wiki_id_1") is not None]
    ids_bil  = [p["wiki_id_2"] for p in pairs if p.get("wiki_id_2") is not None]

    logger.info(f"Fetching {len(ids_2026)} articles from wiki_2026_corpus...")
    articles_2026 = _fetch_articles(
        CORPUS_2026, ids_2026,
        id_col="id", title_col="title",
    )

    logger.info(f"Fetching {len(ids_bil)} articles from wiki_full_bil corpus...")
    articles_bil = _fetch_articles(
        CORPUS_BIL, ids_bil,
        id_col="wikipedia_id", title_col="wikipedia_title",
    )

    # Both sides are scored in corpus-specific indices configured identically.
    # This prevents absent 2026 pages from being mistaken for zero-scored old
    # corpus retrieval targets.
    config = IndexingConfig(
        chunk_size=1_000,
        chunk_overlap=100,
        embedding_provider="huggingface",
        embedding_model="Lajavaness/bilingual-embedding-small",
        gpu_batch_size=254,
        request_batch_size=254,
        normalise_embeddings=True,
        trust_remote_code=True,
    )
    new_dense_service = FaissRagService(config=config, strategy="ivfpq", distance_strategy="cosine")
    old_dense_service = FaissRagService(config=config, strategy="ivfpq", distance_strategy="cosine")
    new_bm25_service = BM25RagService(method="bm25+")
    old_bm25_service = BM25RagService(method="bm25+")
    new_scorer = SimilarityScorer(
        bm25_service=new_bm25_service,
        dense_service=new_dense_service,
        index_name=str(FAISS_2026),
        bm25_index_name=BM25_2026,
        top_k=TOP_K,
        id_metadata_key="id",
    )
    old_scorer = SimilarityScorer(
        bm25_service=old_bm25_service,
        dense_service=old_dense_service,
        index_name=str(FAISS_BIL),
        bm25_index_name=BM25_BIL,
        top_k=TOP_K,
        id_metadata_key="wikipedia_id",
    )

    # ── Score each side against its own corpus. ────────────────────────────
    pairs_1 = [
        (p["question_1"], p["wiki_id_1"])
        for p in pairs if p.get("wiki_id_1") is not None
    ]
    pairs_2 = [
        (p["question_2"], p["wiki_id_2"])
        for p in pairs if p.get("wiki_id_2") is not None
    ]

    scores_1 = new_scorer.score_batch(pairs_1) if pairs_1 else []
    scores_2 = old_scorer.score_batch(pairs_2) if pairs_2 else []

    # ── Assemble rows ─────────────────────────────────────────────────────
    rows = []
    s1_iter = iter(scores_1)
    s2_iter = iter(scores_2)

    for p in pairs:
        wid1 = p.get("wiki_id_1")
        wid2 = p.get("wiki_id_2")

        s1 = next(s1_iter) if wid1 is not None else None
        s2 = next(s2_iter) if wid2 is not None else None

        a1 = articles_2026.get(wid1, {}) if wid1 is not None else {}
        a2 = articles_bil.get(wid2, {})  if wid2 is not None else {}

        rows.append({
            "question_1":     p["question_1"],
            "wiki_id_1":      wid1,
            "wiki_title_1":   a1.get("title", ""),
            "bm25_score_1":   s1.bm25_score   if s1 else None,
            "cosine_score_1": s1.cosine_score if s1 else None,
            "question_2":     p["question_2"],
            "wiki_id_2":      wid2,
            "wiki_title_2":   a2.get("title", ""),
            "bm25_score_2":   s2.bm25_score   if s2 else None,
            "cosine_score_2": s2.cosine_score if s2 else None,
        })

    df = pd.DataFrame(rows)
    df.to_parquet(OUTPUT_PATH, index=False)
    logger.info(f"Written {len(df)} row(s) to {OUTPUT_PATH}")
    print(df[["question_1", "wiki_title_1", "bm25_score_1", "cosine_score_1",
              "question_2", "wiki_title_2", "bm25_score_2", "cosine_score_2"]].to_string(index=False))


if __name__ == "__main__":
    main()
