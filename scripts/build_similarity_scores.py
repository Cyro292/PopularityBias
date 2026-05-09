"""Build similarity scores parquet from question/wiki pairs.

question_1 + wiki_id_1 → text fetched from wiki_2026_corpus (new articles)
question_2 + wiki_id_2 → text fetched from wiki_full_bil/wiki_corpus (old corpus)

Scores computed via two ES RAG services (BM25 + dense) on wiki_full_bil index.
Output: data/similarity_scores.parquet
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.rag.elasticsearch_rag_service import ElasticsearchRagService
from src.rag.utils import IndexingConfig
from src.embeddings.modal_embedding import ModalEmbeddings, MODEL_NAME
from src.script_classes.similarity_scorer import SimilarityScorer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Paths / config ─────────────────────────────────────────────────────────────

ROOT              = Path(__file__).parent.parent
PAIRS_JSON        = ROOT / "data" / "similarity_pairs.json"
CORPUS_2026       = ROOT / "data" / "wiki_2026" / "wiki_2026_corpus_clean.parquet"
CORPUS_BIL        = ROOT / "data" / "wiki_full_bil" / "wiki_corpus.parquet"
OUTPUT_PATH       = ROOT / "data" / "similarity_scores.parquet"

ES_URL      = os.environ.get("ELASTICSEARCH_ENDPOINT", "http://localhost:9200")
ES_USER     = os.environ.get("ELASTICSEARCH_USERNAME")
ES_PASSWORD = os.environ.get("ELASTICSEARCH_PASSWORD")
INDEX_NAME  = "wiki_full_bil"
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

    # ── Build ES services ─────────────────────────────────────────────────
    bm25_service = ElasticsearchRagService(
        es_url=ES_URL, es_user=ES_USER, es_password=ES_PASSWORD,
        strategy="bm25",
    )

    embeddings = ModalEmbeddings(
        model_name=MODEL_NAME,
        gpu_batch_size=512,
        request_batch_size=2048,
        normalise_embeddings=True,
    )
    dense_config = IndexingConfig(
        embedding_provider="modal",
        embedding_model=MODEL_NAME,
        gpu_batch_size=512,
        request_batch_size=2048,
        normalise_embeddings=True,
        distance_function="COSINE",
    )
    dense_service = ElasticsearchRagService(
        config=dense_config,
        es_url=ES_URL, es_user=ES_USER, es_password=ES_PASSWORD,
        strategy="approximation",
    )
    dense_service._embeddings = embeddings

    scorer = SimilarityScorer(
        bm25_service=bm25_service,
        dense_service=dense_service,
        index_name=INDEX_NAME,
        top_k=TOP_K,
        id_metadata_key="wikipedia_id",
    )

    # ── Score — question_1 vs wiki_id_1 (2026 corpus text as context,
    #           but scored against the ES index via the question text)
    # The ES services score question text against the index — the article
    # text from the corpus is used only for title display; the scorer
    # finds the target article's score inside the ES result set.
    # ─────────────────────────────────────────────────────────────────────
    pairs_1 = [
        (p["question_1"], p["wiki_id_1"])
        for p in pairs if p.get("wiki_id_1") is not None
    ]
    pairs_2 = [
        (p["question_2"], p["wiki_id_2"])
        for p in pairs if p.get("wiki_id_2") is not None
    ]

    scores_1 = scorer.score_batch(pairs_1) if pairs_1 else []
    scores_2 = scorer.score_batch(pairs_2) if pairs_2 else []

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
