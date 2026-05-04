"""Demo: find top-k BM25 nearest Wikipedia documents for a query via RagBM25SimilarityFinder."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.rag.elasticsearch_rag_service import ElasticsearchRagService
from src.script_classes.rag_bm25_similarity_finder import RagBM25SimilarityFinder

ES_URL      = os.environ.get("ELASTICSEARCH_ENDPOINT", "http://localhost:9200")
ES_USER     = os.environ.get("ELASTICSEARCH_USERNAME")
ES_PASSWORD = os.environ.get("ELASTICSEARCH_PASSWORD")
INDEX_NAME  = "wiki_full_bil"
TOP_K       = 5

QUERY_TEXTS = [
    "Reza Pahlavi is the eldest son of Mohammad Reza Shah, the last Shah of Iran.",
]


def main() -> None:
    service = ElasticsearchRagService(
        es_url=ES_URL,
        es_user=ES_USER,
        es_password=ES_PASSWORD,
        strategy="bm25",
    )

    finder = RagBM25SimilarityFinder(
        rag_service=service,
        index_name=INDEX_NAME,
        id_metadata_key="wikipedia_id",
        title_metadata_key="wikipedia_title",
    )

    results = finder.find(QUERY_TEXTS, top_k=TOP_K)

    for query, matches in results.items():
        print(f"\nQuery: {query!r}")
        print(f"{'Rank':<5} {'Wikipedia ID':<15} {'Score':<10} Title")
        print("-" * 70)
        for rank, match in enumerate(matches, 1):
            print(f"{rank:<5} {str(match.wikipedia_id):<15} {match.score:<10.4f} {match.title}")


if __name__ == "__main__":
    main()
