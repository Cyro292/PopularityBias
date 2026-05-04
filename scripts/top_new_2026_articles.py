"""Print the top 10 Wikipedia articles by 2026 views that have no presence in the old (HF) dataset."""

from __future__ import annotations

import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path

CORPUS_PATH = Path(__file__).parent.parent / "data" / "wiki_2026" / "wiki_2026_corpus.parquet"
COLUMNS = ["id", "title", "views_old_hf", "views_2026"]
BATCH_SIZE = 50_000
TOP_N = 50


def main() -> None:
    pf = pq.ParquetFile(CORPUS_PATH)
    chunks: list[pd.DataFrame] = []

    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=COLUMNS):
        df = batch.to_pandas()
        chunks.append(df[(df["views_old_hf"] == 0) & (df["views_2026"] > 0)])

    result = pd.concat(chunks, ignore_index=True).nlargest(TOP_N, "views_2026")

    print(f"{'Rank':<5} {'Wikipedia ID':<15} {'Views 2026':>12}  Title")
    print("-" * 70)
    for rank, (_, row) in enumerate(result.iterrows(), 1):
        print(f"{rank:<5} {int(row['id']):<15} {int(row['views_2026']):>12,}  {row['title']}")


if __name__ == "__main__":
    main()
