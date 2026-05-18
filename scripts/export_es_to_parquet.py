"""export_es_to_parquet.py — Export Elasticsearch wiki_full_bil index to parquet.

Exports: wikipedia_id (int), text (str), vector (list[float32], 384-dim)
Writes a directory of shard parquets — no merge step, constant peak RAM.

Output directory can be read as a single dataframe:
    pd.read_parquet("data/wiki_full_bil/chunks_with_vectors/")

Usage:
    python scripts/export_es_to_parquet.py
    python scripts/export_es_to_parquet.py --batch-size 2000 --output data/wiki_full_bil/chunks_with_vectors
"""
from __future__ import annotations

import argparse
import gc
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
from tqdm import tqdm

logging.basicConfig(
    level=logging.WARNING,   # suppress info spam — tqdm handles progress
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
ES_ENDPOINT = os.getenv("ELASTICSEARCH_ENDPOINT", "http://localhost:9200")
ES_USER     = os.getenv("ELASTICSEARCH_USERNAME",  "elastic")
ES_PASS     = os.getenv("ELASTICSEARCH_PASSWORD",  "ifCbkIYF")
INDEX       = "wiki_full_bil"
SCROLL_TTL  = "5m"

# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output",     default="data/wiki_full_bil/chunks_with_vectors",
                   help="Output directory for shard parquets")
    p.add_argument("--batch-size", type=int, default=2000)
    p.add_argument("--shard-size", type=int, default=500_000,
                   help="Rows per shard parquet")
    return p.parse_args()


def scroll_es(
    batch_size: int,
    auth: HTTPBasicAuth,
) -> list[dict]:
    """Yield batches of hits from ES scroll API."""
    init_url = f"{ES_ENDPOINT}/{INDEX}/_search?scroll={SCROLL_TTL}&size={batch_size}"
    body = {
        "query": {"match_all": {}},
        "_source": ["text", "metadata"],
        "fields":  ["vector"],
    }
    resp = requests.post(init_url, json=body, auth=auth, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    scroll_id = data["_scroll_id"]
    hits      = data["hits"]["hits"]
    total     = data["hits"]["total"]["value"]

    with tqdm(total=total, unit="doc", unit_scale=True, desc="Scrolling ES") as pbar:
        while hits:
            yield hits
            pbar.update(len(hits))

            resp = requests.post(
                f"{ES_ENDPOINT}/_search/scroll",
                json={"scroll": SCROLL_TTL, "scroll_id": scroll_id},
                auth=auth,
                timeout=120,
            )
            resp.raise_for_status()
            data      = resp.json()
            scroll_id = data["_scroll_id"]
            hits      = data["hits"]["hits"]

    # Clean up scroll context
    requests.delete(
        f"{ES_ENDPOINT}/_search/scroll",
        json={"scroll_id": scroll_id},
        auth=auth,
        timeout=30,
    )


def hits_to_rows(hits: list[dict]) -> tuple[list[int], list[str], list[list[float]]]:
    wiki_ids, texts, vectors = [], [], []
    for h in hits:
        src  = h["_source"]
        meta = src.get("metadata", {})
        wiki_ids.append(int(meta.get("wikipedia_id", -1)))
        texts.append(src.get("text", ""))
        vectors.append(h["fields"]["vector"])   # already list[float]
    return wiki_ids, texts, vectors


def write_shard(
    wiki_ids: list[int],
    texts:    list[str],
    vectors:  list[list[float]],
    path:     Path,
) -> None:
    arr = np.array(vectors, dtype=np.float32)  # (N, 384) contiguous array
    df = pd.DataFrame({
        "wikipedia_id": np.array(wiki_ids, dtype=np.int64),
        "text":         texts,
        "vector":       list(arr),              # list of 1-D float32 arrays
    })
    df.to_parquet(path, index=False, compression="zstd")
    del arr, df
    gc.collect()
    tqdm.write(f"  shard saved → {path.name}  ({len(wiki_ids):,} rows, {path.stat().st_size/1e6:.0f} MB)")


def main() -> None:
    args      = parse_args()
    output    = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    auth = HTTPBasicAuth(ES_USER, ES_PASS)

    # ── Resume: find already-written shards ───────────────────────────────────
    existing    = sorted(output.glob("shard_*.parquet"))
    shard_idx   = len(existing)
    shard_paths = list(existing)
    rows_done   = shard_idx * args.shard_size
    if existing:
        tqdm.write(f"Resuming — {shard_idx} shards already done (~{rows_done:,} docs)")

    # ── Scroll & write shards — one shard in RAM at a time ────────────────────
    buf_ids:   list[int]         = []
    buf_texts: list[str]         = []
    buf_vecs:  list[list[float]] = []

    t0 = time.time()
    for batch in scroll_es(args.batch_size, auth):
        wids, txts, vecs = hits_to_rows(batch)
        buf_ids   += wids
        buf_texts += txts
        buf_vecs  += vecs

        if len(buf_ids) >= args.shard_size:
            path = output / f"shard_{shard_idx:04d}.parquet"
            write_shard(buf_ids, buf_texts, buf_vecs, path)
            shard_paths.append(path)
            shard_idx += 1
            buf_ids, buf_texts, buf_vecs = [], [], []

    # Flush remaining rows
    if buf_ids:
        path = output / f"shard_{shard_idx:04d}.parquet"
        write_shard(buf_ids, buf_texts, buf_vecs, path)
        shard_paths.append(path)

    total_rows = sum(
        len(pd.read_parquet(p, columns=["wikipedia_id"])) for p in shard_paths
    )
    total_size = sum(p.stat().st_size for p in shard_paths) / 1e9
    tqdm.write(
        f"\nDone in {time.time()-t0:.0f}s — "
        f"{total_rows:,} rows across {len(shard_paths)} shards, "
        f"{total_size:.2f} GB total → {output}/"
    )
    tqdm.write("Read back with: pd.read_parquet('" + str(output) + "/')")


if __name__ == "__main__":
    main()
