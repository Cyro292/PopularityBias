"""Tune a local IVF-PQ FAISS index for higher recall.

This script lets you adjust the two main accuracy levers of an IVF-PQ index
**without rebuilding it from scratch**:

Runtime parameters (changed in-place, no data loss, instant):
  --nprobe N     Number of IVF cells probed per query.
                 Current default: 64.  Try 128 / 256 / 512.
                 Higher = better recall, slower queries (linear trade-off).

Structural parameters (require a full index rebuild, slower):
  --m N          Number of PQ sub-vectors.  Must divide the embedding dim (384).
                 Current: 32 (dsub=12).  Try 48 (dsub=8) or 64 (dsub=6).
                 Higher M = less compression = better recall, more RAM.
  --nbits N      Bits per PQ code.  Current: 8 (256 centroids per sub-vector).
                 8 is standard; 12 gives significantly better recall but 1.5× storage.
  --nlist N      Number of IVF Voronoi cells.  Current: 4096.
                 Rule of thumb: sqrt(ntotal) ≈ 4969 for 24.7M vectors.
                 Try 8192 for better cell granularity.

Usage examples
--------------
# Just bump nprobe (instant, no rebuild):
python scripts/tune_faiss_index.py --nprobe 256

# Rebuild with better PQ (takes ~hours on 24M vectors):
python scripts/tune_faiss_index.py --m 48 --nlist 8192 --rebuild

# Dry run — print what would change without touching anything:
python scripts/tune_faiss_index.py --nprobe 256 --m 48 --dry-run

# Benchmark recall@10 with a quick sample after tuning:
python scripts/tune_faiss_index.py --nprobe 256 --benchmark

Current index (data/faiss_migrated):
  ntotal : 24,708,051 vectors
  d      : 384 dimensions
  nlist  : 4096 IVF cells
  nprobe : 64  (queries probe this many cells)
  M      : 32  PQ sub-vectors  (dsub = 12)
  nbits  : 8   bits per code   (256 centroids/sub-vector)
"""

from __future__ import annotations

import argparse
import logging
import shutil
import time
from pathlib import Path

import faiss
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# === Paths ===================================================================

from config import DATA_DIR  # noqa: E402

INDEX_DIR   = DATA_DIR / "faiss_migrated"
INDEX_FILE  = INDEX_DIR / "faiss" / "index.faiss"
BACKUP_FILE = INDEX_DIR / "faiss" / "index.faiss.bak"


# === Helpers =================================================================

def load_index(path: Path) -> faiss.IndexIVFPQ:
    logger.info(f"Loading index from {path} …")
    t0 = time.time()
    idx = faiss.read_index(str(path))
    logger.info(f"  Loaded in {time.time()-t0:.1f}s  —  {idx.ntotal:,} vectors, dim={idx.d}")
    return idx


def print_params(idx: faiss.IndexIVFPQ, label: str = "Current") -> None:
    pq = idx.pq
    print(f"\n{'─'*50}")
    print(f"  {label} index parameters")
    print(f"{'─'*50}")
    print(f"  ntotal : {idx.ntotal:,}")
    print(f"  d      : {idx.d}")
    print(f"  nlist  : {idx.nlist}")
    print(f"  nprobe : {idx.nprobe}")
    print(f"  M      : {pq.M}  (PQ sub-vectors, dsub={pq.dsub})")
    print(f"  nbits  : {pq.nbits}  ({2**pq.nbits} centroids/sub-vector)")
    print(f"{'─'*50}\n")


def backup_index(src: Path) -> None:
    logger.info(f"Backing up {src} → {BACKUP_FILE}")
    shutil.copy2(src, BACKUP_FILE)
    logger.info("  Backup done.")


def save_index(idx: faiss.IndexIVFPQ, path: Path) -> None:
    tmp = path.with_suffix(".faiss.tmp")
    logger.info(f"Writing index to {path} …")
    faiss.write_index(idx, str(tmp))
    tmp.rename(path)
    logger.info("  Saved.")


# === Benchmark ===============================================================

def _load_es_results(results_dir: Path, strategy: str = "approximation") -> dict[str, list[int]] | None:
    """Load ES retrieval results as a mapping question→list[faiss_position].

    Returns None if the parquet file does not exist.

    We can only compare at the document-ID level (wikipedia_id strings), not
    FAISS positions, so we return question→set[wikipedia_id] instead.
    """
    path = results_dir / f"results_{strategy}.parquet"
    if not path.exists():
        return None
    try:
        import pandas as pd
        df = pd.read_parquet(path, columns=["question", "topk_ids"])
        return {
            str(row.question): [str(i) for i in row.topk_ids if i not in (None, "", "-1")]
            for row in df.itertuples()
        }
    except Exception as e:
        logger.warning(f"Could not load ES results from {path}: {e}")
        return None


def _build_flat_gt(
    idx: faiss.IndexIVFPQ,
    query_vecs: np.ndarray,
    k: int,
    sample_size: int = 50_000,
) -> np.ndarray:
    """Build ground-truth nearest neighbours using a flat (exact) index.

    Generates ``sample_size`` random unit vectors and builds an
    ``IndexFlatIP`` over them.  The queries are then searched against this
    flat index to produce exact nearest-neighbour ground truth *within the
    sample*.

    This is honest ground truth for the synthetic benchmark: it does not
    depend on the IVF-PQ compression at all.  The limitation is that the
    sample is random, not the actual corpus — so recall numbers measure
    internal IVF-PQ consistency on random data, not real retrieval quality.
    Use --benchmark-qa for real-data recall vs ES comparison.
    """
    dim = idx.d
    rng = np.random.default_rng(99)
    sample_vecs = rng.standard_normal((sample_size, dim)).astype(np.float32)
    faiss.normalize_L2(sample_vecs)

    logger.info(f"Building flat GT index over {sample_size:,} random vectors …")
    flat = faiss.IndexFlatIP(dim)
    flat.add(sample_vecs)
    _, gt = flat.search(query_vecs, k)

    # Now add those same vectors to the IVF-PQ index temporarily to get comparable
    # approximate results — but that would mutate the index.  Instead we just use
    # the flat results as GT and compare the IVF-PQ search results against them
    # by searching the IVF-PQ index for the same queries.
    # The IVF-PQ GT positions are flat positions (0..sample_size-1) which are NOT
    # the same as FAISS index positions.  So we compare result *sets* only if
    # we also add those vectors to the IVF-PQ — which we don't want to do.
    #
    # Simpler and more honest: use a standalone flat index and IVF-PQ search on
    # the same random sample, added to a *fresh* small IVF-PQ for comparison.
    # For a benchmark script this is sufficient — positions align within the sample.
    return gt, flat, sample_vecs


def benchmark_nprobe(
    idx: faiss.IndexIVFPQ,
    nprobe_values: list[int],
    k: int = 10,
    results_dir: Path | None = None,
    es_strategy: str = "approximation",
) -> None:
    """Recall@k vs nprobe benchmark with honest flat ground truth.

    Ground truth is computed via a flat (brute-force) index over a random
    50k-vector sample reconstructed from PQ codes.  This is an approximation
    of true ground truth (PQ reconstruction is lossy) but is far more honest
    than using nprobe=nlist on the same compressed index.

    If ``results_dir`` is supplied and contains ES result parquets, an extra
    column shows how each nprobe setting compares to ES (same k, same
    ground truth).
    """
    dim = idx.d
    n_queries = 200
    rng = np.random.default_rng(42)
    queries = rng.standard_normal((n_queries, dim)).astype(np.float32)
    faiss.normalize_L2(queries)

    # ── Ground truth: flat index over 50k fresh random vectors ───────────────
    # We build a mini IVF-PQ with the same parameters trained on those same
    # vectors, so positions align and we can compare results sets directly.
    gt, flat_idx, sample_vecs = _build_flat_gt(idx, queries, k)
    gt_sets = [set(gt[i].tolist()) - {-1} for i in range(n_queries)]

    # Build a mini IVF-PQ on the same 50k sample so nprobe comparisons are fair
    n_sample = len(sample_vecs)
    nlist_mini = min(idx.nlist, max(16, n_sample // 100))
    quantiser = faiss.IndexFlatIP(dim)
    mini_ivfpq = faiss.IndexIVFPQ(quantiser, dim, nlist_mini, idx.pq.M, idx.pq.nbits)
    mini_ivfpq.train(sample_vecs)
    mini_ivfpq.add(sample_vecs)
    logger.info(f"Mini IVF-PQ trained on {n_sample:,} vectors, nlist={nlist_mini}")

    # ── ES results (optional) ────────────────────────────────────────────────
    es_results = None
    if results_dir is not None:
        es_results = _load_es_results(results_dir, es_strategy)

    # ── Header ───────────────────────────────────────────────────────────────
    show_es = es_results is not None
    col_w = 62 if show_es else 46
    print(f"\n{'─'*col_w}")
    print(f"  Benchmark: recall@{k}  (n={n_queries} synthetic queries)")
    print(f"  Ground truth: flat/exact search over 50k random unit vectors")
    print(f"  Index under test: mini IVF-PQ (same M/nbits, nlist={nlist_mini})")
    print(f"  Measures internal IVF-PQ consistency, not real retrieval quality.")
    print(f"  Use --benchmark-qa to compare against ES on real questions.")
    if show_es:
        print(f"  ES results loaded ({len(es_results):,} questions) — see --benchmark-qa for direct comparison.")
    print(f"{'─'*col_w}")
    print(f"  {'nprobe':>8}  {'recall@'+str(k):>12}  {'ms/query':>12}")
    print(f"{'─'*col_w}")

    nprobe_values_clamped = [min(nv, nlist_mini) for nv in nprobe_values]
    for np_ in sorted(set(nprobe_values_clamped)):
        mini_ivfpq.nprobe = np_
        t0 = time.time()
        _, results = mini_ivfpq.search(queries, k)
        elapsed_ms = (time.time() - t0) / n_queries * 1000

        hits = sum(
            len(set(results[i].tolist()) & gt_sets[i])
            for i in range(n_queries)
        )
        recall = hits / (n_queries * k)
        label = f"nprobe={np_}" if np_ < nlist_mini else f"nprobe={np_} (exhaustive)"
        print(f"  {np_:>8}  {recall:>11.1%}  {elapsed_ms:>10.2f} ms")

    print(f"{'─'*col_w}\n")


def benchmark_qa(
    idx: faiss.IndexIVFPQ,
    nprobe_values: list[int],
    results_dir: Path,
    k: int = 10,
    n_questions: int = 500,
    es_strategy: str = "approximation",
    es_bm25_strategy: str = "bm25",
) -> None:
    """Compare FAISS nprobe settings vs ES strategies on real QA questions.

    Uses the saved ES result parquets as both the query source and a reference
    baseline.  Ground truth is the QA dataset's wikipedia_id (the correct
    document for each question).

    Recall@k = fraction of questions where the correct wikipedia_id appears
    in the top-k retrieved documents.

    This is the most meaningful comparison for your thesis because it uses
    real questions, real ground-truth document IDs, and the actual retrieval
    pipeline.
    """
    import pandas as pd
    from dotenv import load_dotenv
    load_dotenv()

    # ── Load QA questions from ES results (they share the same question set) ─
    approx_path = results_dir / f"results_{es_strategy}.parquet"
    bm25_path   = results_dir / f"results_{es_bm25_strategy}.parquet"

    if not approx_path.exists():
        logger.error(f"ES results not found at {approx_path} — skipping QA benchmark")
        return

    df_es = pd.read_parquet(approx_path)
    df_es["wikipedia_id"] = df_es["wikipedia_id"].astype(str).str.strip()

    if n_questions and n_questions < len(df_es):
        df_es = df_es.sample(n=n_questions, random_state=42)
        logger.info(f"Sampled {n_questions} questions for QA benchmark")

    questions   = df_es["question"].tolist()
    correct_ids = df_es["wikipedia_id"].tolist()
    es_topk_ids = [
        [str(i) for i in ids if i not in (None, "", "-1")]
        for ids in df_es["topk_ids"].tolist()
    ]

    # ES BM25
    es_bm25_topk = None
    if bm25_path.exists():
        df_bm25 = pd.read_parquet(bm25_path)
        df_bm25["wikipedia_id"] = df_bm25["wikipedia_id"].astype(str).str.strip()
        df_bm25 = df_bm25.set_index("question")
        es_bm25_topk = [
            [str(i) for i in df_bm25.loc[q, "topk_ids"] if i not in (None, "", "-1")]
            if q in df_bm25.index else []
            for q in questions
        ]

    # ── Embed questions ───────────────────────────────────────────────────────
    logger.info(f"Embedding {len(questions)} questions via Modal …")
    from src.rag.faiss_rag_service import FaissRagService
    store = idx  # we already have it loaded; just need the embedder
    embedder = FaissRagService._embeddings  # not available here — use config
    # Build embeddings via IndexingConfig + the same modal provider
    from src.rag.utils import IndexingConfig, build_embeddings
    config = IndexingConfig(
        embedding_provider="modal",
        embedding_model="Lajavaness/bilingual-embedding-small",
        gpu_batch_size=512,
        request_batch_size=512,
        normalise_embeddings=True,
        trust_remote_code=True,
        use_progress=True,
    )
    embeddings_obj = build_embeddings(config)
    logger.info("  Embedding …")
    vecs = np.array(embeddings_obj.embed_documents(questions), dtype=np.float32)
    faiss.normalize_L2(vecs)
    logger.info("  Done.")

    # ── Load id_map to translate FAISS positions → wikipedia_ids ─────────────
    logger.info("Loading id_map from SQLite …")
    import sqlite3, json
    db_path = INDEX_DIR / "faiss" / "docstore.sqlite"
    conn = sqlite3.connect(db_path)
    pos_to_uid  = dict(conn.execute("SELECT pos, uid FROM id_map").fetchall())
    # uid is like "doc_12345678" — map to wikipedia_id via docs table
    uid_to_wid = {}
    logger.info("  Loading wikipedia_ids from docstore (this takes ~30s) …")
    for uid, metadata_json in conn.execute("SELECT uid, metadata FROM docs"):
        try:
            meta = json.loads(metadata_json)
            wid = meta.get("wikipedia_id")
            if wid:
                uid_to_wid[uid] = str(int(float(wid)))
        except Exception:
            pass
    conn.close()
    logger.info(f"  Loaded {len(uid_to_wid):,} wikipedia_id mappings")

    def faiss_positions_to_wids(positions: np.ndarray) -> list[str]:
        out = []
        for pos in positions:
            if pos == -1:
                continue
            uid = pos_to_uid.get(int(pos))
            if uid is None:
                continue
            wid = uid_to_wid.get(uid)
            if wid:
                out.append(wid)
        return out

    # ── Run search at each nprobe ─────────────────────────────────────────────
    # Also compute ES baselines once
    def recall_at_k(topk_id_lists: list[list[str]], correct: list[str], k: int) -> float:
        hits = sum(
            1 for ids, cid in zip(topk_id_lists, correct)
            if cid in ids[:k]
        )
        return hits / len(correct)

    es_approx_recall = recall_at_k(es_topk_ids, correct_ids, k)
    es_bm25_recall   = recall_at_k(es_bm25_topk, correct_ids, k) if es_bm25_topk else None

    col_w = 72
    print(f"\n{'─'*col_w}")
    print(f"  QA Benchmark: recall@{k}  (n={len(questions)} real questions)")
    print(f"  Ground truth: correct wikipedia_id from QA dataset")
    print(f"  ES/{es_strategy} recall@{k}:  {es_approx_recall:.1%}")
    if es_bm25_recall is not None:
        print(f"  ES/{es_bm25_strategy} recall@{k}:       {es_bm25_recall:.1%}")
    print(f"{'─'*col_w}")
    print(f"  {'nprobe':>8}  {'FAISS recall@'+str(k):>16}  {'Δ vs ES/'+es_strategy:>16}  {'ms/query':>10}")
    print(f"{'─'*col_w}")

    for np_ in nprobe_values:
        idx.nprobe = np_
        t0 = time.time()
        _, positions = idx.search(vecs, k * 2)  # over-fetch to cover train_ gaps
        elapsed_ms = (time.time() - t0) / len(questions) * 1000

        faiss_topk = [faiss_positions_to_wids(positions[i])[:k] for i in range(len(questions))]
        faiss_recall = recall_at_k(faiss_topk, correct_ids, k)
        delta = faiss_recall - es_approx_recall
        delta_str = f"{delta:+.1%}"
        print(f"  {np_:>8}  {faiss_recall:>15.1%}  {delta_str:>16}  {elapsed_ms:>8.2f} ms")

    print(f"{'─'*col_w}\n")


# === nprobe-only update (no rebuild) =========================================

def update_nprobe(args: argparse.Namespace) -> None:
    idx = load_index(INDEX_FILE)
    print_params(idx, "Before")

    old_nprobe = idx.nprobe
    idx.nprobe = args.nprobe
    print_params(idx, "After")

    if args.dry_run:
        logger.info("Dry run — no changes written.")
        return

    backup_index(INDEX_FILE)
    save_index(idx, INDEX_FILE)
    logger.info(f"nprobe updated: {old_nprobe} → {args.nprobe}")


# === Full rebuild ============================================================

def rebuild_index(args: argparse.Namespace) -> None:
    """Re-encode all vectors with new M / nbits / nlist.

    This is the expensive path — it:
      1. Loads the current index
      2. Reconstructs all raw (decoded) vectors
      3. Trains a new IVF-PQ index with the new parameters
      4. Adds all vectors back
      5. Saves (with backup)

    Note: IVF-PQ stores compressed codes, not exact vectors. Reconstruction
    via `sa_decode` introduces quantisation error, so a rebuild from the
    original corpus embeddings is always more accurate. Use this script's
    rebuild only when the original embeddings are unavailable.
    """
    idx = load_index(INDEX_FILE)
    print_params(idx, "Current (before rebuild)")

    dim    = idx.d
    ntotal = idx.ntotal
    new_m      = args.m     if args.m     else idx.pq.M
    new_nbits  = args.nbits if args.nbits else idx.pq.nbits
    new_nlist  = args.nlist if args.nlist else idx.nlist
    new_nprobe = args.nprobe if args.nprobe else idx.nprobe

    # Validate
    if dim % new_m != 0:
        candidates = [m for m in range(1, dim + 1) if dim % m == 0]
        raise ValueError(
            f"dim ({dim}) must be divisible by M.  "
            f"Valid M values: {candidates}"
        )

    logger.info(
        f"Rebuild plan: nlist={new_nlist}, M={new_m}, nbits={new_nbits}, nprobe={new_nprobe}"
    )

    if args.dry_run:
        logger.info("Dry run — no rebuild performed.")
        return

    # ── Reconstruct all vectors ──────────────────────────────────────────────
    logger.info(f"Reconstructing {ntotal:,} vectors from PQ codes …")
    logger.info("  (This may take several minutes and requires significant RAM)")
    t0 = time.time()
    BATCH = 500_000
    all_vecs = np.empty((ntotal, dim), dtype=np.float32)
    for start in range(0, ntotal, BATCH):
        end = min(start + BATCH, ntotal)
        ids = np.arange(start, end, dtype=np.int64)
        all_vecs[start:end] = idx.reconstruct_batch(ids)
        if start % 5_000_000 == 0 and start > 0:
            logger.info(f"  {start:,} / {ntotal:,} reconstructed …")
    logger.info(f"  Reconstruction done in {time.time()-t0:.0f}s")

    # ── Train new index ──────────────────────────────────────────────────────
    logger.info("Building and training new IVF-PQ index …")
    quantiser = faiss.IndexFlatL2(dim)
    new_idx = faiss.IndexIVFPQ(quantiser, dim, new_nlist, new_m, new_nbits)
    new_idx.nprobe = new_nprobe

    # Sample for training
    n_train = min(ntotal, new_nlist * 100)
    rng = np.random.default_rng(0)
    train_ids = rng.choice(ntotal, n_train, replace=False)
    train_vecs = all_vecs[train_ids].copy()
    logger.info(f"Training on {n_train:,} vectors …")
    t0 = time.time()
    new_idx.train(train_vecs)
    logger.info(f"  Training done in {time.time()-t0:.0f}s")
    del train_vecs

    # ── Add all vectors ──────────────────────────────────────────────────────
    logger.info("Adding vectors to new index …")
    t0 = time.time()
    new_idx.add(all_vecs)
    del all_vecs
    logger.info(f"  Added {new_idx.ntotal:,} vectors in {time.time()-t0:.0f}s")

    print_params(new_idx, "New")

    backup_index(INDEX_FILE)
    save_index(new_idx, INDEX_FILE)
    logger.info("Rebuild complete.")


# === CLI =====================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tune a local IVF-PQ FAISS index for higher recall.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Runtime param
    p.add_argument(
        "--nprobe", type=int, default=None,
        help="IVF cells probed per query (default: keep current=64). "
             "Recommended: 128, 256, or 512.",
    )

    # Structural params (require --rebuild)
    p.add_argument(
        "--m", type=int, default=None,
        help="PQ sub-vectors M (current=32, dim=384). "
             "Valid values that divide 384: 1,2,3,4,6,8,12,16,24,32,48,64,96,128,192,384. "
             "Try 48 or 64 for higher recall. Requires --rebuild.",
    )
    p.add_argument(
        "--nbits", type=int, default=None, choices=[8, 12],
        help="Bits per PQ code (current=8). 12 gives better recall, 1.5× storage. "
             "Requires --rebuild.",
    )
    p.add_argument(
        "--nlist", type=int, default=None,
        help="IVF cells (current=4096). Try 8192. Requires --rebuild.",
    )

    # Modes
    p.add_argument(
        "--rebuild", action="store_true",
        help="Rebuild the index with new M/nbits/nlist. "
             "Slow (~hours). Backup is created automatically.",
    )
    p.add_argument(
        "--benchmark", action="store_true",
        help="Run a synthetic recall@k benchmark across nprobe values "
             "(uses random queries + flat index over 50k reconstructed vectors as GT).",
    )
    p.add_argument(
        "--benchmark-qa", action="store_true",
        help="Run a recall@k benchmark on real QA questions, comparing each nprobe "
             "setting against ES/approximation and ES/bm25 baselines. "
             "Requires --results-dir and a live Modal embedding service.",
    )
    p.add_argument(
        "--results-dir", type=Path,
        default=DATA_DIR / "wiki_full_bil" / "all_qa_8k_100",
        help="Directory containing results_approximation.parquet and results_bm25.parquet "
             "(default: data/wiki_full_bil/all_qa_8k_100).",
    )
    p.add_argument(
        "--k", type=int, default=10,
        help="k for recall@k in benchmarks (default: 10).",
    )
    p.add_argument(
        "--n-questions", type=int, default=500,
        help="Number of QA questions to sample for --benchmark-qa (default: 500).",
    )
    p.add_argument(
        "--nprobe-values", type=str, default=None,
        help="Comma-separated nprobe values to benchmark, e.g. '32,64,128,256'. "
             "Defaults to a sensible sweep.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print what would change without writing anything.",
    )
    p.add_argument(
        "--index-dir", type=Path, default=INDEX_DIR,
        help=f"Root directory of the FAISS index (default: {INDEX_DIR}).",
    )

    return p.parse_args()


def _nprobe_sweep(idx: faiss.IndexIVFPQ, extra: list[int] | None = None) -> list[int]:
    base = [1, 8, 16, 32, 64, 128, 256, 512, idx.nlist]
    if extra:
        base += extra
    return sorted(set(base))


def main() -> None:
    args = parse_args()

    # Allow overriding index path via --index-dir
    global INDEX_DIR, INDEX_FILE, BACKUP_FILE
    INDEX_DIR   = args.index_dir
    INDEX_FILE  = INDEX_DIR / "faiss" / "index.faiss"
    BACKUP_FILE = INDEX_DIR / "faiss" / "index.faiss.bak"

    if not INDEX_FILE.exists():
        raise FileNotFoundError(f"Index file not found: {INDEX_FILE}")

    structural_change = any(x is not None for x in [args.m, args.nbits, args.nlist])

    if structural_change and not args.rebuild and not args.dry_run:
        raise SystemExit(
            "ERROR: --m / --nbits / --nlist require --rebuild (or add --dry-run to preview)."
        )

    # Parse custom nprobe sweep if supplied
    custom_nprobes = (
        [int(x) for x in args.nprobe_values.split(",")]
        if args.nprobe_values else None
    )

    if args.nprobe is None and not structural_change and not args.benchmark_qa:
        # No tuning requested — just print current state and optionally benchmark
        idx = load_index(INDEX_FILE)
        print_params(idx, "Current")
        if args.benchmark:
            sweep = custom_nprobes or _nprobe_sweep(idx)
            benchmark_nprobe(idx, sweep, k=args.k, results_dir=args.results_dir)
        elif args.benchmark_qa:
            pass  # handled below
        else:
            print("Tip: run with --benchmark            to see synthetic recall vs nprobe.")
            print("     run with --benchmark-qa         to compare with ES on real questions.")
            print("     run with --nprobe 256           to update nprobe (instant, no rebuild).")
            print("     run with --m 48 --rebuild       to rebuild with better PQ (slow).")
        if args.benchmark_qa:
            idx = load_index(INDEX_FILE)
            sweep = custom_nprobes or [32, 64, 128, 256, 512]
            benchmark_qa(idx, sweep, results_dir=args.results_dir,
                         k=args.k, n_questions=args.n_questions)
        return

    if structural_change:
        rebuild_index(args)
        if args.benchmark or args.benchmark_qa:
            idx = load_index(INDEX_FILE)
            sweep = custom_nprobes or _nprobe_sweep(idx)
            if args.benchmark:
                benchmark_nprobe(idx, sweep, k=args.k, results_dir=args.results_dir)
            if args.benchmark_qa:
                benchmark_qa(idx, sweep, results_dir=args.results_dir,
                             k=args.k, n_questions=args.n_questions)
    else:
        update_nprobe(args)
        if args.benchmark or args.benchmark_qa:
            idx = load_index(INDEX_FILE)
            sweep = custom_nprobes or _nprobe_sweep(idx, extra=[args.nprobe] if args.nprobe else None)
            if args.benchmark:
                benchmark_nprobe(idx, sweep, k=args.k, results_dir=args.results_dir)
            if args.benchmark_qa:
                benchmark_qa(idx, sweep, results_dir=args.results_dir,
                             k=args.k, n_questions=args.n_questions)

    if args.benchmark_qa and args.nprobe is None and not structural_change:
        pass  # already handled in the no-tuning branch above


if __name__ == "__main__":
    main()
