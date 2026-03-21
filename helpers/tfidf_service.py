"""TF-IDF per-decile analysis service.

Public API
----------
``load_or_compute_tfidf_stats``
    Cache-aware entry point: reads cached stats from ``metadata.json`` when
    available, otherwise runs the full sampling + vectorisation pipeline and
    writes the results back to the same file.

``plot_tfidf_stats``
    Produces a 3-panel matplotlib figure (vocabulary breadth, keyword
    intensity, term specificity) from the stats dict and saves it to disk.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Keys used inside metadata.json
_CACHE_KEY = "decile_tfidf_stats"
_NUM_DECILES = 10


def _cache_key(chunk_size: int | None, chunk_overlap: int, sample_per_decile: int) -> str:
    """Return the metadata.json key for the given chunking + sample settings."""
    if chunk_size:
        return f"{_CACHE_KEY}_chunks_{chunk_size}_{chunk_overlap}_n{sample_per_decile}"
    return f"{_CACHE_KEY}_n{sample_per_decile}"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _chunk_texts(
    texts: list[str],
    deciles: list[int],
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[list[str], list[int]]:
    """Split each document into chunks using RecursiveCharacterTextSplitter.

    Each chunk inherits the popularity decile of its parent document.

    Returns
    -------
    texts   : flat list of chunk strings
    deciles : parallel list of 0-based decile indices
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    out_texts: list[str] = []
    out_deciles: list[int] = []
    for text, d in zip(texts, deciles):
        chunks = splitter.split_text(text)
        out_texts.extend(chunks)
        out_deciles.extend([d] * len(chunks))
    return out_texts, out_deciles


def _reservoir_sample_corpus(
    corpus_path: Path | str,
    boundaries: np.ndarray,
    sample_per_decile: int,
    *,
    seed: int = 42,
) -> tuple[list[str], list[int]]:
    """Stream *corpus_path* and reservoir-sample up to *sample_per_decile*
    documents per decile.  Never holds more than one batch in RAM.

    Returns
    -------
    texts   : flat list of document strings
    deciles : parallel list of 0-based decile indices
    """
    import pyarrow.parquet as pq
    from tqdm.auto import tqdm
    from helpers.decile_utils import assign_decile

    random.seed(seed)
    reservoirs: dict[int, list[str]] = {d: [] for d in range(_NUM_DECILES)}
    counts: dict[int, int] = {d: 0 for d in range(_NUM_DECILES)}

    pf = pq.ParquetFile(str(corpus_path))
    for batch in tqdm(
        pf.iter_batches(batch_size=100_000, columns=["popularity_avg", "text"]),
        desc="Sampling corpus",
    ):
        batch_df = batch.to_pandas()
        batch_df = batch_df.dropna(subset=["popularity_avg", "text"])
        batch_df = batch_df[batch_df["popularity_avg"] >= 0].copy()

        # assign_decile returns a pandas IntegerArray — convert to plain ndarray
        decile_arr = assign_decile(batch_df["popularity_avg"], boundaries).to_numpy(
            dtype=int, na_value=-1
        )
        batch_df["_d"] = decile_arr

        for d in range(_NUM_DECILES):
            for text in batch_df.loc[batch_df["_d"] == d, "text"]:
                counts[d] += 1
                if len(reservoirs[d]) < sample_per_decile:
                    reservoirs[d].append(text)
                else:
                    j = random.randint(0, counts[d] - 1)
                    if j < sample_per_decile:
                        reservoirs[d][j] = text

        # Early exit: all deciles full and we've seen a reasonable oversample
        if all(len(reservoirs[d]) >= sample_per_decile for d in range(_NUM_DECILES)):
            sampled_so_far = sum(counts.values())
            if sampled_so_far >= sample_per_decile * 50:
                print(f"  Early exit after {sampled_so_far:,} docs seen")
                break

    texts: list[str] = []
    deciles: list[int] = []
    for d in range(_NUM_DECILES):
        for text in reservoirs[d]:
            texts.append(text)
            deciles.append(d)

    n_deciles_covered = len({d for d in deciles})
    print(f"  Sampled {len(texts):,} documents across {n_deciles_covered} deciles")
    return texts, deciles


def _compute_stats(
    texts: list[str],
    deciles: list[int],
    max_features: int,
    source_deciles: list[int] | None = None,
) -> dict[str, dict]:
    """Fit TF-IDF vectorizer and compute per-decile summary statistics.

    Parameters
    ----------
    source_deciles : list[int] | None
        If texts are chunks, pass the decile list of the *original documents*
        so that ``n_source_docs`` is counted correctly.  When ``None``,
        ``n_source_docs`` equals ``n_items``.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    print("  Fitting TF-IDF vectorizer…")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        strip_accents="unicode",
        sublinear_tf=True, 
        min_df=2,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)  # CSR (n_docs, vocab)
    idf_values = vectorizer.idf_                    # shape (vocab,)
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_):,} terms")

    all_deciles_arr = np.array(deciles)
    source_counts = {}
    if source_deciles is not None:
        src_arr = np.array(source_deciles)
        for d in range(_NUM_DECILES):
            source_counts[d] = int((src_arr == d).sum())
    stats: dict[str, dict] = {}

    texts_arr = np.array(texts, dtype=object)

    for d in range(_NUM_DECILES):
        mask = all_deciles_arr == d
        n_docs_d = int(mask.sum())
        if n_docs_d == 0:
            stats[str(d)] = {}
            continue

        sub = tfidf_matrix[mask]  # (n_d, vocab) CSR

        # A. Unique terms per document (non-zero columns per row)
        unique_per_doc = np.diff(sub.indptr).astype(float)

        # B. Mean TF-IDF score of non-zero entries → keyword intensity
        nnz = sub.nnz
        mean_score = float(sub.sum() / nnz) if nnz > 0 else 0.0

        # C. IDF of terms used in this decile → term specificity
        used_idx = np.unique(sub.nonzero()[1])
        mean_idf = float(idf_values[used_idx].mean()) if len(used_idx) > 0 else 0.0
        vocab_cover = float(len(used_idx) / tfidf_matrix.shape[1])

        # D. Average character length of each item (document or chunk)
        lengths = np.array([len(t) for t in texts_arr[mask]])
        mean_chunk_len = float(np.mean(lengths))
        std_chunk_len = float(np.std(lengths))

        # E. Mean sublinear TF (recover TF component: tfidf / idf per entry)
        coo = sub.tocoo()
        if coo.nnz > 0:
            tf_vals = coo.data / idf_values[coo.col]
            mean_tf = float(np.mean(tf_vals))
        else:
            mean_tf = 0.0

        stats[str(d)] = {
            "n_docs_sampled": n_docs_d,
            "n_source_docs": source_counts.get(d, n_docs_d),
            "mean_unique_terms": float(np.mean(unique_per_doc)),
            "std_unique_terms": float(np.std(unique_per_doc)),
            "mean_tfidf_score": mean_score,
            "mean_idf_of_used_terms": mean_idf,
            "vocab_coverage": vocab_cover,
            "mean_chunk_length": mean_chunk_len,
            "std_chunk_length": std_chunk_len,
            "mean_tf_score": mean_tf,
        }

    return stats


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_or_compute_corpus_vectorizer(
    metadata_path: Path | str,
    corpus_path: Path | str,
    boundaries: np.ndarray,
    *,
    sample_per_decile: int = 20_000,
    max_features: int = 50_000,
    chunk_size: int | None = 1000,
    chunk_overlap: int = 100,
    force_recompute: bool = False,
) -> "TfidfVectorizer":  # type: ignore[name-defined]  # noqa: F821
    """Return a ``TfidfVectorizer`` fitted on a stratified corpus sample.

    The fitted vectorizer is cached as a ``tfidf_vectorizer_<key>.pkl`` file
    in the same directory as *metadata_path*.  On subsequent calls the cached
    file is loaded directly (fast) unless *force_recompute* is ``True``.

    The vectorizer is fitted with the same settings used by
    :func:`load_or_compute_tfidf_stats` so that IDF values are consistent
    across both functions.

    Parameters
    ----------
    metadata_path : Path | str
        Path to ``metadata.json`` — used only to derive the cache directory.
    corpus_path : Path | str
        Path to the Parquet corpus file.
    boundaries : np.ndarray
        Decile boundary array as returned by ``boundaries_for``.
    sample_per_decile : int
        Documents sampled per decile when building the vectorizer.
    max_features : int
        Vocabulary cap — must match the value passed to
        :func:`load_or_compute_tfidf_stats`.
    chunk_size : int | None
        If set, documents are chunked before fitting (same as in
        :func:`load_or_compute_tfidf_stats`).
    chunk_overlap : int
        Chunk overlap in characters.
    force_recompute : bool
        Skip the cache and recompute from scratch.

    Returns
    -------
    sklearn.feature_extraction.text.TfidfVectorizer
        A fitted vectorizer whose ``idf_`` array reflects corpus-wide term
        rarity.
    """
    import pickle
    from sklearn.feature_extraction.text import TfidfVectorizer

    metadata_path = Path(metadata_path)
    cache_dir = metadata_path.parent
    key = _cache_key(chunk_size, chunk_overlap, sample_per_decile)
    pkl_path = cache_dir / f"tfidf_vectorizer_{key}.pkl"

    if not force_recompute and pkl_path.exists():
        print(f"✓ Loading corpus vectorizer from cache: {pkl_path.name}")
        with open(pkl_path, "rb") as f:
            return pickle.load(f)

    print(
        f"No vectorizer cache found — sampling corpus and fitting "
        f"(sample_per_decile={sample_per_decile}, max_features={max_features})…"
    )
    texts, deciles = _reservoir_sample_corpus(corpus_path, boundaries, sample_per_decile)

    if chunk_size:
        print(f"  Chunking {len(texts):,} docs (size={chunk_size}, overlap={chunk_overlap})…")
        texts, deciles = _chunk_texts(texts, deciles, chunk_size, chunk_overlap)
        print(f"  → {len(texts):,} chunks")

    print(f"  Fitting TfidfVectorizer on {len(texts):,} texts…")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        strip_accents="unicode",
        sublinear_tf=True,
        min_df=2,
    )
    vectorizer.fit(texts)
    print(f"  Vocabulary: {len(vectorizer.vocabulary_):,} terms")

    with open(pkl_path, "wb") as f:
        pickle.dump(vectorizer, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✓ Saved corpus vectorizer → {pkl_path.name}")

    return vectorizer


def compute_tfidf_stats_for_ids(
    query_df: "pd.DataFrame",
    corpus_path: Path | str,
    *,
    decile_col: str,
    group_col: str | None = None,
    corpus_vectorizer: "TfidfVectorizer | None" = None,  # type: ignore[name-defined]  # noqa: F821
    max_features: int = 50_000,
    chunk_size: int | None = None,
    chunk_overlap: int = 100,
) -> "dict[str, dict[int, dict]]":
    """Compute per-(group, decile) TF-IDF statistics from the target documents
    of a set of queries, using corpus-wide IDF values when available.

    For each query row in *query_df* the function fetches the corresponding
    gold document text from *corpus_path* by ``wikipedia_id``.  It then uses
    *corpus_vectorizer* (a ``TfidfVectorizer`` pre-fitted on the full corpus)
    to **transform** (not re-fit) those texts, ensuring that IDF values reflect
    term rarity across the whole corpus rather than just the query subset.

    When *corpus_vectorizer* is ``None`` the function falls back to fitting a
    new vectorizer on the query target docs only (original behaviour).

    Parameters
    ----------
    query_df : pd.DataFrame
        Must contain ``wikipedia_id`` and the column named by *decile_col*.
        If *group_col* is provided that column must also be present.
    corpus_path : Path | str
        Path to the Parquet corpus (``wikipedia_id`` int64, ``text`` string).
    decile_col : str
        Name of the 0-based decile column in *query_df*.
    group_col : str | None
        Column to split results by (e.g. ``"dataset"``).  ``None`` → single
        group keyed ``"all"``.
    corpus_vectorizer : TfidfVectorizer | None
        A ``TfidfVectorizer`` already fitted on the corpus (e.g. from
        :func:`load_or_compute_corpus_vectorizer`).  When supplied, IDF values
        come from the corpus; when ``None``, a new vectorizer is fitted on the
        query target docs.
    max_features : int
        Vocabulary cap — only used when *corpus_vectorizer* is ``None``.
    chunk_size : int | None
        When set, each document is split into chunks before TF-IDF is applied.
    chunk_overlap : int
        Overlap between consecutive chunks.

    Returns
    -------
    dict mapping group_label → dict mapping decile_int → stats_dict.

    Stats dict keys: ``n_docs_sampled``, ``n_source_docs``,
    ``mean_unique_terms``, ``std_unique_terms``, ``mean_tfidf_score``,
    ``mean_idf_of_used_terms``, ``vocab_coverage``, ``mean_chunk_length``,
    ``std_chunk_length``, ``mean_tf_score``.
    """
    import pyarrow.parquet as pq
    from tqdm.auto import tqdm
    from sklearn.feature_extraction.text import TfidfVectorizer

    corpus_path = Path(corpus_path)

    # ── 1. Fetch target document texts from corpus ────────────────────────────
    needed_ids = set(str(wid).strip() for wid in query_df["wikipedia_id"].dropna())
    print(f"  Fetching {len(needed_ids):,} unique document IDs from corpus…")

    id_to_text: dict[str, str] = {}
    pf = pq.ParquetFile(str(corpus_path))
    for batch in tqdm(
        pf.iter_batches(batch_size=100_000, columns=["wikipedia_id", "text"]),
        desc="Scanning corpus",
    ):
        batch_df = batch.to_pandas()
        batch_df["wikipedia_id"] = batch_df["wikipedia_id"].astype(str).str.strip()
        mask = batch_df["wikipedia_id"].isin(needed_ids)
        for wid, text in zip(
            batch_df.loc[mask, "wikipedia_id"], batch_df.loc[mask, "text"]
        ):
            if wid not in id_to_text and isinstance(text, str) and text.strip():
                id_to_text[wid] = text
        if len(id_to_text) >= len(needed_ids):
            break

    print(f"  Matched {len(id_to_text):,} / {len(needed_ids):,} IDs in corpus")

    # ── 2. Build per-group (text, decile) lists ───────────────────────────────
    if group_col and group_col in query_df.columns:
        groups = sorted(query_df[group_col].dropna().astype(str).unique())
    else:
        group_col = None
        groups = ["all"]

    group_texts: dict[str, list[tuple[str, int]]] = {g: [] for g in groups}
    for _, row in query_df.iterrows():
        wid = str(row["wikipedia_id"]).strip()
        text = id_to_text.get(wid)
        if text is None:
            continue
        decile = int(row[decile_col])
        grp = str(row[group_col]) if group_col else "all"
        group_texts[grp].append((text, decile))

    # ── 3. Assemble flat text list (optionally chunked) ───────────────────────
    all_texts: list[str] = []
    all_labels: list[tuple[str, int]] = []   # (group, decile)
    all_source_labels: list[tuple[str, int]] = []

    for grp, pairs in group_texts.items():
        if not pairs:
            continue
        texts_g, deciles_g = zip(*pairs)
        texts_g = list(texts_g)
        deciles_g = list(deciles_g)

        if chunk_size:
            chunks_g, chunk_deciles_g = _chunk_texts(texts_g, deciles_g, chunk_size, chunk_overlap)
            for t, d in zip(chunks_g, chunk_deciles_g):
                all_texts.append(t)
                all_labels.append((grp, d))
            for d in deciles_g:
                all_source_labels.append((grp, d))
        else:
            for t, d in zip(texts_g, deciles_g):
                all_texts.append(t)
                all_labels.append((grp, d))

    if not all_texts:
        print("  ⚠ No texts found — returning empty stats")
        return {}

    # ── 4. Transform texts using corpus vectorizer (or fit a new one) ─────────
    if corpus_vectorizer is not None:
        print(
            f"  Transforming {len(all_texts):,} texts with corpus vectorizer "
            f"(vocab={len(corpus_vectorizer.vocabulary_):,})…"
        )
        tfidf_matrix = corpus_vectorizer.transform(all_texts)
        idf_values = corpus_vectorizer.idf_
        vocab_size = len(corpus_vectorizer.vocabulary_)
    else:
        print(
            f"  No corpus vectorizer supplied — fitting on {len(all_texts):,} "
            f"query target texts (vocab cap {max_features:,})…"
        )
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            strip_accents="unicode",
            sublinear_tf=True,
            min_df=2,
        )
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        idf_values = vectorizer.idf_
        vocab_size = len(vectorizer.vocabulary_)

    print(f"  Vocabulary in use: {vocab_size:,} terms")

    labels_arr = np.array(all_labels, dtype=object)   # shape (N, 2)
    texts_arr = np.array(all_texts, dtype=object)

    # ── 5. Compute stats per (group, decile) ──────────────────────────────────
    results: dict[str, dict[int, dict]] = {g: {} for g in groups}

    for grp in groups:
        source_counts: dict[int, int] = {}
        if chunk_size:
            for sg, sd in all_source_labels:
                if sg == grp:
                    source_counts[sd] = source_counts.get(sd, 0) + 1

        for d in range(_NUM_DECILES):
            mask = (labels_arr[:, 0] == grp) & (labels_arr[:, 1].astype(int) == d)
            n = int(mask.sum())
            if n == 0:
                continue

            sub = tfidf_matrix[mask]
            texts_sub = texts_arr[mask]

            unique_per_doc = np.diff(sub.indptr).astype(float)
            nnz = sub.nnz
            mean_score = float(sub.sum() / nnz) if nnz > 0 else 0.0
            used_idx = np.unique(sub.nonzero()[1])
            mean_idf = float(idf_values[used_idx].mean()) if len(used_idx) > 0 else 0.0
            vocab_cover = float(len(used_idx) / vocab_size)
            lengths = np.array([len(t) for t in texts_sub])
            mean_chunk_len = float(np.mean(lengths))
            std_chunk_len = float(np.std(lengths))
            coo = sub.tocoo()
            if coo.nnz > 0:
                tf_vals = coo.data / idf_values[coo.col]
                mean_tf = float(np.mean(tf_vals))
            else:
                mean_tf = 0.0

            results[grp][d] = {
                "n_docs_sampled": n,
                "n_source_docs": source_counts.get(d, n),
                "mean_unique_terms": float(np.mean(unique_per_doc)),
                "std_unique_terms": float(np.std(unique_per_doc)),
                "mean_tfidf_score": mean_score,
                "mean_idf_of_used_terms": mean_idf,
                "vocab_coverage": vocab_cover,
                "mean_chunk_length": mean_chunk_len,
                "std_chunk_length": std_chunk_len,
                "mean_tf_score": mean_tf,
            }

    return results


def load_or_compute_tfidf_stats(
    metadata_path: Path | str,
    corpus_path: Path | str,
    boundaries: np.ndarray,
    *,
    sample_per_decile: int = 300,
    max_features: int = 20_000,
    chunk_size: int | None = None,
    chunk_overlap: int = 100,
    force_recompute: bool = False,
) -> dict[str, dict[str, Any]]:
    """Return per-decile TF-IDF statistics, using the cache when possible.

    Parameters
    ----------
    metadata_path : Path | str
        Path to the ``metadata.json`` file used to store/retrieve the cache.
    corpus_path : Path | str
        Path to the Parquet corpus file (must have ``popularity_avg`` and
        ``text`` columns).
    boundaries : np.ndarray
        Decile boundary array (length ``NUM_DECILES + 1``) as returned by
        ``rag.decile_utils.boundaries_for``.
    sample_per_decile : int
        Maximum number of documents to sample per decile.  When
        ``chunk_size`` is set this is the number of *documents* sampled;
        the actual TF-IDF corpus will contain more units (one per chunk).
    max_features : int
        Vocabulary cap for ``TfidfVectorizer``.
    chunk_size : int | None
        When set, each sampled document is split into chunks of this many
        characters using LangChain's ``RecursiveCharacterTextSplitter``
        before TF-IDF is computed.  Stats then reflect chunk-level
        vocabulary rather than full-document vocabulary.  ``None`` (default)
        keeps the original document-level behaviour.
    chunk_overlap : int
        Overlap in characters between consecutive chunks (only used when
        ``chunk_size`` is not ``None``).
    force_recompute : bool
        If ``True``, skip the cache and recompute from scratch.

    Returns
    -------
    dict mapping str(decile_index_0_based) → stats dict with keys:
        n_docs_sampled, mean_unique_terms, std_unique_terms,
        mean_tfidf_score, mean_idf_of_used_terms, vocab_coverage
    """
    metadata_path = Path(metadata_path)
    key = _cache_key(chunk_size, chunk_overlap, sample_per_decile)

    # ---- Try cache first ------------------------------------------------
    if not force_recompute and metadata_path.exists():
        with open(metadata_path) as f:
            meta = json.load(f)
        if key in meta:
            level = f"chunk (size={chunk_size}, overlap={chunk_overlap})" if chunk_size else "document"
            print(f"✓ Loaded TF-IDF stats from metadata.json [{level} level, cached]")
            return meta[key]

    level = f"chunk (size={chunk_size}, overlap={chunk_overlap})" if chunk_size else "document"
    print(
        f"No cache found — computing TF-IDF stats [{level} level] "
        f"(sampling {sample_per_decile} docs/decile from corpus)…"
    )

    # ---- Full computation -----------------------------------------------
    texts, deciles = _reservoir_sample_corpus(
        corpus_path,
        boundaries,
        sample_per_decile,
    )

    if chunk_size:
        print(f"  Splitting {len(texts):,} documents into chunks "
              f"(size={chunk_size}, overlap={chunk_overlap})…")
        source_deciles_for_stats = list(deciles)  # save before splitting
        texts, deciles = _chunk_texts(texts, deciles, chunk_size, chunk_overlap)
        print(f"  → {len(texts):,} chunks across all deciles")
    else:
        source_deciles_for_stats = None

    stats = _compute_stats(texts, deciles, max_features, source_deciles_for_stats)

    # ---- Write cache back to metadata.json ------------------------------
    with open(metadata_path) as f:
        meta = json.load(f)
    meta[key] = stats
    with open(metadata_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"✓ Saved TF-IDF stats [{level} level] → metadata.json  (future runs will load instantly)")

    return stats


def print_tfidf_stats(stats: dict[str, dict]) -> None:
    """Print a formatted summary table of per-decile TF-IDF statistics."""
    header = (
        f"{'Decile':>7} {'src_docs':>9} {'n_chunks':>9} {'mean_unique_terms':>18} "
        f"{'mean_tfidf':>12} {'mean_idf':>10} {'vocab_cov':>10}"
    )
    print(f"\n{header}")
    for d in range(_NUM_DECILES):
        s = stats.get(str(d), {})
        if not s:
            continue
        n_chunks = s['n_docs_sampled']
        n_src = s.get('n_source_docs', n_chunks)
        print(
            f"  {d + 1:>5}   {n_src:>9}  {n_chunks:>9}   "
            f"{s['mean_unique_terms']:>16.1f}   "
            f"{s['mean_tfidf_score']:>10.4f}   "
            f"{s['mean_idf_of_used_terms']:>8.4f}   "
            f"{s['vocab_coverage']:>8.4f}"
        )


def plot_retrieval_difficulty(
    corpus_docs: np.ndarray,
    corpus_chunks: np.ndarray,
    questions_per_decile: np.ndarray,
    *,
    tfidf_stats: dict | None = None,
    avg_chunk_length: np.ndarray | None = None,
    decile_mode: str = "",
    out_path: "Path | str | None" = None,
) -> "plt.Figure":  # type: ignore[name-defined]  # noqa: F821
    """Plot corpus size, gold-question count, random-baseline hit probability,
    and average chunk length per decile.

    Visualises why later (popular) deciles are intrinsically easier to retrieve:
    there are fewer documents in the index yet more QA questions target them,
    so even a random chunk selector has a much higher chance of returning the
    gold document.

    Parameters
    ----------
    corpus_docs : np.ndarray
        Absolute document count per decile (length 10, index 0 = decile 1).
    corpus_chunks : np.ndarray
        Absolute chunk count per decile (length 10).
    questions_per_decile : np.ndarray
        Number of gold QA questions per decile (length 10).
    avg_chunk_length : np.ndarray | None
        Average character count per chunk per decile (length 10).  When
        provided, a 4th panel is added showing how chunk size varies across
        deciles.  Pass ``None`` to omit the panel (3-panel layout).
    decile_mode : str
        Label for the figure title (e.g. ``"chunk_weighted"``).
    out_path : Path | str | None
        If provided, the figure is saved to this path at 150 dpi.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    d_idx = np.arange(1, _NUM_DECILES + 1)
    mode_label = f" — {decile_mode} mode" if decile_mode else ""

    # Auto-derive avg_chunk_length from tfidf_stats if not supplied explicitly
    if avg_chunk_length is None and tfidf_stats is not None:
        avg_chunk_length = np.array([
            tfidf_stats.get(str(d), {}).get("mean_chunk_length", np.nan)
            for d in range(_NUM_DECILES)
        ])

    n_panels = 4 if avg_chunk_length is not None else 3

    # Random baseline: P(random chunk = gold doc) = 1 / corpus_docs[d]
    with np.errstate(divide="ignore", invalid="ignore"):
        random_p_hit = np.where(corpus_docs > 0, 1.0 / corpus_docs, np.nan)

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))

    # Panel A — Corpus document & chunk count per decile
    ax = axes[0]
    width = 0.4
    ax.bar(d_idx - width / 2, corpus_docs, width, color="#2ecc71",
           alpha=0.85, edgecolor="white", linewidth=1, label="Documents")
    ax.bar(d_idx + width / 2, corpus_chunks, width, color="#1abc9c",
           alpha=0.85, edgecolor="white", linewidth=1, label="Chunks")
    ax.set_xlabel("Popularity Decile (1=Rare → 10=Famous)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Count", fontsize=11, fontweight="bold")
    ax.set_title(f"A. Corpus Size per Decile{mode_label}\n(absolute docs & chunks)", fontsize=12, fontweight="bold")
    ax.set_xticks(d_idx)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))

    # Panel B — Gold QA questions per decile
    ax = axes[1]
    ax.bar(d_idx, questions_per_decile, color="#3498db", alpha=0.85, edgecolor="white", linewidth=1)
    for xi, n in zip(d_idx, questions_per_decile):
        if n > 0:
            ax.text(xi, n + max(questions_per_decile) * 0.01, f"{int(n):,}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xlabel("Popularity Decile (1=Rare → 10=Famous)", fontsize=11, fontweight="bold")
    ax.set_ylabel("# Gold QA Questions", fontsize=11, fontweight="bold")
    ax.set_title("B. Gold Standard Questions per Decile\n(questions in the QA dataset)", fontsize=12, fontweight="bold")
    ax.set_xticks(d_idx)
    ax.grid(axis="y", alpha=0.3)

    # Panel C — Random-baseline hit probability
    ax = axes[2]
    pct = random_p_hit * 100
    ax.bar(d_idx, pct, color="#e74c3c", alpha=0.85, edgecolor="white", linewidth=1)
    valid_pct = pct[~np.isnan(pct)]
    for xi, p in zip(d_idx, pct):
        if not np.isnan(p):
            ax.text(xi, p + (max(valid_pct) * 0.01 if len(valid_pct) else 0), f"{p:.3f}%",
                    ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax.set_xlabel("Popularity Decile (1=Rare → 10=Famous)", fontsize=11, fontweight="bold")
    ax.set_ylabel("P(random chunk = gold doc) %", fontsize=11, fontweight="bold")
    ax.set_title(
        "C. Random-Baseline Hit Probability\n"
        "(= 1 / corpus_docs[d] — higher = trivially easier)",
        fontsize=12, fontweight="bold",
    )
    ax.set_xticks(d_idx)
    ax.grid(axis="y", alpha=0.3)

    # Panel D — Average chunk character length per decile (optional)
    if avg_chunk_length is not None:
        ax = axes[3]
        ax.bar(d_idx, avg_chunk_length, color="#9b59b6", alpha=0.85, edgecolor="white", linewidth=1)
        for xi, v in zip(d_idx, avg_chunk_length):
            if not np.isnan(v):
                ax.text(xi, v + max(avg_chunk_length[~np.isnan(avg_chunk_length)]) * 0.01,
                        f"{int(v):,}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.set_xlabel("Popularity Decile (1=Rare → 10=Famous)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Avg Characters per Chunk", fontsize=11, fontweight="bold")
        ax.set_title("D. Avg Chunk Length per Decile\n(characters — longer = more content per chunk)",
                     fontsize=12, fontweight="bold")
        ax.set_xticks(d_idx)
        ax.grid(axis="y", alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))

    plt.suptitle(
        f"Retrieval Difficulty per Decile{mode_label}\n"
        "(popular deciles have fewer documents → higher baseline hit rate)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"✓ Saved → {out_path}")

    return fig


def plot_tfidf_stats(
    stats: dict[str, dict],
    *,
    decile_mode: str = "",
    sample_per_decile: int = 300,
    max_features: int = 20_000,
    chunk_size: int | None = None,
    chunk_overlap: int = 100,
    out_path: Path | str | None = None,
) -> "plt.Figure":  # type: ignore[name-defined]  # noqa: F821
    """Create and return a 3-panel TF-IDF distribution figure.

    Parameters
    ----------
    stats : dict
        Output of :func:`load_or_compute_tfidf_stats`.
    decile_mode : str
        Label for the figure title (e.g. ``"chunk_weighted"``).
    sample_per_decile : int
        Used only in the subtitle for context.
    max_features : int
        Used only in the subtitle for context.
    chunk_size : int | None
        When set, axis labels refer to chunks rather than documents.
        Should match the value passed to :func:`load_or_compute_tfidf_stats`.
    chunk_overlap : int
        Used only in the subtitle when ``chunk_size`` is set.
    out_path : Path | str | None
        If provided, the figure is saved to this path at 150 dpi.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    unit = f"chunk ({chunk_size} chars)" if chunk_size else "document"
    unit_short = "Chunk" if chunk_size else "Doc"

    d_idx = np.arange(1, _NUM_DECILES + 1)

    mean_unique   = [stats.get(str(d), {}).get("mean_unique_terms",     np.nan) for d in range(_NUM_DECILES)]
    std_unique    = [stats.get(str(d), {}).get("std_unique_terms",       np.nan) for d in range(_NUM_DECILES)]
    mean_score    = [stats.get(str(d), {}).get("mean_tfidf_score",       np.nan) for d in range(_NUM_DECILES)]
    mean_idf      = [stats.get(str(d), {}).get("mean_idf_of_used_terms", np.nan) for d in range(_NUM_DECILES)]
    mean_tf       = [stats.get(str(d), {}).get("mean_tf_score",          np.nan) for d in range(_NUM_DECILES)]
    mean_chunk_len = [stats.get(str(d), {}).get("mean_chunk_length",     np.nan) for d in range(_NUM_DECILES)]
    std_chunk_len  = [stats.get(str(d), {}).get("std_chunk_length",      np.nan) for d in range(_NUM_DECILES)]
    n_chunks      = [stats.get(str(d), {}).get("n_docs_sampled",         np.nan) for d in range(_NUM_DECILES)]
    n_src         = [stats.get(str(d), {}).get("n_source_docs",          np.nan) for d in range(_NUM_DECILES)]

    # 2-row layout: 3 panels on top row, 2 new panels centred on bottom row
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 6, figure=fig, hspace=0.45, wspace=0.35)
    ax_A = fig.add_subplot(gs[0, 0:2])
    ax_B = fig.add_subplot(gs[0, 2:4])
    ax_C = fig.add_subplot(gs[0, 4:6])
    ax_D = fig.add_subplot(gs[1, 1:3])
    ax_E = fig.add_subplot(gs[1, 3:5])

    # Panel A — Vocabulary breadth
    ax_A.bar(d_idx, mean_unique, color="#3498db", alpha=0.85, edgecolor="white", linewidth=1)
    ax_A.errorbar(
        d_idx, mean_unique, yerr=std_unique,
        fmt="none", color="black", capsize=4, linewidth=1.5, elinewidth=1,
    )
    if chunk_size:
        for xi, nc, ns in zip(d_idx, n_chunks, n_src):
            if not np.isnan(nc):
                ax_A.text(xi, 0, f"{int(nc):,}\nchunks\n({int(ns):,} docs)",
                          ha="center", va="bottom", fontsize=6.5, color="#555555")
    ax_A.set_xlabel("Popularity Decile (1=Rare → 10=Famous)", fontsize=11, fontweight="bold")
    ax_A.set_ylabel(f"Mean Unique Terms / {unit_short}", fontsize=11, fontweight="bold")
    ax_A.set_title(f"A. Vocabulary Breadth\n(unique TF-IDF terms per {unit})", fontsize=12, fontweight="bold")
    ax_A.set_xticks(d_idx)
    ax_A.grid(axis="y", alpha=0.3)

    # Panel B — Keyword intensity
    ax_B.bar(d_idx, mean_score, color="#e67e22", alpha=0.85, edgecolor="white", linewidth=1)
    ax_B.set_xlabel("Popularity Decile (1=Rare → 10=Famous)", fontsize=11, fontweight="bold")
    ax_B.set_ylabel("Mean TF-IDF Score (non-zero entries)", fontsize=11, fontweight="bold")
    ax_B.set_title(f"B. Keyword Intensity\n(sublinear TF-IDF of matched terms per {unit})", fontsize=12, fontweight="bold")
    ax_B.set_xticks(d_idx)
    ax_B.grid(axis="y", alpha=0.3)

    # Panel C — Term specificity
    ax_C.bar(d_idx, mean_idf, color="#27ae60", alpha=0.85, edgecolor="white", linewidth=1)
    ax_C.set_xlabel("Popularity Decile (1=Rare → 10=Famous)", fontsize=11, fontweight="bold")
    ax_C.set_ylabel("Mean IDF of Used Terms", fontsize=11, fontweight="bold")
    ax_C.set_title("C. Term Specificity\n(higher IDF = rarer/more unique terms)", fontsize=12, fontweight="bold")
    ax_C.set_xticks(d_idx)
    ax_C.grid(axis="y", alpha=0.3)

    # Panel D — Average chunk / document length
    ax_D.bar(d_idx, mean_chunk_len, color="#9b59b6", alpha=0.85, edgecolor="white", linewidth=1)
    ax_D.errorbar(
        d_idx, mean_chunk_len, yerr=std_chunk_len,
        fmt="none", color="black", capsize=4, linewidth=1.5, elinewidth=1,
    )
    ax_D.set_xlabel("Popularity Decile (1=Rare → 10=Famous)", fontsize=11, fontweight="bold")
    ax_D.set_ylabel(f"Mean {unit_short} Length (chars)", fontsize=11, fontweight="bold")
    ax_D.set_title(f"D. Avg {unit_short} Length\n(character count per {unit})", fontsize=12, fontweight="bold")
    ax_D.set_xticks(d_idx)
    ax_D.grid(axis="y", alpha=0.3)

    # Panel E — Mean sublinear TF (TF component recovered from TF-IDF / IDF)
    ax_E.bar(d_idx, mean_tf, color="#e74c3c", alpha=0.85, edgecolor="white", linewidth=1)
    ax_E.set_xlabel("Popularity Decile (1=Rare → 10=Famous)", fontsize=11, fontweight="bold")
    ax_E.set_ylabel("Mean Sublinear TF (1 + log tf)", fontsize=11, fontweight="bold")
    ax_E.set_title("E. Term Frequency\n(mean sublinear TF across matched terms)", fontsize=12, fontweight="bold")
    ax_E.set_xticks(d_idx)
    ax_E.grid(axis="y", alpha=0.3)

    mode_label = f" — {decile_mode} mode" if decile_mode else ""
    chunk_label = (
        f" · chunked {chunk_size} chars / {chunk_overlap} overlap"
        if chunk_size else ""
    )
    fig.suptitle(
        f"TF-IDF Distribution Per Decile{mode_label} [{unit} level]\n"
        f"(sampled ≤{sample_per_decile} docs/decile · vocab cap {max_features:,}{chunk_label})",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"✓ Saved → {out_path}")

    return fig
