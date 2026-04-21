"""HuggingFace-backed question input loading from the Cyro1 QA dataset."""

from __future__ import annotations

import gc
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:
    import pandas as pd

from src.corpus_handler.base import CorpusHandler
from src.question_input.base import QuestionInput, QuestionItem

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

HF_REPO = "Cyro1/popularity-enriched-qa-datasets"
AVAILABLE_DATASETS = Literal["natural_questions", "hotpot_qa", "trivia_qa", "pop_qa", "trex", "fever"]

_DEFAULT_SPLIT = "train+test"
_BATCH_SIZE = 50_000


# ── Implementation ────────────────────────────────────────────────────────────

class HuggingFaceCyroInput(QuestionInput):
    """Loads questions from the ``Cyro1/popularity-enriched-qa-datasets`` repo.

    Call :meth:`load` once to fetch from HuggingFace, optionally assign
    popularity deciles via a :class:`~src.corpus_handler.base.CorpusHandler`,
    filter, balance, and write a local parquet file.  Subsequent calls to
    :meth:`get_items` stream from that file — no network access or full-table
    load required.

    Args:
        dataset_names: Sub-dataset names to load.
        parquet_path: Path where the filtered parquet is written by ``load``
            and read back by ``get_items``. Required.
        corpus_handler: A :class:`~src.corpus_handler.base.CorpusHandler`
            instance used to obtain decile boundaries and a
            ``wikipedia_id → popularity`` lookup.  Required for decile
            assignment and decile-based filtering/balancing.
        split: HuggingFace dataset split expression. Defaults to
            ``"train+test"``.
        cache_dir: HuggingFace datasets cache directory.
        deciles: If given, only keep rows in these deciles (0–9). Requires
            ``corpus_handler``.
        datasets_filter: If given, only fetch sub-datasets whose name is in
            this list (skips the rest entirely).
        max_items: Cap the total number of rows written to the parquet.
        balance_deciles: If ``True``, downsample each decile to the count of
            the smallest decile (or ``target_per_decile`` if provided).
            Requires ``corpus_handler``.
        balance_datasets: If ``True``, equalise question counts across source
            datasets within each decile using fill-from-bottom.  Requires
            ``corpus_handler``.
        target_per_decile: Target count per decile when ``balance_deciles``
            is ``True``. Ignored otherwise.
        balance_decile_mode: Which decile column to use when filtering and
            balancing.  ``"unweighted"`` (default) uses
            ``pop_decile_unweighted``; ``"chunk_weighted"`` uses
            ``pop_decile_chunk_weighted``.  Both columns are always written
            to the parquet regardless of this setting.
        shuffle: Shuffle rows before applying ``max_items``.
        random_state: Random seed for sampling. Defaults to ``42``.
        hf_token: HuggingFace API token. Falls back to ``HUGGINGFACE_TOKEN``
            environment variable.
    """

    def __init__(
        self,
        dataset_names: list[str] | None = None,
        *,
        parquet_path: str | Path,
        corpus_handler: CorpusHandler | None = None,
        split: str = _DEFAULT_SPLIT,
        cache_dir: str | None = None,
        deciles: list[int] | None = None,
        datasets_filter: list[str] | None = None,
        max_items: int | None = None,
        balance_deciles: bool = False,
        balance_datasets: bool = False,
        target_per_decile: int | None = None,
        balance_decile_mode: Literal["unweighted", "chunk_weighted"] = "unweighted",
        shuffle: bool = False,
        random_state: int = 42,
        hf_token: str | None = None,
    ) -> None:
        super().__init__()

        if dataset_names is None:
            raise ValueError("At least one dataset name must be provided")

        needs_corpus = balance_deciles or balance_datasets or deciles is not None
        if needs_corpus and corpus_handler is None:
            raise ValueError(
                "corpus_handler is required when using deciles, balance_deciles, or balance_datasets"
            )

        self.dataset_names: list[str] = dataset_names
        self.parquet_path: Path = Path(parquet_path)
        self.corpus_handler = corpus_handler
        self.split = split
        self.cache_dir = cache_dir
        self.deciles = deciles
        self.datasets_filter = datasets_filter
        self.max_items = max_items
        self.balance_deciles = balance_deciles
        self.balance_datasets = balance_datasets
        self.target_per_decile = target_per_decile
        self.balance_decile_mode = balance_decile_mode
        self.shuffle = shuffle
        self.random_state = random_state
        self.hf_token = hf_token or os.environ.get("HUGGINGFACE_TOKEN")

    # ── Load (HuggingFace → filtered parquet) ────────────────────────────────

    def load(self, *, force: bool = False) -> Path:
        """Stream from HuggingFace, assign deciles, filter, balance, and write parquet.

        Pass 1 (HuggingFace fetch) is skipped when the cache parquet already
        exists and ``force`` is ``False``.  **Pass 2 (decile assignment,
        filtering, and balancing) always runs** — it reads whatever parquet
        Pass 1 left behind (or the existing cache) and overwrites it with the
        balanced result.  This ensures that changing ``balance_decile_mode``,
        ``target_per_decile``, or any balancing flag always takes effect
        without needing ``force=True``.

        Pass ``force=True`` to also re-download from HuggingFace (re-runs
        both passes unconditionally).

        Args:
            force: Re-download from HuggingFace even if the cache already
                exists.  Defaults to ``False``.

        Returns:
            The path to the written parquet file.
        """
        # ── Pass 1: stream HF → raw parquet (skipped on cache hit) ───────
        if not force and self.parquet_path.exists():
            raw_rows = pq.read_metadata(self.parquet_path).num_rows
            logger.info(
                "Pass 1 — cache hit, skipping HuggingFace fetch (%s, %d rows)",
                self.parquet_path,
                raw_rows,
            )
        else:
            from datasets import load_dataset, Dataset

            names_to_fetch = (
                [n for n in self.dataset_names if n in self.datasets_filter]
                if self.datasets_filter is not None
                else self.dataset_names
            )

            self.parquet_path.parent.mkdir(parents=True, exist_ok=True)
            writer: pq.ParquetWriter | None = None
            total_written = 0

            try:
                for name in names_to_fetch:
                    logger.info("Pass 1 — fetching %s / %s (split=%s)", HF_REPO, name, self.split)

                    ds = load_dataset(
                        HF_REPO,
                        name,
                        split=self.split,
                        cache_dir=self.cache_dir,
                        token=self.hf_token,
                    )

                    if not isinstance(ds, Dataset):
                        raise ValueError(f"Expected a Dataset for {name!r}, got {type(ds)}")
                    assert isinstance(ds, Dataset)

                    for batch in ds.data.to_batches(max_chunksize=_BATCH_SIZE):
                        batch = batch.append_column(
                            pa.field("dataset", pa.string()),
                            pa.array([name] * len(batch), type=pa.string()),
                        )

                        if len(batch) == 0:
                            continue

                        if writer is None:
                            writer = pq.ParquetWriter(self.parquet_path, batch.schema)
                        writer.write_batch(batch)
                        total_written += len(batch)

                    del ds
                    gc.collect()

            finally:
                if writer is not None:
                    writer.close()

            logger.info("Pass 1 complete — %d raw rows written to %s", total_written, self.parquet_path)

            if total_written == 0:
                return self.parquet_path

        # ── Pass 2: assign deciles + filter + balance (always runs) ───────
        if self.corpus_handler is not None:
            import pandas as pd
            from src.metrics.decile_utils import (
                assign_decile,
                COL_POPULARITY,
                COL_DECILE_UNWEIGHTED,
                COL_DECILE_CHUNK_WEIGHTED,
            )

            logger.info("Pass 2 — starting (decile_mode=%s, balance_deciles=%s, target_per_decile=%s, balance_datasets=%s)",
                        self.balance_decile_mode, self.balance_deciles, self.target_per_decile, self.balance_datasets)

            # Read parquet (either freshly written by Pass 1, or existing cache)
            df: pd.DataFrame = pq.read_table(self.parquet_path).to_pandas()  # type: ignore[assignment]
            logger.info("Pass 2 — loaded %d rows from parquet", len(df))

            # ── Boundaries from corpus handler ─────────────────────────────
            boundaries_uw, boundaries_cw = self.corpus_handler.get_boundaries()

            # ── Build wikipedia_id → popularity lookup (stream corpus) ─────
            logger.info("Pass 2 — building popularity lookup from corpus…")
            qa_ids = set(df["wikipedia_id"].dropna().astype(int).unique())
            id_to_pop: dict[int, float] = {}

            from src.corpus_handler.parquet_corpus_handler import ParquetCorpusHandler
            if isinstance(self.corpus_handler, ParquetCorpusHandler):
                corpus_pf = pq.ParquetFile(self.corpus_handler.corpus_path)
                for batch in corpus_pf.iter_batches(
                    batch_size=_BATCH_SIZE,
                    columns=["wikipedia_id", COL_POPULARITY],
                ):
                    wids = batch.column("wikipedia_id").to_pylist()
                    pops = batch.column(COL_POPULARITY).to_pylist()
                    for wid, pop in zip(wids, pops):
                        if wid is None or pop is None:
                            continue
                        wid = int(wid)
                        if wid in qa_ids and wid not in id_to_pop:
                            id_to_pop[wid] = float(pop)
            else:
                docs = self.corpus_handler.get_documents(list(qa_ids))
                for doc in docs:
                    wid = doc.metadata.get("wikipedia_id")
                    pop = doc.metadata.get(COL_POPULARITY)
                    if wid is not None and pop is not None:
                        id_to_pop[int(wid)] = float(pop)

            gc.collect()
            logger.info("Pass 2 — popularity lookup: %d / %d QA ids matched", len(id_to_pop), len(qa_ids))

            # ── Assign both decile columns ─────────────────────────────────
            pop_series = df["wikipedia_id"].dropna().astype(int).map(id_to_pop)  # type: ignore[arg-type]
            pop_arr = np.asarray(pop_series, dtype=np.float64)
            df[COL_DECILE_UNWEIGHTED]     = assign_decile(pop_arr, boundaries_uw)  # type: ignore[arg-type]
            df[COL_DECILE_CHUNK_WEIGHTED] = assign_decile(pop_arr, boundaries_cw)  # type: ignore[arg-type]

            # ── Select balancing column based on mode ──────────────────────
            # Both decile columns are always written to the parquet.
            # balance_decile_mode controls which one drives balancing/filtering.
            _balance_col = (
                COL_DECILE_CHUNK_WEIGHTED
                if self.balance_decile_mode == "chunk_weighted"
                else COL_DECILE_UNWEIGHTED
            )
            logger.info("Pass 2 — balancing column: %s", _balance_col)

            unmapped = df[COL_DECILE_UNWEIGHTED].isna().sum()
            if unmapped:
                logger.warning("Pass 2 — %d questions have no corpus match — dropping", unmapped)
                df = df.dropna(subset=[COL_DECILE_UNWEIGHTED, COL_DECILE_CHUNK_WEIGHTED])
            df[COL_DECILE_UNWEIGHTED]     = df[COL_DECILE_UNWEIGHTED].astype(int)
            df[COL_DECILE_CHUNK_WEIGHTED] = df[COL_DECILE_CHUNK_WEIGHTED].astype(int)
            df["decile"] = df[_balance_col]

            dist_uw = df[COL_DECILE_UNWEIGHTED].value_counts().sort_index().to_dict()
            dist_cw = df[COL_DECILE_CHUNK_WEIGHTED].value_counts().sort_index().to_dict()
            logger.info("Pass 2 — pre-balance decile distribution (unweighted):     %s", dist_uw)
            logger.info("Pass 2 — pre-balance decile distribution (chunk_weighted): %s", dist_cw)
            logger.info("Pass 2 — %d rows total before balancing", len(df))

            # ── Decile filter ──────────────────────────────────────────────
            if self.deciles is not None:
                before = len(df)
                df = df.loc[df["decile"].isin(self.deciles)].copy()
                logger.info("Pass 2 — decile filter %s: %d → %d rows", self.deciles, before, len(df))

            # ── Shuffle before sampling so balance is random ───────────────
            if self.shuffle:
                df = df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
                logger.info("Pass 2 — shuffled %d rows (random_state=%d)", len(df), self.random_state)

            # ── Dataset balancing ──────────────────────────────────────────
            if self.balance_datasets and "dataset" in df.columns:
                df = _balance_per_dataset(df, self.target_per_decile, self.random_state)

            # ── Decile balancing ───────────────────────────────────────────
            if self.balance_deciles:
                df = _balance_by_decile(df, self.target_per_decile, self.random_state)

            if self.max_items is not None:
                before = len(df)
                df = df.iloc[: self.max_items].copy()
                logger.info("Pass 2 — max_items cap: %d → %d rows", before, len(df))

            dist_final = df["decile"].value_counts().sort_index().to_dict()
            logger.info("Pass 2 — final decile distribution (%s): %s", _balance_col, dist_final)
            logger.info("Pass 2 complete — %d rows total", len(df))

            tmp_path = self.parquet_path.with_suffix(".tmp.parquet")
            pq.write_table(pa.Table.from_pandas(df, preserve_index=False), tmp_path)
            tmp_path.replace(self.parquet_path)

            del df
            gc.collect()

        elif self.shuffle or self.max_items is not None:
            # No corpus handler — still apply shuffle / max_items
            import pandas as pd

            df = pq.read_table(self.parquet_path).to_pandas()  # type: ignore[assignment]
            logger.info("Pass 2 — no corpus handler, %d rows loaded", len(df))

            if self.shuffle:
                df = df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
                logger.info("Pass 2 — shuffled %d rows", len(df))
            if self.max_items is not None:
                before = len(df)
                df = df.iloc[: self.max_items].copy()
                logger.info("Pass 2 — max_items cap: %d → %d rows", before, len(df))

            logger.info("Pass 2 complete — %d rows", len(df))

            tmp_path = self.parquet_path.with_suffix(".tmp.parquet")
            pq.write_table(pa.Table.from_pandas(df, preserve_index=False), tmp_path)
            tmp_path.replace(self.parquet_path)

            del df
            gc.collect()

        else:
            logger.info("Pass 2 — skipped (no corpus_handler, shuffle=False, max_items=None)")

        logger.info("Saved to %s", self.parquet_path)
        return self.parquet_path

    # ── Get items (parquet → QuestionItem list, streamed) ─────────────────────

    def get_items(self) -> list[QuestionItem]:
        """Stream the filtered parquet and return :class:`QuestionItem` list.

        Reads in batches of ``_BATCH_SIZE`` rows.  When a
        :attr:`corpus_handler` is set, each batch's ``wikipedia_id`` values
        are looked up via
        :meth:`~src.corpus_handler.base.CorpusHandler.get_documents` and the
        resulting ``page_content`` is attached to every :class:`QuestionItem`.

        Returns:
            List of :class:`QuestionItem` instances, with
            :attr:`~src.question_input.base.QuestionItem.page_content`
            populated when a corpus handler is available.

        Raises:
            FileNotFoundError: If :attr:`parquet_path` does not exist.
                Call :meth:`load` first.
        """
        if not self.parquet_path.exists():
            raise FileNotFoundError(
                f"Parquet cache not found at {self.parquet_path}. Call load() first."
            )

        items: list[QuestionItem] = []
        pf = pq.ParquetFile(self.parquet_path)

        for batch in pf.iter_batches(batch_size=_BATCH_SIZE):
            content_map: dict[int, str] = {}
            if self.corpus_handler is not None and "wikipedia_id" in batch.schema.names:
                batch_wids = [
                    int(v) for v in batch.column("wikipedia_id").to_pylist()
                    if v is not None
                ]
                if batch_wids:
                    docs = self.corpus_handler.get_documents(batch_wids)
                    content_map = {
                        int(doc.metadata["wikipedia_id"]): doc.page_content
                        for doc in docs
                        if "wikipedia_id" in doc.metadata
                    }
            items.extend(_batch_to_items(batch, content_map))

        return items

    def get_questions(self) -> list[str]:
        """Return only the question text strings."""
        return [item.question_text for item in self.get_items()]


# ── Private helpers ───────────────────────────────────────────────────────────

def _batch_to_items(
    batch: pa.RecordBatch,
    content_map: dict[int, str] | None = None,
) -> list[QuestionItem]:
    """Convert an Arrow record batch to a list of :class:`QuestionItem`.

    Args:
        batch: Arrow record batch from the filtered parquet.
        content_map: Optional mapping of ``wikipedia_id → page_content``
            pre-fetched from the corpus.  When provided, each item's
            :attr:`~src.question_input.base.QuestionItem.page_content`
            is populated from this map.
    """
    schema_names = set(batch.schema.names)

    question_ids   = batch.column("question_id").to_pylist()     if "question_id"     in schema_names else [None] * len(batch)
    question_texts = batch.column("question_text").to_pylist()   if "question_text"   in schema_names else [None] * len(batch)
    answer_texts   = batch.column("answer_texts").to_pylist()    if "answer_texts"    in schema_names else [None] * len(batch)
    wikipedia_ids  = batch.column("wikipedia_id").to_pylist()    if "wikipedia_id"    in schema_names else [None] * len(batch)
    popularities   = batch.column("popularity_avg").to_pylist()  if "popularity_avg"  in schema_names else [None] * len(batch)
    wiki_titles    = batch.column("wikipedia_title").to_pylist() if "wikipedia_title" in schema_names else [None] * len(batch)
    deciles        = batch.column("decile").to_pylist()                        if "decile"                        in schema_names else [None] * len(batch)
    deciles_uw     = batch.column("pop_decile_unweighted").to_pylist()         if "pop_decile_unweighted"         in schema_names else [None] * len(batch)
    deciles_cw     = batch.column("pop_decile_chunk_weighted").to_pylist()     if "pop_decile_chunk_weighted"     in schema_names else [None] * len(batch)
    datasets       = batch.column("dataset").to_pylist()                       if "dataset"                       in schema_names else [None] * len(batch)

    _content_map = content_map or {}

    items = []
    for qid, qt, at, wid, pop, wtitle, dec, dec_uw, dec_cw, ds in zip(
        question_ids, question_texts, answer_texts,
        wikipedia_ids, popularities, wiki_titles, deciles, deciles_uw, deciles_cw, datasets,
    ):
        wid_int = int(wid) if wid is not None else None
        items.append(QuestionItem(
            question_id=str(qid) if qid is not None else "",
            question_text=str(qt) if qt is not None else "",
            answer_texts=at if isinstance(at, list) else [],
            wikipedia_id=str(wid_int) if wid_int is not None else "",
            popularity_avg=float(pop) if pop is not None else None,
            wikipedia_title=str(wtitle) if wtitle is not None else "",
            decile=int(dec) if dec is not None else -1,
            decile_unweighted=int(dec_uw) if dec_uw is not None else -1,
            decile_chunk_weighted=int(dec_cw) if dec_cw is not None else -1,
            dataset=str(ds) if ds is not None else "",
            page_content=_content_map.get(wid_int, "") if wid_int is not None else "",
        ))
    return items


def _balance_by_decile(
    df: "pd.DataFrame",
    target: int | None,
    random_state: int,
) -> "pd.DataFrame":
    """Downsample each decile to *target* (or the smallest decile count)."""
    import pandas as pd
    n = target or int(df["decile"].value_counts().min())
    parts = []
    for _, group in df.groupby("decile"):
        if len(group) > n:
            group = group.sample(n=n, random_state=random_state)
        parts.append(group)
    result = pd.concat(parts, ignore_index=True)
    logger.info("After decile balance (target=%d): %d rows", n, len(result))
    return result


def _balance_per_dataset(
    df: "pd.DataFrame",
    target_per_decile: int | None,
    random_state: int,
) -> "pd.DataFrame":
    """Cap each dataset to at most target_per_decile // n_datasets rows per decile.

    When *target_per_decile* is set, each dataset is allowed at most
    ``target_per_decile // n_datasets`` rows per decile so that no single
    source can crowd out the others.  Surplus rows from smaller datasets are
    *not* redistributed — use ``_balance_by_decile`` afterwards to enforce the
    overall per-decile total.
    """
    import pandas as pd
    dataset_names = df["dataset"].unique().tolist()
    n_datasets = len(dataset_names)

    parts = []
    for _, decile_df in df.groupby("decile"):
        for ds in dataset_names:
            group = decile_df[decile_df["dataset"] == ds]
            if target_per_decile is not None:
                cap = target_per_decile // n_datasets
                if len(group) > cap:
                    group = group.sample(n=cap, random_state=random_state)
            parts.append(group)

    result = pd.concat(parts, ignore_index=True)
    logger.info("After dataset balance (cap=%s per dataset per decile): %d rows",
                f"{target_per_decile}//{n_datasets}" if target_per_decile else "none",
                len(result))
    return result
