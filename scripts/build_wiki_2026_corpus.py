"""Build a Wikipedia 2026 corpus parquet from a MediaWiki XML dump.

Parses a standard Wikipedia XML dump into a parquet file (id, title, text),
then joins one or two views datasets (old / new) to add view-count columns.
Views datasets are loaded via a pluggable `ViewsSource` abstraction that
supports both local parquet files and HuggingFace datasets.
"""

from __future__ import annotations

import gc
import logging
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

MEDIAWIKI_NS = "http://www.mediawiki.org/xml/export-0.11/"
XML_CHUNK_SIZE = 10_000   # pages per parquet write batch
XML_TAG_PAGE = f"{{{MEDIAWIKI_NS}}}page"
XML_TAG_TITLE = f"{{{MEDIAWIKI_NS}}}title"
XML_TAG_ID = f"{{{MEDIAWIKI_NS}}}id"
XML_TAG_REVISION = f"{{{MEDIAWIKI_NS}}}revision"
XML_TAG_TEXT = f"{{{MEDIAWIKI_NS}}}text"
XML_TAG_NS = f"{{{MEDIAWIKI_NS}}}ns"


# ── Views source abstraction ───────────────────────────────────────────────────

class ViewsSource(ABC):
    """Abstract source for pageview data.

    Implementations must yield DataFrames with at least an ``id_column``
    column and a numeric ``views`` column.
    """

    def __init__(self, id_column: str = "id", views_column: str = "views") -> None:
        self.id_column = id_column
        self.views_column = views_column

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Load the full views dataset into a DataFrame.

        Returns:
            DataFrame with at minimum ``id_column`` (int64) and
            ``views_column`` (int64) columns.
        """
        raise NotImplementedError

    def load_id_set(self) -> dict[int, int]:
        """Return a mapping of page_id -> view_count.

        Aggregates by id in case the source has duplicate entries.
        """
        df = self.load()
        df = df[[self.id_column, self.views_column]].copy()
        df[self.id_column] = pd.to_numeric(df[self.id_column], errors="coerce")
        df[self.views_column] = pd.to_numeric(df[self.views_column], errors="coerce").fillna(0)
        df = df.dropna(subset=[self.id_column])
        df[self.id_column] = df[self.id_column].astype("int64")
        df[self.views_column] = df[self.views_column].astype("int64")
        df = df.groupby(self.id_column, as_index=False)[self.views_column].sum()
        return dict(zip(df[self.id_column], df[self.views_column]))


class ParquetViewsSource(ViewsSource):
    """Load pageviews from a local parquet file."""

    def __init__(
        self,
        path: str | Path,
        *,
        id_column: str = "id",
        views_column: str = "views",
    ) -> None:
        super().__init__(id_column=id_column, views_column=views_column)
        self.path = Path(path)

    def load(self) -> pd.DataFrame:
        """Load parquet file into a DataFrame.

        Returns:
            DataFrame with id and views columns.

        Raises:
            FileNotFoundError: If the parquet file does not exist.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Parquet views file not found: {self.path}")
        logger.info(f"Loading parquet views from {self.path}")
        return pd.read_parquet(self.path, columns=[self.id_column, self.views_column])


class HuggingFaceViewsSource(ViewsSource):
    """Load pageviews from a HuggingFace dataset."""

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        *,
        id_column: str = "id",
        views_column: str = "views",
        hf_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(id_column=id_column, views_column=views_column)
        self.dataset_name = dataset_name
        self.split = split
        self.hf_kwargs = hf_kwargs or {}

    def load(self) -> pd.DataFrame:
        """Load HuggingFace dataset into a DataFrame.

        Returns:
            DataFrame with id and views columns.
        """
        from datasets import load_dataset  # type: ignore  # lazy import

        logger.info(f"Loading HuggingFace dataset '{self.dataset_name}' split='{self.split}'")
        ds = load_dataset(self.dataset_name, split=self.split, **self.hf_kwargs)
        df = ds.to_pandas()[[self.id_column, self.views_column]]
        return df


# ── XML parsing helpers ────────────────────────────────────────────────────────

def _iter_xml_pages(xml_path: Path, file_size: int) -> Iterator[dict[str, Any]]:
    """Iterate over article pages in a MediaWiki XML dump.

    Skips non-article namespaces (ns != 0).

    Args:
        xml_path: Path to the Wikipedia XML dump.
        file_size: Total file size in bytes (used for tqdm progress).

    Yields:
        Dicts with keys ``id`` (int), ``title`` (str), ``text`` (str).
    """
    with (
        open(xml_path, "rb") as fh,
        tqdm(total=file_size, unit="B", unit_scale=True, desc="Parsing XML") as pbar,
    ):
        last_pos = 0
        context = ET.iterparse(fh, events=("end",))
        for _event, elem in context:
            if elem.tag != XML_TAG_PAGE:
                continue

            # Update progress bar based on file position
            current_pos = fh.tell()
            pbar.update(current_pos - last_pos)
            last_pos = current_pos

            ns_elem = elem.find(XML_TAG_NS)
            if ns_elem is None or ns_elem.text != "0":
                elem.clear()
                continue

            title_elem = elem.find(XML_TAG_TITLE)
            id_elem = elem.find(XML_TAG_ID)
            revision = elem.find(XML_TAG_REVISION)
            text_elem = revision.find(XML_TAG_TEXT) if revision is not None else None

            title = title_elem.text if title_elem is not None else ""
            page_id_raw = id_elem.text if id_elem is not None else None
            text = text_elem.text if text_elem is not None else ""

            if page_id_raw is None:
                elem.clear()
                continue

            yield {
                "id": int(page_id_raw),
                "title": title or "",
                "text": text or "",
            }
            elem.clear()


# ── Main class ─────────────────────────────────────────────────────────────────

class Wiki2026CorpusBuilder:
    """Build a Wikipedia corpus parquet from a MediaWiki XML dump.

    Parses the XML dump into (id, title, text) rows and optionally joins
    two views sources to add ``views_1`` and ``views_2`` columns for
    comparison across time periods (e.g. 2019 vs 2026).

    Args:
        output_path: Destination parquet file path.
        xml_path: Path to the MediaWiki XML dump.
    """

    def __init__(self, output_path: str | Path, xml_path: str | Path) -> None:
        self.output_path = Path(output_path)
        self.xml_path = Path(xml_path)
        self._check_paths()

    def _check_paths(self) -> None:
        if not self.xml_path.exists():
            raise FileNotFoundError(f"XML dump not found: {self.xml_path}")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def convert_xml_to_parquet(self) -> None:
        """Parse the XML dump and write a raw (id, title, text) parquet file.

        Uses streaming XML parsing and chunked parquet writes to stay
        RAM-efficient for multi-GB dumps.
        """
        if self.output_path.exists():
            raise FileExistsError(f"Output file already exists: {self.output_path}. Delete it first.")

        file_size = self.xml_path.stat().st_size
        writer: pq.ParquetWriter | None = None
        batch: list[dict[str, Any]] = []
        total_pages = 0

        schema = pa.schema([
            pa.field("id", pa.int64()),
            pa.field("title", pa.string()),
            pa.field("text", pa.string()),
        ])

        try:
            for page in _iter_xml_pages(self.xml_path, file_size):
                batch.append(page)
                if len(batch) >= XML_CHUNK_SIZE:
                    table = pa.Table.from_pylist(batch, schema=schema)
                    if writer is None:
                        writer = pq.ParquetWriter(self.output_path, schema)
                    writer.write_table(table)
                    total_pages += len(batch)
                    batch.clear()
                    gc.collect()

            if batch:
                table = pa.Table.from_pylist(batch, schema=schema)
                if writer is None:
                    writer = pq.ParquetWriter(self.output_path, schema)
                writer.write_table(table)
                total_pages += len(batch)
        finally:
            if writer:
                writer.close()

        logger.info(f"Written {total_pages:,} articles to {self.output_path}")

    def build_corpus(
        self,
        views_source_1: ViewsSource | None = None,
        views_source_2: ViewsSource | None = None,
        *,
        views_1_col: str = "views_1",
        views_2_col: str = "views_2",
        row_group_size: int = 50_000,
    ) -> None:
        """Join view counts onto the raw corpus parquet in a RAM-efficient way.

        Loads both views sources into lightweight id→views dicts, then streams
        through the corpus parquet one row-group at a time, enriching and
        writing to a temp file before atomically replacing the original.

        Args:
            views_source_1: First views source (e.g. old HF pageviews).
            views_source_2: Second views source (e.g. 2026 parquet pageviews).
            views_1_col: Output column name for views from source 1.
            views_2_col: Output column name for views from source 2.
            row_group_size: Rows per parquet row-group when writing output.

        Raises:
            FileNotFoundError: If the raw parquet has not been created yet.
        """
        if not self.output_path.exists():
            raise FileNotFoundError(
                f"Raw parquet not found at {self.output_path}. "
                "Run convert_xml_to_parquet() first."
            )

        # Load id→views dicts up front (ints only — tiny RAM footprint)
        id_map_1: dict[int, int] = {}
        if views_source_1 is not None:
            logger.info(f"Loading views source 1 into id map...")
            id_map_1 = views_source_1.load_id_set()
            logger.info(f"  {len(id_map_1):,} entries loaded")

        id_map_2: dict[int, int] = {}
        if views_source_2 is not None:
            logger.info(f"Loading views source 2 into id map...")
            id_map_2 = views_source_2.load_id_set()
            logger.info(f"  {len(id_map_2):,} entries loaded")

        tmp_path = self.output_path.with_suffix(".tmp.parquet")
        pf = pq.ParquetFile(self.output_path)
        total_rows = pf.metadata.num_rows
        writer: pq.ParquetWriter | None = None

        try:
            with tqdm(total=total_rows, unit=" rows", desc="Enriching corpus") as pbar:
                for batch in pf.iter_batches(batch_size=row_group_size):
                    df = batch.to_pandas()

                    if id_map_1:
                        df[views_1_col] = df["id"].map(id_map_1).fillna(0).astype("int64")
                    if id_map_2:
                        df[views_2_col] = df["id"].map(id_map_2).fillna(0).astype("int64")

                    table = pa.Table.from_pandas(df, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(tmp_path, table.schema)
                    writer.write_table(table)

                    pbar.update(len(df))
                    del df, table
                    gc.collect()
        finally:
            if writer:
                writer.close()

        del id_map_1, id_map_2
        gc.collect()

        # Atomically replace original with enriched file
        tmp_path.replace(self.output_path)
        logger.info(f"Done. Enriched corpus written to {self.output_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

HF_REPO = "Cyro1/popularity-enriched-qa-datasets"


def main() -> None:
    """Build the 2026 Wikipedia corpus parquet.

    XML source : data/popularity/enwiki-2026-04-01-p10p1141529.xml
    Old views  : HuggingFace repo (all subsets merged, id=wikipedia_id)
    New views  : data/popularity/en_wikipedia-202601.parquet
    Output     : data/wiki_2026/wiki_2026_corpus.parquet
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ROOT = Path(__file__).parent.parent
    xml_path = ROOT / "data" / "popularity" / "enwiki-2026-04-01-p10p1141529.xml"
    output_path = ROOT / "data" / "wiki_2026" / "wiki_2026_corpus.parquet"
    views_2026_path = ROOT / "data" / "popularity" / "en_wikipedia-202601.parquet"

    # Old views: aggregate all subsets from the HF repo into a single id→views map.
    # Each subset has wikipedia_id (page id) and popularity_avg (avg monthly views).
    HF_SUBSETS = ["fever", "hotpot_qa", "natural_questions", "pop_qa", "trivia_qa"]

    class MergedHFViewsSource(ViewsSource):
        """Load and merge multiple HF subsets into one id→views mapping."""

        def load(self) -> pd.DataFrame:
            from datasets import load_dataset  # type: ignore

            frames: list[pd.DataFrame] = []
            for subset in tqdm(HF_SUBSETS, desc="Loading HF subsets"):
                logger.info(f"Loading HF subset '{subset}'")
                for split in ("train", "test"):
                    try:
                        ds = load_dataset(HF_REPO, name=subset, split=split)
                        df = ds.to_pandas()[["wikipedia_id", "popularity_avg"]]
                        frames.append(df)
                    except Exception as e:
                        logger.warning(f"Skipping {subset}/{split}: {e}")
            merged = pd.concat(frames, ignore_index=True)
            merged = merged.rename(columns={"wikipedia_id": self.id_column, "popularity_avg": self.views_column})
            return merged

    builder = Wiki2026CorpusBuilder(output_path=output_path, xml_path=xml_path)

    if not output_path.exists():
        logger.info("Step 1/2: Converting XML dump to parquet...")
        builder.convert_xml_to_parquet()
    else:
        logger.info("Step 1/2: Raw parquet already exists, skipping XML conversion.")

    logger.info("Step 2/2: Joining pageview counts...")
    builder.build_corpus(
        views_source_1=MergedHFViewsSource(id_column="id", views_column="views"),
        views_source_2=ParquetViewsSource(views_2026_path),
        views_1_col="views_old_hf",
        views_2_col="views_2026",
    )


if __name__ == "__main__":
    main()
