"""Clean wiki_2026_corpus.parquet: strip MediaWiki markup to plain text.

The 2026 corpus contains raw wikitext ({{templates}}, [[links]], '''bold''',
<ref> tags, etc.) while the older wiki_full_bil corpus already holds plain
text.  This script normalises the 2026 corpus to the same plain-text style so
both can be used interchangeably in build_similarity_scores.py.

Strategy: stream the parquet in row-group batches, clean only the `text`
column in-place, and write each batch to a new parquet file via an
incremental writer.  Peak RAM is one batch (~50 k rows) at a time.

Output: data/wiki_2026/wiki_2026_corpus_clean.parquet
"""

from __future__ import annotations

import gc
import logging
import re
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ───────────────────────────────────────────────────────────────────────

ROOT        = Path(__file__).parent.parent
INPUT_PATH  = ROOT / "data" / "wiki_2026" / "wiki_2026_corpus.parquet"
OUTPUT_PATH = ROOT / "data" / "wiki_2026" / "wiki_2026_corpus_clean.parquet"

BATCH_SIZE = 50_000  # rows per batch — keeps peak RAM well under 1 GB


# ── Compiled regexes ────────────────────────────────────────────────────────────

# Step 1: HTML comments
_COMMENT      = re.compile(r"<!--.*?-->", re.DOTALL)

# Step 2: <ref> tags
_REF_BLOCK    = re.compile(r"<ref[^>]*>.*?</ref>", re.DOTALL | re.IGNORECASE)
_REF_SELF     = re.compile(r"<ref[^/]*/\s*>", re.IGNORECASE)

# Step 3: remaining HTML/XML tags
_HTML_TAG     = re.compile(r"<[^>]+>", re.DOTALL)

# Step 4: {{ }} templates (innermost-first, iterative)
_TEMPLATE     = re.compile(r"\{\{[^{}]*\}\}")

# Step 5: [[File/Image/Media:...]] blocks
# First resolve inner [[link|display]] that are not File links so the outer
# File block contains no nested [[ and can be matched cleanly.
_INNER_LINK   = re.compile(
    r"\[\[(?!(?:File|Image|Media):)([^\[\]|]*\|)?([^\[\]]+)\]\]",
    re.IGNORECASE,
)
_FILE_BLOCK   = re.compile(r"\[\[(?:File|Image|Media):[^\[\]]*\]\]", re.IGNORECASE)

# Step 6: [[link|display]] → display,  [[link]] → link
_WIKI_LINK    = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]")

# Step 7: external links
_EXT_LINK     = re.compile(r"\[https?://[^\s\]]+\s+([^\]]+)\]")
_EXT_BARE     = re.compile(r"\[https?://[^\]]+\]")

# Step 8: bold / italic markup
_BOLD_ITALIC  = re.compile(r"'{2,3}")

# Step 9: section headings == Heading == → Heading
_HEADING      = re.compile(r"^=+\s*(.*?)\s*=+\s*$", re.MULTILINE)

# Step 10: #REDIRECT lines
_REDIRECT     = re.compile(r"^#REDIRECT.*$", re.IGNORECASE | re.MULTILINE)

# Step 11: horizontal rules
_HRULE        = re.compile(r"^-{4,}\s*$", re.MULTILINE)

# Step 12: table rows (lines starting with | or !)
_TABLE_ROW    = re.compile(r"^\s*[|!].*$", re.MULTILINE)

# Step 13: list / definition markers at start of line (; : * #)
_LIST_MARKER  = re.compile(r"^[;:*#]+\s*", re.MULTILINE)

# Step 14: collapse multiple blank lines
_BLANK_LINES  = re.compile(r"\n{3,}")


# ── Cleaning logic ──────────────────────────────────────────────────────────────

def _iter_sub(pattern: re.Pattern, repl: str, text: str) -> str:
    """Apply a substitution repeatedly until the text stops changing."""
    prev = None
    while prev != text:
        prev = text
        text = pattern.sub(repl, text)
    return text


def _resolve_inner_links(text: str) -> str:
    """Replace [[link|display]] / [[link]] inside File caption text."""
    return _iter_sub(_INNER_LINK, lambda m: m.group(2) if m.group(2) else "", text)


def clean_text(raw: str) -> str:
    """Convert raw MediaWiki markup to clean plain text."""
    t = raw

    t = _COMMENT.sub("", t)
    t = _REF_BLOCK.sub("", t)
    t = _REF_SELF.sub("", t)
    t = _HTML_TAG.sub("", t)

    # Templates (iterative — handles nesting)
    t = _iter_sub(_TEMPLATE, "", t)

    # File/Image/Media blocks: resolve inner links first so the block regex
    # sees no nested [[ ]], then remove the whole block.
    t = _resolve_inner_links(t)
    t = _iter_sub(_FILE_BLOCK, "", t)

    # Remaining wiki links
    t = _WIKI_LINK.sub(r"\1", t)

    t = _EXT_LINK.sub(r"\1", t)
    t = _EXT_BARE.sub("", t)
    t = _BOLD_ITALIC.sub("", t)
    t = _HEADING.sub(r"\1", t)
    t = _REDIRECT.sub("", t)
    t = _HRULE.sub("", t)
    t = _TABLE_ROW.sub("", t)
    t = _LIST_MARKER.sub("", t)
    t = _BLANK_LINES.sub("\n\n", t)

    return t.strip()


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    pf = pq.ParquetFile(INPUT_PATH)
    total_rows = pf.metadata.num_rows
    logger.info(f"Input:  {INPUT_PATH} ({total_rows:,} rows)")
    logger.info(f"Output: {OUTPUT_PATH}")

    writer: pq.ParquetWriter | None = None
    rows_written = 0

    try:
        for batch_idx, batch in enumerate(pf.iter_batches(batch_size=BATCH_SIZE)):
            df = batch.to_pandas()

            df["text"] = df["text"].map(clean_text)

            table = pa.Table.from_pandas(df, preserve_index=False)

            if writer is None:
                writer = pq.ParquetWriter(OUTPUT_PATH, table.schema, compression="zstd")

            writer.write_table(table)
            rows_written += len(df)
            logger.info(
                f"Batch {batch_idx + 1}: wrote {len(df):,} rows "
                f"({rows_written:,}/{total_rows:,} total)"
            )

            del df, table
            gc.collect()

    finally:
        if writer is not None:
            writer.close()

    logger.info(f"Done. {rows_written:,} rows written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
