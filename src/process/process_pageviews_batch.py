#!/usr/bin/env python3
"""
Batch Wikipedia Popularity Table Generator

Processes multiple pageview files sequentially, creating a popularity table for each.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sympy import python
from tqdm import tqdm
from datasets import load_dataset
import aiohttp
import argparse
import glob

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, CACHE_DIR


def load_and_filter_pageviews(pageviews_file, wiki_source="en.wikipedia", chunksize=1_000_000):
    """
    Load and filter pageview data from a file.

    Args:
        pageviews_file: Path to pageviews file
        wiki_source: Wikipedia source to filter (default: "en.wikipedia")
        chunksize: Number of rows to process at a time

    Returns:
        DataFrame with filtered pageview data
    """
    print(f"\n{'='*80}")
    print(f"Processing: {pageviews_file}")
    print(f"{'='*80}")

    # Get file size for progress bar
    file_size = os.path.getsize(pageviews_file)

    chunks = []

    # Wrap the file with tqdm for accurate progress tracking
    with open(pageviews_file, 'r') as f:
        with tqdm(total=file_size, unit='B', unit_scale=True, unit_divisor=1024, desc="Reading & Filtering") as pbar:
            # Track last position for progress updates
            last_pos = 0

            for chunk in pd.read_table(
                f,
                sep=" ",
                names=["source", "entity", "id", "device", "popularity", "features"],
                dtype={
                    "source": "category",
                    "entity": str,
                    "id": str,
                    "device": "category",
                    "popularity": str,
                    "features": str
                },
                usecols=["source", "id", "device", "popularity"],
                chunksize=chunksize,
                engine='c',
                low_memory=False,
            ):
                # Filter immediately to reduce memory
                filtered = chunk[chunk["source"] == wiki_source]
                if len(filtered) > 0:
                    chunks.append(filtered)

                # Update progress bar based on actual file position
                current_pos = f.tell()
                pbar.update(current_pos - last_pos)
                last_pos = current_pos

    # Concatenate all filtered chunks
    print("Concatenating filtered chunks...")
    wiki_data = pd.concat(chunks, ignore_index=True)

    # Drop source column as it's no longer needed
    wiki_data = wiki_data.drop(columns=["source"])

    # Convert popularity to numeric, coercing errors to NaN
    wiki_data["popularity"] = pd.to_numeric(wiki_data["popularity"], errors='coerce')

    # Drop rows with non-finite popularity values (NA, inf, -inf)
    before_count = len(wiki_data)
    wiki_data = wiki_data[wiki_data["popularity"].notna() & np.isfinite(wiki_data["popularity"])]
    dropped_count = before_count - len(wiki_data)
    if dropped_count > 0:
        print(f"⚠ Dropped {dropped_count:,} rows with invalid popularity values")

    # Now safely convert to int32
    wiki_data["popularity"] = wiki_data["popularity"].astype('int32')

    print(f"✓ Filtered to {len(wiki_data):,} rows for {wiki_source}")
    print(f"✓ Memory usage: {wiki_data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    return wiki_data


def aggregate_pageviews(wiki_data):
    """
    Aggregate pageviews by page ID.

    Args:
        wiki_data: DataFrame with pageview data

    Returns:
        DataFrame with aggregated pageviews by ID
    """
    print("\nAggregating pageviews by page ID...")
    pageviews_df = wiki_data[["id", "popularity"]].groupby("id", as_index=False).agg({
        "popularity": "sum"
    })
    print(f"✓ Unique pages: {len(pageviews_df):,}")

    return pageviews_df


def merge_with_metadata(pageviews_df, wiki_df):
    """
    Merge pageviews with Wikipedia metadata and add rankings.

    Args:
        pageviews_df: DataFrame with aggregated pageviews
        wiki_df: DataFrame with Wikipedia metadata

    Returns:
        Merged DataFrame with rankings
    """
    print("\nMerging with Wikipedia metadata...")
    merged_df = wiki_df.merge(
        pageviews_df,
        left_on="wikipedia_id",
        right_on="id",
        how="left"
    ).drop(columns=["id"])
    print(f"✓ Merged {len(merged_df):,} rows")

    # Filter only rows with titles and add rankings
    print("Adding popularity rankings...")
    merged_df_with_title = merged_df[merged_df['wikipedia_title'].notna()].copy()
    merged_df_with_title['rank'] = merged_df_with_title['popularity'].rank(
        method='average',
        na_option='bottom',
        ascending=False
    )

    print(f"✓ {len(merged_df_with_title):,} articles with titles")

    return merged_df_with_title


def save_output(merged_df, output_file):
    """
    Save the popularity table to a parquet file.

    Args:
        merged_df: Merged DataFrame to save
        output_file: Path to output file
    """
    print(f"\nSaving to: {output_file}")

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    merged_df.to_parquet(output_file, index=False)
    file_size = os.path.getsize(output_file) / 1e6
    print(f"✓ Saved successfully ({file_size:.1f} MB)")


def extract_date_from_filename(filename):
    """
    Extract date from filename like 'pageviews-202101-user'.

    Args:
        filename: Name of the pageviews file

    Returns:
        Date string (e.g., "202101") or None
    """
    import re
    match = re.search(r'pageviews-(\d{6})-', filename)
    if match:
        return match.group(1)
    return None


def process_file(pageviews_file, wiki_df, output_dir, wiki_source="en.wikipedia"):
    """
    Process a single pageviews file.

    Args:
        pageviews_file: Path to pageviews file
        wiki_df: DataFrame with Wikipedia metadata (shared across files)
        output_dir: Directory to save output files
        wiki_source: Wikipedia source to filter
    """
    try:
        # Load and filter
        wiki_data = load_and_filter_pageviews(pageviews_file, wiki_source)

        # Aggregate
        pageviews_df = aggregate_pageviews(wiki_data)

        # Free memory
        del wiki_data

        # Merge with metadata
        merged_df = merge_with_metadata(pageviews_df, wiki_df)

        # Free memory
        del pageviews_df

        # Determine output filename
        date_str = extract_date_from_filename(Path(pageviews_file).name)
        if date_str:
            output_filename = f"popularity_table_{date_str}.parquet"
        else:
            output_filename = f"popularity_table_{Path(pageviews_file).stem}.parquet"

        output_file = output_dir / output_filename

        # Save
        save_output(merged_df, output_file)

        return True

    except Exception as e:
        print(f"\n❌ Error processing {pageviews_file}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Batch process Wikipedia pageview files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process specific files
  python process_pageviews_batch.py file1 file2 file3

  # Process all files in a directory
  python process_pageviews_batch.py ../data/raw_wikiPop_dumps/pageviews-*

  # Specify custom output directory
  python process_pageviews_batch.py --output-dir ./output pageviews-202101-user pageviews-202102-user
        """
    )

    parser.add_argument(
        'files',
        nargs='+',
        help='Pageview files to process (can use wildcards)'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DATA_DIR / "cleaned_wikiPop_dumps",
        help='Output directory for popularity tables (default: data/cleaned_wikiPop_dumps)'
    )

    parser.add_argument(
        '--wiki-source',
        default='en.wikipedia',
        help='Wikipedia source to filter (default: en.wikipedia)'
    )

    parser.add_argument(
        '--wikipedia-dataset',
        default='facebook/kilt_wikipedia',
        help='Wikipedia dataset to use (default: facebook/kilt_wikipedia)'
    )

    parser.add_argument(
        '--wikipedia-version',
        default='2019-08-01',
        help='Wikipedia dataset version (default: 2019-08-01)'
    )

    args = parser.parse_args()

    # Expand wildcards and resolve paths
    input_files = []
    for pattern in args.files:
        expanded = glob.glob(pattern)
        if expanded:
            input_files.extend([Path(f).resolve() for f in expanded])
        else:
            # Try as literal path
            p = Path(pattern)
            if p.exists():
                input_files.append(p.resolve())
            else:
                print(f"Warning: File not found: {pattern}")

    if not input_files:
        print("Error: No valid input files found")
        return 1

    # Remove duplicates and sort
    input_files = sorted(set(input_files))

    print(f"\nFound {len(input_files)} file(s) to process:")
    for f in input_files:
        print(f"  - {f}")

    # Load Wikipedia metadata once (shared across all files)
    print(f"\n{'='*80}")
    print("Loading Wikipedia metadata (shared for all files)")
    print(f"{'='*80}")
    print(f"Dataset: {args.wikipedia_dataset} ({args.wikipedia_version})")

    ds = load_dataset(
        path=args.wikipedia_dataset,
        name=args.wikipedia_version,
        split="full",
        cache_dir=CACHE_DIR,
        storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}}
    )

    print(f"✓ Loaded {len(ds):,} Wikipedia articles")

    # Convert to DataFrame
    print("Converting metadata to DataFrame...")
    wiki_df = ds.select_columns(["wikipedia_id", "wikipedia_title"]).to_pandas()
    print(f"✓ Ready to process files")

    # Process each file
    results = []
    for i, pageviews_file in enumerate(input_files, 1):
        print(f"\n\n{'#'*80}")
        print(f"# File {i}/{len(input_files)}")
        print(f"{'#'*80}")

        success = process_file(
            pageviews_file,
            wiki_df,
            args.output_dir,
            args.wiki_source
        )
        results.append((pageviews_file.name, success))

    # Summary
    print(f"\n\n{'='*80}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"\nResults:")

    success_count = sum(1 for _, success in results if success)
    for filename, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {status}: {filename}")

    print(f"\nTotal: {success_count}/{len(results)} files processed successfully")

    if success_count < len(results):
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())


# cd /Users/cyro/Documents/VSC/PopularityBias/scripts
# python process_pageviews_batch.py ../data/raw_wikiPop_dumps/pageviews-*