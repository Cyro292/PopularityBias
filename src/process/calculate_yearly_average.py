"""
Calculate yearly average popularity from monthly popularity tables.

Usage:
    python calculate_yearly_average.py
"""

import sys
sys.path.insert(0, "..")

import os
import pandas as pd
from config import DATA_DIR

# ============== CONFIGURATION ==============
YEAR = 2022
# ============================================


def calculate_yearly_average(year: int) -> pd.DataFrame:
    """Load monthly files and return DataFrame with average popularity."""
    files = sorted(DATA_DIR.glob(f"popularity_table_{year}*.parquet"))
    
    if not files:
        raise FileNotFoundError(f"No files found for year {year}")
    
    dfs = [pd.read_parquet(f)[["id", "title", "popularity"]] for f in files]
    combined = pd.concat(dfs)
    
    result = combined.groupby(["id", "title"], as_index=False).agg({
        "popularity": "mean"
    }).rename(columns={"popularity": "popularity_avg"})
    
    return result


def main():
    output_file = DATA_DIR / f"popularity_table_{YEAR}_avg.parquet"
    result = calculate_yearly_average(YEAR)
    result.to_parquet(output_file, index=False)
    print(f"Saved {len(result):,} articles to {output_file} ({os.path.getsize(output_file)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
