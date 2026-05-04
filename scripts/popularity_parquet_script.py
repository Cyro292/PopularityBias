import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm

ASSET_DIR = Path(__file__).parent.parent / "data" / "popularity"
INPUT_PATH = ASSET_DIR / "pageviews-202601-user"
OUTPUT_PATH = ASSET_DIR / "en_wikipedia-202601.parquet"

def create_parquet():
    file_size = INPUT_PATH.stat().st_size

    with open(INPUT_PATH, "r") as f, tqdm(total=file_size, unit="B", unit_scale=True) as pbar:
        reader = pd.read_csv(
            f,
            sep=" ",
            header=None,
            names=["source", "name", "id", "device", "views"],
            usecols=[0, 1, 2, 3, 4],       # skip name, y, z entirely
            chunksize=500_000,
            engine="c",
            dtype={"source": "category", "name": "str", "device": "category", "views": "str"},
        )

        writer = None

        for chunk in reader:
            pbar.update(f.tell() - pbar.n)  # real disk bytes

            chunk = chunk[chunk["source"] == "en.wikipedia"]
            if chunk.empty:
                continue

            chunk = chunk.drop(columns=["source"])
            chunk = chunk.dropna(subset=["id"])
            chunk["id"] = chunk["id"].astype("int64")
            chunk["views"] = pd.to_numeric(chunk["views"], errors="coerce").fillna(0).astype("int64")

            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(OUTPUT_PATH, table.schema)
            writer.write_table(table)

        if writer:
            writer.close()

def analyse_sources() -> dict[str, int]:
    total_counts: dict[str, int] = {}
    file_size = INPUT_PATH.stat().st_size
    with open(INPUT_PATH, "r") as f, tqdm(total=file_size, unit="B", unit_scale=True) as pbar:
        for chunk in pd.read_csv(
            f,
            sep=" ",
            header=None,
            names=["source"],
            usecols=[0],           # only parse the first column
            chunksize=500_000,
            engine="c",
            dtype={"source": "category"},  # category = far less RAM than object strings
        ):
            vc = chunk["source"].value_counts()
            for k, v in vc.items():
                total_counts[k] = total_counts.get(k, 0) + int(v)
            pbar.update(f.tell() - pbar.n)  # track actual disk bytes read
    return total_counts

def main():

    if OUTPUT_PATH.exists():
        print(f"Output file found: {OUTPUT_PATH}")
        df = pd.read_parquet(OUTPUT_PATH)
        df = df.groupby(["id"], as_index=False, observed=True)["views"].sum()
    
        print(len(df['id'] > 0))
        print(df[df["views"] > 1_000_000].head(20))
        return

    print(f"Creating Parquet file at: {OUTPUT_PATH}")
    create_parquet()

    print("Deduplicating globally — one row per id...")
    df = pd.read_parquet(OUTPUT_PATH)
    df = df.groupby("id", as_index=False)["views"].sum()
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Done. {len(df):,} unique articles written.")

if __name__ == "__main__":
    main()