"""Prepare QA datasets: load, filter to corpus, enrich with deciles, balance."""

from __future__ import annotations

import argparse
import gc
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, CACHE_DIR, SYTHETNIC_QA_PROMPT_PATH


DEFAULT_HF_QA_REPO = "Cyro1/popularity-enriched-qa-datasets"
DEFAULT_POPULARITY_DATASET = "Cyro1/enwiki_pageviews_m"

import dotenv
dotenv.load_dotenv()

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def load_qa_datasets(dataset_names: list[str], hf_repo: str, cache_dir: str | None) -> pd.DataFrame:
    """Load QA datasets from HuggingFace."""
    from datasets import load_dataset
    
    dfs = []
    for name in dataset_names:
        logger.info(f"Loading {name}...")
        ds = load_dataset(hf_repo, name, split="train+test", cache_dir=cache_dir)
        df = ds.to_pandas()
        df["dataset"] = name
        dfs.append(df)
        logger.info(f"  ✓ {len(df):,} questions")
        del ds
        gc.collect()
    
    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.dropna(subset=["wikipedia_id"])
    merged["wikipedia_id"] = merged["wikipedia_id"].astype(int)
    return merged


def filter_to_corpus(qa_df: pd.DataFrame, corpus_path: Path) -> pd.DataFrame:
    """Keep only questions whose wikipedia_id exists in the corpus."""
    import pyarrow.parquet as pq
    
    logger.info("Loading corpus IDs...")
    pf = pq.ParquetFile(corpus_path)
    
    corpus_ids = set()
    for batch in pf.iter_batches(batch_size=100_000, columns=["wikipedia_id"]):
        batch_df = batch.to_pandas()
        batch_df["wikipedia_id"] = batch_df["wikipedia_id"].astype(int)
        corpus_ids.update(batch_df["wikipedia_id"])
    
    before = len(qa_df)
    qa_df = qa_df[qa_df["wikipedia_id"].isin(corpus_ids)].copy()
    logger.info(f"Kept {len(qa_df):,} / {before:,} questions (in corpus)")
    
    return qa_df


def calculate_corpus_decile_boundaries(
    corpus_path: Path,
    batch_size: int = 100_000,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Calculate both unweighted and chunk-weighted decile boundaries from corpus.
    
    Memory-efficient streaming implementation.
    
    Args:
        corpus_path: Path to corpus parquet file
        batch_size: Batch size for streaming
        chunk_size: Chunk size for text splitting
        chunk_overlap: Chunk overlap for text splitting
    
    Returns:
        Tuple of (boundaries_unweighted, boundaries_weighted, stats_dict, id_to_pop)
        Both boundaries are arrays of length 11 (0th, 10th, ..., 100th percentiles)
    
    Never loads the full corpus into memory.
    """
    import pyarrow.parquet as pq
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    pf = pq.ParquetFile(corpus_path)
    total_rows = pf.metadata.num_rows
    
    logger.info(f"Calculating decile boundaries from {total_rows:,} corpus documents...")
    
    # Use lists for memory efficiency (avoid repeated array creation)
    id_to_pop: dict[int, float] = {}
    pops_unweighted = []  # Collect popularity values directly
    pops_weighted = []    # Collect chunk-weighted popularity values
    docs_processed = 0
    total_chunks = 0
    
    for batch in tqdm(
        pf.iter_batches(batch_size=batch_size, columns=["wikipedia_id", "popularity_avg", "text"]),
        total=(total_rows + batch_size - 1) // batch_size,
        desc="Reading corpus",
    ):
        batch_df = batch.to_pandas()
        batch_df["wikipedia_id"] = batch_df["wikipedia_id"].astype(int)
        
        # Drop rows without popularity
        valid = batch_df.dropna(subset=["popularity_avg"])
        valid = valid[valid["popularity_avg"] >= 0]
        
        if len(valid) == 0:
            del batch_df, valid
            gc.collect()
            continue
        
        # Process each row
        for _, row in valid.iterrows():
            wid = int(row["wikipedia_id"])
            pop = float(row["popularity_avg"])
            
            # Skip duplicates (keep first occurrence)
            if wid in id_to_pop:
                continue
            
            # Count chunks
            text_str = str(row.get("text") or "")
            if text_str:
                chunks = len(splitter.split_text(text_str))
                chunks = max(1, chunks)
            else:
                chunks = 1
            
            # Store mapping
            id_to_pop[wid] = pop
            
            # Append to lists (more efficient than np.repeat later)
            pops_unweighted.append(pop)
            pops_weighted.extend([pop] * chunks)
            total_chunks += chunks
        
        docs_processed += len(batch_df)
        del batch_df, valid
        gc.collect()
    
    unique_docs = len(id_to_pop)
    
    logger.info(f"Processed {docs_processed:,} rows → {unique_docs:,} unique docs with popularity")
    logger.info(f"Total chunks (after splitting): {total_chunks:,}")
    
    # Convert to numpy arrays (use float32 to save memory)
    pops_unweighted = np.array(pops_unweighted, dtype=np.float32)
    pops_weighted = np.array(pops_weighted, dtype=np.float32)
    
    # Calculate boundaries
    boundaries_unweighted = np.percentile(pops_unweighted, np.linspace(0, 100, 11))
    boundaries_weighted = np.percentile(pops_weighted, np.linspace(0, 100, 11))
    
    logger.info("Decile boundaries (unweighted - 1 doc = 1 count):")
    for i in range(10):
        logger.info(f"  Decile {i}: [{boundaries_unweighted[i]:.4f}, {boundaries_unweighted[i+1]:.4f})")
    
    logger.info(f"Decile boundaries (chunk-weighted - chunk_size={chunk_size}, overlap={chunk_overlap}):")
    for i in range(10):
        logger.info(f"  Decile {i}: [{boundaries_weighted[i]:.4f}, {boundaries_weighted[i+1]:.4f})")
    
    stats = {
        "total_documents": docs_processed,
        "unique_documents_with_popularity": unique_docs,
        "total_chunks_after_splitting": total_chunks,
    }
    
    # Clean up arrays
    del pops_unweighted, pops_weighted
    gc.collect()
    
    return boundaries_unweighted, boundaries_weighted, stats, id_to_pop


def calculate_deciles(
    qa_df: pd.DataFrame,
    corpus_path: Path,
    batch_size: int = 100_000,
    weight_by_chunks: bool = False,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> pd.DataFrame:
    """Calculate deciles by streaming through the corpus parquet file in batches.
    
    Args:
        weight_by_chunks: If True, use chunk-weighted boundaries.
            If False, use unweighted boundaries (1 doc = 1 count).
        chunk_size: Chunk size for text splitting.
        chunk_overlap: Chunk overlap for text splitting.
    
    Never loads the full corpus into memory.
    """
    if chunk_size is None:
        chunk_size = 1000
    if chunk_overlap is None:
        chunk_overlap = 100
    
    # Calculate boundaries using the shared function
    boundaries_unweighted, boundaries_weighted, stats, id_to_pop = calculate_corpus_decile_boundaries(
        corpus_path=corpus_path,
        batch_size=batch_size,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    # Choose which boundaries to use
    boundaries = boundaries_weighted if weight_by_chunks else boundaries_unweighted
    weight_label = "chunk-weighted" if weight_by_chunks else "unweighted"
    
    # Assign deciles via boundaries
    decile_lookup = {}
    for wid, pop in id_to_pop.items():
        d = int(np.searchsorted(boundaries[1:-1], pop, side="right"))  # 0-9
        decile_lookup[wid] = d
    
    del id_to_pop
    gc.collect()
    
    # Map to QA dataframe
    logger.info("Mapping deciles to QA...")
    qa_df = qa_df.copy()
    qa_df["decile"] = qa_df["wikipedia_id"].map(decile_lookup)
    
    # Also add popularity_avg if not already present
    if "popularity_avg" not in qa_df.columns:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(corpus_path)
        pop_lookup = {}
        for batch in pf.iter_batches(batch_size=batch_size, columns=["wikipedia_id", "popularity_avg"]):
            batch_df = batch.to_pandas()
            batch_df["wikipedia_id"] = batch_df["wikipedia_id"].astype(int)
            valid = batch_df.dropna(subset=["popularity_avg"])
            pop_lookup.update(dict(zip(valid["wikipedia_id"], valid["popularity_avg"])))
        qa_df["popularity_avg"] = qa_df["wikipedia_id"].map(pop_lookup)
    
    unmapped = qa_df["decile"].isna().sum()
    if unmapped:
        logger.warning(f"{unmapped:,} questions not in corpus (dropping)")
        qa_df = qa_df.dropna(subset=["decile"])
    
    qa_df["decile"] = qa_df["decile"].astype(int)
    
    dist = qa_df["decile"].value_counts().sort_index().to_dict()
    logger.info(f"QA decile distribution ({weight_label}): {dist}")
    logger.info(f"Total corpus docs: {stats['total_documents']:,} | Unique: {stats['unique_documents_with_popularity']:,}")
    
    return qa_df


def balance_by_decile(qa_df: pd.DataFrame, target: int | None = None) -> pd.DataFrame:
    """Downsample each decile to target count."""
    counts = qa_df["decile"].value_counts().sort_index()
    target = target or int(counts.min())
    
    logger.info(f"Balancing to {target} per decile...")
    
    balanced = []
    for decile in range(10):
        subset = qa_df[qa_df["decile"] == decile]
        n = len(subset)
        if n > target:
            subset = subset.sample(n=target, random_state=42)
            logger.debug(f"Decile {decile}: {n:,} → {target}")
        elif n < target:
            logger.warning(f"Decile {decile}: {n:,} (short by {target - n})")
        balanced.append(subset)
    
    result = pd.concat(balanced, ignore_index=True)
    logger.info(f"Balanced to {len(result):,} questions ({target} x 10)")
    return result


def sample_corpus_by_decile(
    corpus_path: Path,
    samples_per_decile: dict[int, int],
    columns: list[str] = ["wikipedia_id", "wikipedia_title", "text"],
    text_limit: int = 2000,
    seed: int = 42,
) -> list[dict]:
    """Sample documents from corpus using reservoir sampling on filtered batches.
    
    Never loads more than needed into RAM - uses streaming with early termination.
    """
    import pyarrow.parquet as pq
    import random
    
    random.seed(seed)
    sampled_docs = []
    
    for decile, needed in tqdm(samples_per_decile.items(), desc="Sampling corpus"):
        pf = pq.ParquetFile(corpus_path)
        
        # Reservoir sampling: maintain exactly 'needed' samples
        reservoir = []
        count = 0
        
        # Stream batches
        for batch in pf.iter_batches(batch_size=50_000, columns=columns + ["decile"]):
            batch_df = batch.to_pandas()
            
            # Filter to this decile
            batch_df = batch_df[batch_df["decile"] == decile]
            
            for _, row in batch_df.iterrows():
                count += 1
                
                if len(reservoir) < needed:
                    # Fill reservoir
                    doc = {col: row[col] for col in columns}
                    doc["decile"] = decile
                    if "text" in doc and doc["text"]:
                        doc["text"] = doc["text"][:text_limit]
                    reservoir.append(doc)
                else:
                    # Reservoir sampling: replace with probability needed/count
                    j = random.randint(0, count - 1)
                    if j < needed:
                        doc = {col: row[col] for col in columns}
                        doc["decile"] = decile
                        if "text" in doc and doc["text"]:
                            doc["text"] = doc["text"][:text_limit]
                        reservoir[j] = doc
            
            del batch_df
            
            # Early termination: if reservoir is full and we've seen enough samples
            if len(reservoir) >= needed and count > needed * 10:
                break
        
        if len(reservoir) == 0:
            logger.warning(f"Decile {decile}: no documents found")
        else:
            sampled_docs.extend(reservoir)
            logger.info(f"Decile {decile}: sampled {len(reservoir)} / {count}")
        
        del pf
    
    gc.collect()
    return sampled_docs


def generate_questions_from_docs(
    docs: list[dict],
    prompt_template: str,
    model_name: str = "gpt-4.1-nano",
) -> list[dict]:
    """Generate synthetic questions from sampled documents."""
    from llm.openAi_service import OpenAIService
    
    service = OpenAIService(model_name=model_name)
    questions = []
    
    for doc in tqdm(docs, desc="Generating questions"):
        try:
            prompt = prompt_template.format(passage=doc.get("text", ""))
            question_text = service.invoke(prompt)
            
            questions.append({
                "question_id": f"syn_{doc['decile']}_{len(questions)}",
                "question_text": question_text.strip(),
                "answer_texts": [],
                "wikipedia_id": doc["wikipedia_id"],
                "wikipedia_title": doc.get("wikipedia_title", ""),
                "decile": doc["decile"],
                "dataset": "synthetic",
            })
        except Exception as e:
            logger.warning(f"Failed: {e}")
    
    return questions


def generate_synthetic(
    qa_df: pd.DataFrame,
    corpus_path: Path,
    questions_per_decile: int = 100,
    model_name: str = "gpt-4.1-nano",
    prompt_path: Path | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Generate synthetic questions for underrepresented deciles."""
    
    # Load prompt
    prompt_path = prompt_path or SYTHETNIC_QA_PROMPT_PATH
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt not found: {prompt_path}")
    
    prompt_template = prompt_path.read_text().strip()
    logger.info(f"Loaded prompt from {prompt_path}")
    
    # Determine what's needed per decile
    existing = qa_df["decile"].value_counts().to_dict() if len(qa_df) > 0 else {}
    samples_needed = {}
    
    for decile in range(10):
        current = existing.get(decile, 0)
        needed = questions_per_decile - current
        if needed > 0:
            samples_needed[decile] = needed
            logger.info(f"Decile {decile}: need {needed} ({current} existing)")
    
    if not samples_needed:
        logger.info("All deciles have sufficient questions")
        return qa_df
    
    # Sample documents
    docs = sample_corpus_by_decile(corpus_path, samples_needed)
    
    # Generate questions
    new_questions = generate_questions_from_docs(docs, prompt_template, model_name)
    
    if new_questions:
        qa_df = pd.concat([qa_df, pd.DataFrame(new_questions)], ignore_index=True)
        logger.info(f"Added {len(new_questions)} synthetic questions")
    
    return qa_df


def prepare_qa_dataset(
    qa_datasets: list[str] | None = None,
    existing_qa_path: Path | None = None,
    corpus_path: Path | None = None,
    popularity_dataset: str = DEFAULT_POPULARITY_DATASET,
    output_path: Path = None,
    balance: bool = False,
    target_per_decile: int | None = None,
    generate_synthetic: bool = False,
    questions_per_decile: int | None = None,
    model_name: str = "gpt-4.1-nano",
    hf_repo: str = DEFAULT_HF_QA_REPO,
    cache_dir: str | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Main pipeline: load → filter → enrich → [synthetic] → balance → save."""
    
    logger.info("PREPARE QA DATASET")
    
    # Load
    logger.info("[1/4] LOAD")
    dfs = []
    if qa_datasets:
        dfs.append(load_qa_datasets(qa_datasets, hf_repo, cache_dir))
    if existing_qa_path and existing_qa_path.exists():
        df = pd.read_parquet(existing_qa_path)
        df["wikipedia_id"] = df["wikipedia_id"].astype(int)
        logger.info(f"Loaded {len(df):,} from {existing_qa_path.name}")
        dfs.append(df)
    
    if not dfs:
        if generate_synthetic:
            # Synthetic-only mode: start with empty dataframe
            if not corpus_path or not corpus_path.exists():
                raise ValueError("--corpus required for synthetic-only generation")
            logger.info("Synthetic-only mode: starting with empty QA set")
            qa_df = pd.DataFrame(columns=["question_id", "question_text", "answer_texts", "wikipedia_id", "wikipedia_title", "decile", "dataset"])
        else:
            raise ValueError("Provide --qa-datasets or --existing-qa (or use --generate-synthetic with --corpus)")
    else:
        qa_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total: {len(qa_df):,} questions")
    
    # Filter
    logger.info("[2/4] FILTER TO CORPUS")
    if len(qa_df) == 0:
        logger.info("(empty QA, skipping filter)")
    elif corpus_path and corpus_path.exists():
        qa_df = filter_to_corpus(qa_df, corpus_path)
    else:
        logger.info("(no corpus filter)")
    
    # Enrich
    logger.info("[3/4] ASSIGN DECILES")
    if len(qa_df) == 0:
        logger.info("(empty QA, skipping decile assignment)")
    else:
        logger.info("Calculating deciles from corpus...")
        if not corpus_path or not corpus_path.exists():
            raise ValueError("--corpus required to calculate deciles")
        qa_df = calculate_deciles(qa_df, corpus_path)
    
    # Synthetic generation
    if generate_synthetic:
        logger.info("[4/5] SYNTHETIC GENERATION")
        if not corpus_path or not corpus_path.exists():
            raise ValueError("--corpus required for synthetic generation")
        target = questions_per_decile or target_per_decile or 100
        qa_df = generate_synthetic_fn(
            qa_df,
            corpus_path=corpus_path,
            questions_per_decile=target,
            model_name=model_name,
        )
    
    # Balance
    logger.info("[5/5] BALANCE" if generate_synthetic else "[4/4] BALANCE")
    if balance:
        qa_df = balance_by_decile(qa_df, target_per_decile)
    else:
        logger.info("(skipped)")
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    qa_df.to_parquet(output_path, index=False)
    
    logger.info(f"Saved {len(qa_df):,} questions to {output_path}")
    logger.info(f"Distribution: {qa_df['decile'].value_counts().sort_index().to_dict()}")
    
    return qa_df


# Alias for internal use (avoid name collision with parameter)
generate_synthetic_fn = generate_synthetic


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--qa-datasets", nargs="*")
    p.add_argument("--existing-qa", type=Path)
    p.add_argument("--corpus", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--popularity-dataset", default=DEFAULT_POPULARITY_DATASET)
    p.add_argument("--balance", action="store_true")
    p.add_argument("--target-per-decile", type=int)
    p.add_argument("--generate-synthetic", action="store_true", help="Generate synthetic questions for underrepresented deciles")
    p.add_argument("--questions-per-decile", type=int, help="Target questions per decile for synthetic generation")
    p.add_argument("--model-name", default="gpt-4.1-nano", help="OpenAI model for synthetic generation")
    p.add_argument("--hf-repo", default=DEFAULT_HF_QA_REPO)
    p.add_argument("--cache-dir", type=Path)
    
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    
    prepare_qa_dataset(
        qa_datasets=args.qa_datasets,
        existing_qa_path=args.existing_qa,
        corpus_path=args.corpus,
        popularity_dataset=args.popularity_dataset,
        output_path=args.output,
        balance=args.balance,
        target_per_decile=args.target_per_decile,
        generate_synthetic=args.generate_synthetic,
        questions_per_decile=args.questions_per_decile,
        model_name=args.model_name,
        hf_repo=args.hf_repo,
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
    )


if __name__ == "__main__":
    main()
