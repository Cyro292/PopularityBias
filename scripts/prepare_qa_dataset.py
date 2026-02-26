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
from helpers.decile_utils import (
    compute_corpus_boundaries as _compute_corpus_boundaries,
    assign_decile,
    assign_both_deciles,
    boundaries_to_metadata,
    COL_DECILE_UNWEIGHTED,
    COL_DECILE_CHUNK_WEIGHTED,
    COL_POPULARITY,
    NUM_DECILES,
)


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
) -> tuple[np.ndarray, np.ndarray, dict, dict[int, float]]:
    """Calculate both unweighted and chunk-weighted decile boundaries from corpus.

    Delegates to ``rag.decile_utils.compute_corpus_boundaries`` — kept here
    for backward compatibility with callers that import from this module.
    """
    return _compute_corpus_boundaries(
        corpus_path=corpus_path,
        batch_size=batch_size,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def calculate_deciles(
    qa_df: pd.DataFrame,
    corpus_path: Path,
    batch_size: int = 100_000,
    weight_by_chunks: bool = False,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> pd.DataFrame:
    """Assign **both** decile columns to the QA dataframe.

    Uses ``rag.decile_utils`` for all boundary computation and bin
    assignment so that retrieval and evaluation notebooks produce
    identical labels.

    The legacy ``decile`` column is kept for backward compatibility
    and mirrors whichever mode *weight_by_chunks* selects.
    """
    chunk_size = chunk_size or 1000
    chunk_overlap = chunk_overlap or 100

    # 1. Boundaries
    boundaries_uw, boundaries_cw, stats, id_to_pop = calculate_corpus_decile_boundaries(
        corpus_path=corpus_path,
        batch_size=batch_size,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # 2. Build a tiny lookup DataFrame so we can use assign_both_deciles
    pop_df = pd.DataFrame([
        {"wikipedia_id": wid, COL_POPULARITY: pop}
        for wid, pop in id_to_pop.items()
    ])
    pop_df = assign_both_deciles(pop_df, boundaries_uw, boundaries_cw,
                                 popularity_col=COL_POPULARITY, drop_missing=False)

    del id_to_pop
    gc.collect()

    # 3. Map both decile columns onto the QA frame
    qa_df = qa_df.copy()
    lookup_uw = pop_df.set_index("wikipedia_id")[COL_DECILE_UNWEIGHTED]
    lookup_cw = pop_df.set_index("wikipedia_id")[COL_DECILE_CHUNK_WEIGHTED]

    qa_df[COL_DECILE_UNWEIGHTED] = qa_df["wikipedia_id"].map(lookup_uw)
    qa_df[COL_DECILE_CHUNK_WEIGHTED] = qa_df["wikipedia_id"].map(lookup_cw)

    # Legacy column — mirrors the chosen mode
    legacy_col = COL_DECILE_CHUNK_WEIGHTED if weight_by_chunks else COL_DECILE_UNWEIGHTED
    qa_df["decile"] = qa_df[legacy_col]

    unmapped = qa_df[COL_DECILE_UNWEIGHTED].isna().sum()
    if unmapped:
        logger.warning("%s questions not in corpus (dropping)", f"{unmapped:,}")
        qa_df = qa_df.dropna(subset=[COL_DECILE_UNWEIGHTED, COL_DECILE_CHUNK_WEIGHTED])

    for col in (COL_DECILE_UNWEIGHTED, COL_DECILE_CHUNK_WEIGHTED, "decile"):
        qa_df[col] = qa_df[col].astype(int)

    weight_label = "chunk-weighted" if weight_by_chunks else "unweighted"
    dist = qa_df["decile"].value_counts().sort_index().to_dict()
    logger.info("QA decile distribution (%s): %s", weight_label, dist)
    logger.info("Total corpus docs: %s | Unique: %s",
                f"{stats['total_documents']:,}",
                f"{stats['unique_documents_with_popularity']:,}")

    return qa_df


def balance_per_dataset(qa_df: pd.DataFrame, total_target: int | None = None) -> pd.DataFrame:
    """Equalise question count across datasets using a fill-from-bottom approach.

    Distributes *total_target* questions evenly across datasets.  Datasets that
    have fewer samples than their equal share contribute everything they have;
    their unused quota is redistributed proportionally to the remaining datasets.

    Examples (total_target=5000, 2 datasets):
      ds_x=120k, ds_y=10k  →  2500 + 2500  (equal split)
      ds_x=120k, ds_y=5    →  4995 + 5     (ds_y exhausted; remainder to ds_x)

    If *total_target* is None, defaults to min(dataset_sizes) × n_datasets.
    """
    if "dataset" not in qa_df.columns:
        logger.warning("No 'dataset' column — skipping per-dataset balancing")
        return qa_df

    groups = {d: qa_df[qa_df["dataset"] == d] for d in qa_df["dataset"].unique()}
    # Sort smallest first so underflows are resolved before larger datasets
    datasets_sorted = sorted(groups, key=lambda d: len(groups[d]))
    n = len(datasets_sorted)

    if total_target is None:
        total_target = len(groups[datasets_sorted[0]]) * n  # min × n

    remaining_quota = total_target
    result_parts = []
    for i, ds in enumerate(datasets_sorted):
        remaining_n = n - i
        per_ds = remaining_quota // remaining_n
        available = len(groups[ds])
        take = min(available, per_ds)
        part = groups[ds].sample(n=take, random_state=42) if take < available else groups[ds]
        result_parts.append(part)
        remaining_quota -= take
        logger.info(f"  Per-dataset [{ds}]: {available:,} available → {take:,} taken")

    result = pd.concat(result_parts, ignore_index=True)
    logger.info(f"After per-dataset balance: {len(result):,} questions total")
    return result


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
    balance_datasets: bool = False,
    target_per_decile: int | None = None,
    generate_synthetic: bool = False,
    questions_per_decile: int | None = None,
    model_name: str = "gpt-4.1-nano",
    hf_repo: str = DEFAULT_HF_QA_REPO,
    cache_dir: str | None = None,
    weight_by_chunks: bool = False,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
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
        
        if weight_by_chunks:
            if chunk_size is None or chunk_overlap is None:
                raise ValueError("chunk_size and chunk_overlap must be provided when weight_by_chunks is True")

            logger.info(f"Using chunk-weighted deciles (chunk_size={chunk_size}, chunk_overlap={chunk_overlap})")
            qa_df = calculate_deciles(qa_df, corpus_path, weight_by_chunks=weight_by_chunks, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        else:
            logger.info("Using unweighted deciles (1 doc = 1 count)")
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
    
    # Per-dataset equalisation (before decile balance so each decile sees an
    # even dataset mix; uses fill-from-bottom so tiny datasets aren't wasted)
    if balance_datasets and len(qa_df) > 0 and "dataset" in qa_df.columns:
        logger.info("[%s] BALANCE PER DATASET" % ("5.5/6" if generate_synthetic else "4.5/5"))
        ds_total_target = (target_per_decile * 10) if (balance and target_per_decile) else None
        qa_df = balance_per_dataset(qa_df, total_target=ds_total_target)

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
