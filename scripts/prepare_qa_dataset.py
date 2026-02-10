"""Prepare QA datasets for RAG evaluation: load, merge, enrich, balance, generate.

This script handles the full QA preparation pipeline:
  1. Load QA datasets from HuggingFace
  2. Merge into a single DataFrame
  3. Enrich with popularity deciles
  4. Balance across popularity deciles
  5. (Optional) Generate synthetic questions from a corpus

Usage examples:

    # Load, enrich, and balance QA datasets (no synthetic generation):
    python scripts/prepare_qa_dataset.py \
        --qa-datasets natural_questions triviaqa \
        --popularity-dataset Cyro1/enwiki_pageviews_m \
        --output data/wiki_1m_balanced_qa_b_nqtr/train_questions.parquet \
        --balance

    # Same + generate synthetic to fill under-represented deciles:
    python scripts/prepare_qa_dataset.py \
        --qa-datasets natural_questions triviaqa \
        --popularity-dataset Cyro1/enwiki_pageviews_m \
        --corpus data/wiki_1m_balanced_qa_b_nqtr/wiki_corpus.parquet \
        --output data/wiki_1m_balanced_qa_b_nqtr/train_questions.parquet \
        --balance --generate-synthetic --questions-per-decile 200

    # Generate synthetic only (from existing parquet + corpus):
    python scripts/prepare_qa_dataset.py \
        --existing-qa data/.../train_questions.parquet \
        --corpus data/.../wiki_corpus.parquet \
        --output data/.../train_questions_augmented.parquet \
        --generate-synthetic --questions-per-decile 500
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import logging
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATA_DIR, CACHE_DIR

import dotenv

dotenv.load_dotenv()

logger = logging.getLogger(__name__)

# ── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_HF_QA_REPO = "Cyro1/popularity-enriched-qa-datasets"
DEFAULT_POPULARITY_DATASET = "Cyro1/enwiki_pageviews_m"
DEFAULT_PROMPT_PATH = Path(DATA_DIR) / "prompts" / "synthentic_question_generation_promt.txt"
DEFAULT_PROMPT = (
    "Given this context/document passage, generate one or more relevant questions "
    "that a user might ask based on the passage.\n\nDocument: {passage}"
)
DEFAULT_MODEL = "gpt-4.1-nano"
DEFAULT_QUESTIONS_PER_DECILE = 200
DEFAULT_BATCH_SIZE = 500
DEFAULT_TEXT_FIELD = "text"
DEFAULT_MAX_PASSAGE_CHARS = 2000


# ============================================================================
# 1. LOAD & MERGE
# ============================================================================


def load_qa_datasets(
    dataset_names: Sequence[str],
    *,
    hf_repo: str = DEFAULT_HF_QA_REPO,
    cache_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Load one or more QA datasets from HuggingFace and merge them.

    Args:
        dataset_names: HuggingFace config names (e.g. ["natural_questions", "triviaqa"]).
        hf_repo: HuggingFace dataset repository.
        cache_dir: Directory for HuggingFace cache.

    Returns:
        Merged DataFrame with a ``dataset`` column identifying the source.
    """
    from datasets import load_dataset

    cache = str(cache_dir) if cache_dir else None
    dfs: list[pd.DataFrame] = []

    for name in dataset_names:
        ds = load_dataset(hf_repo, name, split="train+test", cache_dir=cache)
        df = ds.to_pandas()
        df["dataset"] = name
        dfs.append(df)
        print(f"  {name}: {len(df):,} questions")

    merged = pd.concat(dfs, ignore_index=True)

    # Normalize column names
    if "rank_avg" in merged.columns:
        if "popularity_rank" in merged.columns:
            merged["popularity_rank"] = merged["popularity_rank"].combine_first(
                merged["rank_avg"]
            )
            merged = merged.drop(columns=["rank_avg"])
        else:
            merged = merged.rename(columns={"rank_avg": "popularity_rank"})

    # Clean IDs
    merged = merged.dropna(subset=["wikipedia_id"])
    merged["wikipedia_id"] = merged["wikipedia_id"].astype(int)

    if "is_synthetic" not in merged.columns:
        merged["is_synthetic"] = False

    print(f"✓ Merged QA: {len(merged):,} questions from {len(dataset_names)} dataset(s)")
    return merged


def load_existing_qa(path: Path) -> pd.DataFrame:
    """Load an existing QA parquet file, normalizing columns."""
    df = pd.read_parquet(path)
    if "question_text" not in df.columns and "question" in df.columns:
        df = df.rename(columns={"question": "question_text"})
    df["wikipedia_id"] = pd.to_numeric(df["wikipedia_id"], errors="coerce").astype(int)
    if "is_synthetic" not in df.columns:
        df["is_synthetic"] = False
    print(f"✓ Loaded existing QA: {len(df):,} questions from {path}")
    return df


# ============================================================================
# 2. ENRICH WITH POPULARITY DECILES
# ============================================================================


def enrich_with_deciles(
    qa_df: pd.DataFrame,
    popularity_dataset: str = DEFAULT_POPULARITY_DATASET,
    *,
    cache_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Add global popularity deciles to a QA DataFrame.

    Loads the full popularity dataset, computes global deciles, and maps them
    onto ``qa_df`` via ``wikipedia_id``. Rows with unknown deciles are dropped.

    Args:
        qa_df: DataFrame with a ``wikipedia_id`` column.
        popularity_dataset: HuggingFace dataset for popularity scores.
        cache_dir: HuggingFace cache directory.

    Returns:
        Enriched DataFrame (rows with unknown decile are removed).
    """
    from datasets import load_dataset

    print("Loading popularity data for decile calculation...")
    cache = str(cache_dir) if cache_dir else None
    pop_ds = load_dataset(popularity_dataset, split="train+test", cache_dir=cache)

    cols = pop_ds.column_names
    id_col = "wikipedia_id" if "wikipedia_id" in cols else "id"

    pop_df = pop_ds.select_columns([id_col, "popularity_avg"]).to_pandas()
    pop_df[id_col] = pd.to_numeric(pop_df[id_col], errors="coerce").fillna(-1).astype(int)

    print("  Calculating global deciles...")
    pop_df["decile"] = pd.qcut(
        pop_df["popularity_avg"].rank(method="first"),
        10,
        labels=False,
    )

    decile_lookup = pop_df.set_index(id_col)["decile"].to_dict()
    pop_lookup = pop_df.set_index(id_col)["popularity_avg"].to_dict()

    del pop_ds, pop_df
    gc.collect()

    qa_df = qa_df.copy()
    qa_df["decile"] = qa_df["wikipedia_id"].map(decile_lookup).fillna(-1).astype(int)

    # Fill popularity_avg if missing
    if "popularity_avg" not in qa_df.columns:
        qa_df["popularity_avg"] = qa_df["wikipedia_id"].map(pop_lookup)
    else:
        qa_df["popularity_avg"] = qa_df["popularity_avg"].combine_first(
            qa_df["wikipedia_id"].map(pop_lookup)
        )

    invalid = (qa_df["decile"] == -1).sum()
    if invalid:
        print(f"  Dropped {invalid} questions with unknown decile")
    qa_df = qa_df[qa_df["decile"] != -1].copy()

    print(f"✓ Enriched: {len(qa_df):,} questions with deciles")
    return qa_df


# ============================================================================
# 3. BALANCE
# ============================================================================


def balance_by_decile(
    qa_df: pd.DataFrame,
    *,
    target_per_decile: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Balance a QA DataFrame so each popularity decile has equal representation.

    Args:
        qa_df: DataFrame with a ``decile`` column (0-9).
        target_per_decile: Explicit target count. If ``None``, uses the minimum
            decile count (i.e. downsample to the smallest decile).
        random_state: Seed for reproducible sampling.

    Returns:
        Balanced DataFrame.
    """
    counts = qa_df["decile"].value_counts().sort_index()
    target = target_per_decile or int(counts.min())
    print(f"  Balancing to {target} questions per decile...")

    balanced = []
    for decile in range(10):
        subset = qa_df[qa_df["decile"] == decile]
        if len(subset) > target:
            subset = subset.sample(n=target, random_state=random_state)
        balanced.append(subset)

    result = pd.concat(balanced, ignore_index=True)
    print(f"✓ Balanced: {len(result):,} questions ({target} × 10 deciles)")
    print(f"  Distribution:\n{result['decile'].value_counts().sort_index()}")
    return result


# ============================================================================
# 4. SYNTHETIC QUESTION GENERATION
# ============================================================================


def load_prompt(prompt_path: Path | None = None) -> str:
    """Load the question generation prompt template."""
    path = prompt_path or DEFAULT_PROMPT_PATH
    if path.exists():
        return path.read_text().strip()
    logger.warning(f"Prompt {path} not found, using default")
    return DEFAULT_PROMPT


def load_corpus(corpus_path: Path, text_field: str = DEFAULT_TEXT_FIELD) -> pd.DataFrame:
    """Load a corpus parquet and validate required columns."""
    df = pd.read_parquet(corpus_path)
    required = {text_field, "wikipedia_id", "decile"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Corpus missing columns: {missing}. Available: {list(df.columns)}")
    df["wikipedia_id"] = pd.to_numeric(df["wikipedia_id"], errors="coerce").astype(int)
    df["decile"] = pd.to_numeric(df["decile"], errors="coerce").fillna(-1).astype(int)
    return df


async def _generate_for_decile(
    corpus_df: pd.DataFrame,
    decile: int,
    needed: int,
    *,
    llm,
    prompt_template: str,
    text_field: str,
    max_passage_chars: int,
    batch_size: int,
) -> list[dict]:
    """Generate synthetic questions for one decile."""
    decile_docs = corpus_df[corpus_df["decile"] == decile]
    if decile_docs.empty:
        return []

    sampled = decile_docs.sample(
        n=needed, replace=(needed > len(decile_docs)), random_state=42 + decile
    )

    async def _gen_one(row):
        text = str(row[text_field])[:max_passage_chars]
        prompt = prompt_template.format(passage=text)
        try:
            response = await llm.ainvoke(prompt)
            question = response.strip()
            if not question:
                return None
            return {
                "question_text": question,
                "answer_texts": [],
                "wikipedia_id": int(row["wikipedia_id"]),
                "wikipedia_title": row.get("wikipedia_title"),
                "dataset": "synthetic",
                "decile": decile,
                "is_synthetic": True,
                "popularity_avg": row.get("popularity_avg"),
                "popularity_rank": row.get("popularity_rank"),
            }
        except Exception as e:
            logger.debug(f"Generation failed for doc {row['wikipedia_id']}: {e}")
            return None

    tasks = [_gen_one(row) for _, row in sampled.iterrows()]
    results = []
    for start in range(0, len(tasks), batch_size):
        batch = tasks[start : start + batch_size]
        for coro in tqdm(
            asyncio.as_completed(batch),
            total=len(batch),
            desc=f"Decile {decile}",
            leave=False,
        ):
            result = await coro
            if result:
                results.append(result)
    return results


def _run_async(coro):
    """Run async coroutine, handling Jupyter / nested event loops."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    else:
        import nest_asyncio
        nest_asyncio.apply()
        return loop.run_until_complete(coro)


def generate_synthetic(
    qa_df: pd.DataFrame,
    corpus_path: Path,
    *,
    questions_per_decile: int = DEFAULT_QUESTIONS_PER_DECILE,
    model_name: str = DEFAULT_MODEL,
    prompt_path: Path | None = None,
    text_field: str = DEFAULT_TEXT_FIELD,
    max_passage_chars: int = DEFAULT_MAX_PASSAGE_CHARS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    temperature: float = 0.7,
) -> pd.DataFrame:
    """Augment a QA DataFrame with synthetic questions to reach target per decile.

    Only generates for deciles that have fewer than ``questions_per_decile``.

    Args:
        qa_df: Existing QA with ``decile`` column.
        corpus_path: Path to the wiki corpus parquet.
        questions_per_decile: Target count per decile.
        model_name: OpenAI model to use.
        prompt_path: Custom prompt template file.
        text_field: Text column name in corpus.
        max_passage_chars: Max passage characters in prompt.
        batch_size: Concurrent generation batch size.
        temperature: LLM temperature.

    Returns:
        Augmented QA DataFrame (existing + new synthetic rows).
    """
    from llm.openAi_service import OpenAIService

    prompt_template = load_prompt(prompt_path)
    corpus_df = load_corpus(corpus_path, text_field=text_field)
    llm = OpenAIService(temperature=temperature, request_timeout=None, model_name=model_name)
    print(f"  LLM initialized: {model_name}")

    all_dfs = []
    for decile in range(10):
        existing = qa_df[qa_df["decile"] == decile] if "decile" in qa_df.columns else pd.DataFrame()
        current = len(existing)

        if current >= questions_per_decile:
            all_dfs.append(existing)
            print(f"  Decile {decile}: {current} existing (>= {questions_per_decile}) — kept")
            continue

        all_dfs.append(existing)
        needed = questions_per_decile - current
        print(f"  Decile {decile}: {current} existing → generating {needed}...")

        new_qs = _run_async(
            _generate_for_decile(
                corpus_df, decile, needed,
                llm=llm,
                prompt_template=prompt_template,
                text_field=text_field,
                max_passage_chars=max_passage_chars,
                batch_size=batch_size,
            )
        )
        if new_qs:
            all_dfs.append(pd.DataFrame(new_qs))
            print(f"    Generated {len(new_qs)}")
        if len(new_qs) < needed:
            print(f"    Shortfall: {needed - len(new_qs)}")

    result = pd.concat(all_dfs, ignore_index=True)
    result["wikipedia_id"] = result["wikipedia_id"].astype(int)
    print(f"✓ Augmented: {len(result):,} total questions")
    return result


# ============================================================================
# 5. FULL PIPELINE
# ============================================================================


def prepare_qa_dataset(
    *,
    qa_datasets: Sequence[str] | None = None,
    existing_qa_path: Path | None = None,
    popularity_dataset: str = DEFAULT_POPULARITY_DATASET,
    output_path: Path,
    balance: bool = False,
    target_per_decile: int | None = None,
    generate_synthetic_flag: bool = False,
    corpus_path: Path | None = None,
    questions_per_decile: int = DEFAULT_QUESTIONS_PER_DECILE,
    model_name: str = DEFAULT_MODEL,
    prompt_path: Path | None = None,
    text_field: str = DEFAULT_TEXT_FIELD,
    max_passage_chars: int = DEFAULT_MAX_PASSAGE_CHARS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    temperature: float = 0.7,
    hf_repo: str = DEFAULT_HF_QA_REPO,
    cache_dir: Path | str | None = None,
) -> pd.DataFrame:
    """End-to-end QA preparation: load → filter → enrich → generate → balance → save.

    At least one of ``qa_datasets`` or ``existing_qa_path`` must be provided
    (unless using pure synthetic mode with ``generate_synthetic_flag=True``).

    If ``corpus_path`` is provided, filters QA to only include questions whose
    ``wikipedia_id`` exists in the corpus (ensures all questions are answerable).

    Returns:
        Final QA DataFrame.
    """
    # ── Load ─────────────────────────────────────────────────────────────
    dfs: list[pd.DataFrame] = []

    if qa_datasets:
        print(f"\n📥 Loading QA datasets: {qa_datasets}")
        dfs.append(load_qa_datasets(qa_datasets, hf_repo=hf_repo, cache_dir=cache_dir))

    if existing_qa_path and existing_qa_path.exists():
        dfs.append(load_existing_qa(existing_qa_path))

    if not dfs and not generate_synthetic_flag:
        raise ValueError("Provide --qa-datasets and/or --existing-qa (or use --generate-synthetic)")

    if dfs:
        qa_df = pd.concat(dfs, ignore_index=True)

        # Deduplicate by (wikipedia_id, question_text) if merging
        if len(dfs) > 1 and "question_text" in qa_df.columns:
            before = len(qa_df)
            qa_df = qa_df.drop_duplicates(subset=["wikipedia_id", "question_text"], keep="first")
            dupes = before - len(qa_df)
            if dupes:
                print(f"  Removed {dupes:,} duplicate questions")

        print(f"\n📊 Combined QA: {len(qa_df):,} questions")

        # ── Filter to corpus ─────────────────────────────────────────────
        if corpus_path and corpus_path.exists():
            print(f"\n🔍 Filtering QA to match corpus...")
            corpus_ids = set(
                pd.read_parquet(corpus_path, columns=["wikipedia_id"])["wikipedia_id"].astype(int)
            )
            before = len(qa_df)
            qa_df = qa_df[qa_df["wikipedia_id"].isin(corpus_ids)].copy()
            removed = before - len(qa_df)
            if removed:
                print(f"  Removed {removed:,} questions not in corpus ({before:,} → {len(qa_df):,})")
            else:
                print(f"  ✓ All {len(qa_df):,} questions exist in corpus")

        # ── Enrich ───────────────────────────────────────────────────────
        if "decile" not in qa_df.columns or (qa_df["decile"] == -1).any():
            print(f"\n📈 Enriching with popularity deciles...")
            qa_df = enrich_with_deciles(qa_df, popularity_dataset, cache_dir=cache_dir)
        else:
            print(f"\n✓ Deciles already present")
    else:
        # Pure synthetic mode — start from empty DataFrame
        print("\n📊 No existing QA provided — generating all questions synthetically")
        qa_df = pd.DataFrame(columns=[
            "question_text", "answer_texts", "wikipedia_id", "wikipedia_title",
            "dataset", "decile", "is_synthetic", "popularity_avg", "popularity_rank",
        ])

    # ── Generate synthetic ───────────────────────────────────────────────
    if generate_synthetic_flag:
        if corpus_path is None or not corpus_path.exists():
            raise ValueError("--corpus required for synthetic generation")
        print(f"\n🤖 Generating synthetic questions (target: {questions_per_decile}/decile)...")
        qa_df = generate_synthetic(
            qa_df,
            corpus_path,
            questions_per_decile=questions_per_decile,
            model_name=model_name,
            prompt_path=prompt_path,
            text_field=text_field,
            max_passage_chars=max_passage_chars,
            batch_size=batch_size,
            temperature=temperature,
        )

    # ── Balance ──────────────────────────────────────────────────────────
    if balance:
        print(f"\n⚖️ Balancing...")
        qa_df = balance_by_decile(qa_df, target_per_decile=target_per_decile)

    # ── Save ─────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    qa_df.to_parquet(output_path, index=False, engine="pyarrow")

    print(f"\n✅ Done: {len(qa_df):,} questions saved to {output_path}")
    if "decile" in qa_df.columns:
        print(f"Distribution:\n{qa_df['decile'].value_counts().sort_index()}")
    if "dataset" in qa_df.columns:
        print(f"Sources:\n{qa_df['dataset'].value_counts()}")

    return qa_df


# ============================================================================
# CLI
# ============================================================================


def main():
    p = argparse.ArgumentParser(
        description="Prepare QA datasets: load, enrich, balance, generate.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--qa-datasets", nargs="*", default=None,
                   help="HuggingFace QA config names (e.g. natural_questions triviaqa).")
    p.add_argument("--existing-qa", type=Path, default=None,
                   help="Existing QA parquet to load/merge.")
    p.add_argument("--popularity-dataset", default=DEFAULT_POPULARITY_DATASET,
                   help="HuggingFace popularity dataset.")
    p.add_argument("--output", type=Path, required=True,
                   help="Output parquet path.")
    p.add_argument("--balance", action="store_true",
                   help="Balance deciles (downsample to smallest).")
    p.add_argument("--target-per-decile", type=int, default=None,
                   help="Explicit per-decile target for balancing.")
    p.add_argument("--generate-synthetic", action="store_true",
                   help="Generate synthetic questions for under-represented deciles.")
    p.add_argument("--corpus", type=Path, default=None,
                   help="Corpus parquet (required if --generate-synthetic).")
    p.add_argument("--questions-per-decile", type=int, default=DEFAULT_QUESTIONS_PER_DECILE,
                   help=f"Synthetic target per decile (default: {DEFAULT_QUESTIONS_PER_DECILE}).")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"OpenAI model for generation (default: {DEFAULT_MODEL}).")
    p.add_argument("--prompt", type=Path, default=None,
                   help="Custom prompt template file.")
    p.add_argument("--text-field", default=DEFAULT_TEXT_FIELD,
                   help=f"Corpus text column (default: {DEFAULT_TEXT_FIELD}).")
    p.add_argument("--max-chars", type=int, default=DEFAULT_MAX_PASSAGE_CHARS,
                   help=f"Max passage chars (default: {DEFAULT_MAX_PASSAGE_CHARS}).")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                   help=f"LLM batch size (default: {DEFAULT_BATCH_SIZE}).")
    p.add_argument("--temperature", type=float, default=0.7,
                   help="LLM temperature (default: 0.7).")
    p.add_argument("--hf-repo", default=DEFAULT_HF_QA_REPO,
                   help=f"HuggingFace QA repo (default: {DEFAULT_HF_QA_REPO}).")
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="HuggingFace cache directory.")

    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    prepare_qa_dataset(
        qa_datasets=args.qa_datasets,
        existing_qa_path=args.existing_qa,
        popularity_dataset=args.popularity_dataset,
        output_path=args.output,
        balance=args.balance,
        target_per_decile=args.target_per_decile,
        generate_synthetic_flag=args.generate_synthetic,
        corpus_path=args.corpus,
        questions_per_decile=args.questions_per_decile,
        model_name=args.model,
        prompt_path=args.prompt,
        text_field=args.text_field,
        max_passage_chars=args.max_chars,
        batch_size=args.batch_size,
        temperature=args.temperature,
        hf_repo=args.hf_repo,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()
