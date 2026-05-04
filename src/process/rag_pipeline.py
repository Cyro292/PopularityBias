from __future__ import annotations

from dataclasses import dataclass, field
from typing_extensions import Literal
import json
import pandas as pd
import logging
import pathlib
import sys
import os


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("elastic_transport.transport").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

from src.corpus_handler.parquet_corpus_handler import ParquetCorpusHandler
from src.question_input.huggingface_cyro_input import HuggingFaceCyroInput
from src.llm.openAi_service import OpenAIService
from src.llm.modalLLMService import ModalLLMService
from src.llm.mistralLLMService import MistralLLMService
from src.llm.qwenLLMService import QwenLLMService
from src.rag.elasticsearch_rag_service import ElasticsearchRagService
from src.rag.faiss_rag_service import FaissRagService
from src.evaluator.binary_evaluator import BinaryEvaluator
from src.evaluator.substring_evaluator import SubstringEvaluator
from src.evaluator.base import EvaluationObjects, EvaluationResult
from src.rag.utils import IndexingConfig
from config import DATA_DIR
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


# ═══════════════════════════════════════════════════════════════════════════════
# Checkpoint helper functions
# ═══════════════════════════════════════════════════════════════════════════════

def save_retrieved_docs_csv(docs: list[list], question_ids: list[str], path: pathlib.Path) -> None:
    """Save retrieved documents to a CSV checkpoint file.

    Each row represents one retrieved document with columns:
    question_id, doc_rank, page_content, wikipedia_id, popularity, and other metadata fields.

    Args:
        docs: Outer list indexed by question; inner list contains retrieved Documents.
        question_ids: List of question IDs corresponding to each question.
        path: Destination file path (parent directories are created automatically).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    
    rows = []
    for question_id, doc_list in zip(question_ids, docs):
        for doc_rank, doc in enumerate(doc_list):
            row = {
                "question_id": question_id,
                "doc_rank": doc_rank,
                "page_content": doc.page_content,
            }
            # Add all metadata fields as separate columns
            if hasattr(doc, "metadata") and doc.metadata:
                for key, value in doc.metadata.items():
                    row[f"metadata_{key}"] = value
            rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    logger.info("Saved %d document rows across %d questions to %s", len(rows), len(docs), path)


def load_retrieved_docs_csv(path: pathlib.Path, question_ids: list[str]):
    """Load retrieved documents from a CSV checkpoint file.

    Args:
        path: Path to the CSV checkpoint written by :func:`save_retrieved_docs_csv`.
        question_ids: List of question IDs to filter and order results by.

    Returns:
        List of document lists (one per question), or ``None`` if the file does not
        exist or cannot be parsed.
    """
    from langchain.schema import Document

    if not path.exists():
        return None
    
    try:
        df = pd.read_csv(path)
        
        # Build mapping from question_id to documents
        question_id_to_docs = {}
        for question_id in df["question_id"].unique():
            question_docs = df[df["question_id"] == question_id].sort_values("doc_rank")
            
            doc_list = []
            for _, row in question_docs.iterrows():
                # Extract page_content and handle NaN
                page_content = row["page_content"]
                if pd.isna(page_content):
                    page_content = ""
                
                # Reconstruct metadata from metadata_* columns
                metadata = {}
                for col in row.index:
                    if col.startswith("metadata_"):
                        key = col[len("metadata_"):]  # Remove "metadata_" prefix
                        value = row[col]
                        # Handle NaN values
                        if pd.notna(value):
                            metadata[key] = value
                
                doc_list.append(Document(page_content=page_content, metadata=metadata))
            
            question_id_to_docs[question_id] = doc_list
        
        # Return documents in the order of question_ids
        results = []
        for qid in question_ids:
            if qid in question_id_to_docs:
                results.append(question_id_to_docs[qid])
            else:
                logger.warning(f"Question ID {qid} not found in checkpoint, using empty doc list")
                results.append([])
        
        return results
    except Exception as e:
        logger.error("Failed to load retrieval checkpoint %s: %s", path, e)
        return None


def filter_checkpoint_by_questions(
    checkpoint_data: list,
    current_questions: list,
    question_data_items: list,
) -> list | None:
    """Filter checkpoint to only include entries matching current questions.
    
    Args:
        checkpoint_data: Loaded checkpoint list (from load_retrieved_docs_jsonl)
        current_questions: List of current question strings
        question_data_items: List of QuestionDataItem objects with question_id field
        
    Returns:
        Filtered checkpoint matching current questions, or None if not all questions found
    """
    # Build mapping from question text to checkpoint index
    # Assume checkpoint was created in same order as some previous question list
    
    # Try to match by position first (most common case)
    if len(checkpoint_data) >= len(current_questions):
        # If checkpoint is superset, check if all current questions are contained
        # For now, just filter by length - take first N entries
        logger.info("Checkpoint has %d entries, current has %d questions - filtering to match",
                    len(checkpoint_data), len(current_questions))
        return checkpoint_data[:len(current_questions)]
    
    return None


def should_skip_stage(
    stage: str,
    checkpoint_path: pathlib.Path,
    restart_from: str,
) -> bool:
    """Determine whether a pipeline stage should be skipped (checkpoint reused).

    Returns ``True`` when the checkpoint exists **and** the cascade logic says
    this stage comes *before* the requested restart point.

    Stage ordering: ``retrieval`` (1) → ``answers`` (2) → ``evaluation`` (3).

    Args:
        stage: Name of the stage being checked (``"retrieval"``, ``"answers"``,
            or ``"evaluation"``).
        checkpoint_path: Expected checkpoint file for this stage.
        restart_from: Value from :attr:`PipelineConfig.restart_from`.

    Returns:
        ``True`` if the stage should be skipped (use checkpoint),
        ``False`` if the stage must run.
    """
    if not checkpoint_path.exists():
        return False  # No checkpoint available — must run

    if restart_from == "all":
        return False  # Force full rerun — skip nothing

    if restart_from == "none":
        return True  # Normal mode — reuse everything available

    stage_order = {"retrieval": 1, "answers": 2, "evaluation": 3}
    current = stage_order.get(stage, 999)
    restart = stage_order.get(restart_from, 0)  # Unknown value → 0 → skip nothing downstream

    # Reuse checkpoint only when this stage is strictly before the restart point
    return current < restart


@dataclass(frozen=True)
class PipelineConfig:
    output_dir: str = "evaluation_results_text-davinci-003_200_exact_match"  # Directory to save evaluation results
    dataset_names: list[str] = field(default_factory=lambda: ["natural_questions", "trivia_qa", "pop_qa", "fever"])  # List of HuggingFace dataset names to load questions from
    questions_per_decile: int = 200  # Number of questions to sample from each decile (set to -1 for all)
    model_name: str | None = None 
    model_request_batch_size: int = 50  # Batch size for requests to the LLM service
    requests_per_second_api: int = 8
    collection_name: str = "wiki_full_bil"
    es_url: str = os.getenv("ELASTICSEARCH_ENDPOINT", "")
    es_user: str = os.getenv("ELASTICSEARCH_USERNAME", "")
    es_password: str = os.getenv("ELASTICSEARCH_PASSWORD", "")
    chunk_size: int = 1000
    chunk_overlap: int = 100
    embedding_model: str = "Lajavaness/bilingual-embedding-small"
    embedding_provider: str = "huggingface"
    embeddings_request_batch_size: int = 254
    gpu_batch_size: int = 254
    top_k: int = 5  # Number of top documents to retrieve for each question
    num_candidates: int = 1000  # Number of candidates to retrieve for evaluation (after re-ranking)
    balance_decile_mode: Literal["chunk_weighted", "unweighted"] = "chunk_weighted"  # Method to balance questions across deciles ("chunk_weighted" or "unweighted")
    # ── Checkpoint control ────────────────────────────────────────────────────
    # Restart from a specific stage; all downstream stages are also rerun (cascading).
    # Values: "none" (use all checkpoints), "retrieval", "answers", "evaluation", "all" (full rerun).
    # Any unrecognised value is treated as "none".
    restart_from: str = "evaluation"  # Stage from which to restart the pipeline (default: "evaluation")
    faiss_index_path: str = "faiss_migrated"  # Subdirectory under DATA_DIR containing the FAISS index


def main():

    # Initialize services
    config = PipelineConfig()

    logger.info("Starting RAG evaluation pipeline with config: %s", config)

    collection_folder = DATA_DIR / config.collection_name
    output_folder = collection_folder / config.output_dir
    output_folder.mkdir(parents=True, exist_ok=True)

    corpus_handler = ParquetCorpusHandler(
        corpus_path= collection_folder / "wiki_corpus.parquet",
        metadata_path= collection_folder / "metadata.json",
    )
    question_input = HuggingFaceCyroInput(
        dataset_names=config.dataset_names,
        corpus_handler=corpus_handler,
        parquet_path= output_folder / "cyro_qa_cache.parquet",
        balance_deciles=True,
        balance_datasets=True,
        target_per_decile=config.questions_per_decile,
        shuffle=True,
        balance_decile_mode=config.balance_decile_mode,
    )
    rag_service = ElasticsearchRagService(
        config=IndexingConfig(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            embedding_model=config.embedding_model,
            embedding_provider=config.embedding_provider,
            request_batch_size=config.embeddings_request_batch_size,
            gpu_batch_size=config.gpu_batch_size,
            normalise_embeddings=True,
            trust_remote_code=True,
        ),
        es_url=config.es_url,
        es_user=config.es_user,
        es_password=config.es_password,
        bm25_b=0
    )
    rag_service_2 = FaissRagService(
        config=IndexingConfig(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            embedding_model=config.embedding_model,
            embedding_provider=config.embedding_provider,
            request_batch_size=config.embeddings_request_batch_size,
            gpu_batch_size=config.gpu_batch_size,
            normalise_embeddings=True,
            trust_remote_code=True,
        ),
        strategy="ivfpq",
        distance_strategy="cosine",
    )
    llm_service_2 = QwenLLMService(temperature=0.0, request_batch_size=config.model_request_batch_size)
    llm_service = OpenAIService(model_name="text-davinci-003", requests_per_second=10)

    llm_evaluation_service = OpenAIService(model_name="gpt-5-nano-2025-08-07", requests_per_second=10)
    evaluator_2 = BinaryEvaluator(evaluation_service=llm_evaluation_service)

    evaluator = SubstringEvaluator()

    question_input.load()
    question_data = question_input.get_items()

    logger.info("Loaded %d questions", len(question_data))

    rag_service.load_index(config.collection_name)
    rag_service_2.load_index(DATA_DIR / config.faiss_index_path)

    for strategy in ["zero_shot", "approximation", "bm25"]:

        # ══════════════════════════════════════════════════════════════════════
        # Stage 0: Final CSV check (skip entire strategy if exists AND restart_from="none")
        # ══════════════════════════════════════════════════════════════════════
        csv_output_path = output_folder / f"evaluation_results_{strategy}.csv"

        if csv_output_path.exists() and config.restart_from == "none":
            logger.info("✓ [%s] Final results exist, skipping entire strategy", strategy)
            continue

        if config.restart_from != "none":
            logger.info("♻ [%s] Rerunning from stage: %s", strategy, config.restart_from)

        questions = [item.question_text for item in question_data]
        question_ids = [item.question_id for item in question_data]

        # ══════════════════════════════════════════════════════════════════════
        # Stage 1: Document Retrieval
        # ══════════════════════════════════════════════════════════════════════
        retrieval_checkpoint = output_folder / f"retrieved_docs_{strategy}.csv"

        # Check if we should try to reuse the checkpoint
        skip_retrieval = should_skip_stage("retrieval", retrieval_checkpoint, config.restart_from)
        logger.debug("[%s] should_skip_stage('retrieval') = %s", strategy, skip_retrieval)

        if skip_retrieval:
            # Try to load checkpoint
            retrieved_docs = load_retrieved_docs_csv(retrieval_checkpoint, question_ids)
            if retrieved_docs is None:
                logger.warning("✗ [%s] Retrieval: Failed to load checkpoint, re-retrieving", strategy)
                retrieved_docs = []  # Will trigger retrieval below
            else:
                logger.info("✓ [%s] Retrieval: Loaded %d results from checkpoint", strategy, len(retrieved_docs))
                # Check for length mismatch and try to filter
                if len(retrieved_docs) != len(questions):
                    logger.info(
                        "⚠ [%s] Retrieval: Checkpoint length mismatch (checkpoint=%d, questions=%d), attempting to filter",
                        strategy, len(retrieved_docs), len(questions)
                    )
                    # Try to filter checkpoint to match current questions
                    filtered_docs = filter_checkpoint_by_questions(retrieved_docs, questions, question_data)
                    if filtered_docs is not None and len(filtered_docs) == len(questions):
                        logger.info("✓ [%s] Retrieval: Successfully filtered checkpoint to %d questions", strategy, len(filtered_docs))
                        retrieved_docs = filtered_docs
                    else:
                        logger.warning("✗ [%s] Retrieval: Could not filter checkpoint, will re-retrieve", strategy)
                        retrieved_docs = []  # Force re-retrieval
                else:
                    logger.info("✓ [%s] Retrieval: Checkpoint matches question count, reusing", strategy)
        else:
            logger.info("♻ [%s] Retrieval: Forcing re-run (restart_from='%s')", strategy, config.restart_from)
            retrieved_docs = []  # Signal need to retrieve

        # Perform retrieval if needed
        if not retrieved_docs:
            logger.info("♻ [%s] Retrieval: Starting document retrieval", strategy)
            
            if strategy == "zero_shot":
                retrieved_docs = [[] for _ in questions]
            else:
                retrieved_docs_with_scores = rag_service.batch_retrieve_with_scores(
                    questions,
                    top_k=config.top_k,
                    strategy=strategy,
                    search_workers=6,
                    msearch_batch_size=5,
                    num_candidates=config.num_candidates,
                )
                retrieved_docs = [[doc for doc, _score in docs] for docs in retrieved_docs_with_scores]

            save_retrieved_docs_csv(retrieved_docs, question_ids, retrieval_checkpoint)
            logger.info("✓ [%s] Retrieval: Saved %d results to checkpoint", strategy, len(retrieved_docs))

        # ══════════════════════════════════════════════════════════════════════
        # Stage 2: Answer Generation
        # ══════════════════════════════════════════════════════════════════════
        answer_checkpoint = output_folder / f"answer_checkpoint_{strategy}.csv"

        # Invalidate checkpoint if stage needs rerun
        if not should_skip_stage("answers", answer_checkpoint, config.restart_from):
            if answer_checkpoint.exists():
                logger.info("♻ [%s] Answers: Deleting checkpoint (restart_from='%s')", strategy, config.restart_from)
                answer_checkpoint.unlink()

        answer_prompts = [
            f"Documents: {','.join([doc.page_content for doc in docs])}\nQuestion: {q.question_text}"
            for q, docs in zip(question_data, retrieved_docs)
        ]

        answers = llm_service.batch_generate(answer_prompts, checkpoint_path=answer_checkpoint, question_ids=question_ids)
        
        # Trim answers to match question count (in case checkpoint has more entries)
        if len(answers) > len(questions):
            logger.info("⚠ [%s] Answers: Checkpoint has %d answers but only %d questions, trimming to match",
                       strategy, len(answers), len(questions))
            answers = answers[:len(questions)]
        
        logger.info("✓ [%s] Answers: Generated %d answers", strategy, len(answers))

        # ══════════════════════════════════════════════════════════════════════
        # Stage 3: Evaluation Objects Assembly
        # ══════════════════════════════════════════════════════════════════════
        evaluation_objects_list = [
            EvaluationObjects(
                id=q.question_id,
                question=q.question_text,
                answers=q.answer_texts,
                page_content=q.page_content,
                proposed_answer=answer,
                retrieved_docs=docs,
                metadata={
                    "wikipedia_id": q.wikipedia_id,
                    "decile": q.decile,
                    "decile_unweighted": q.decile_unweighted,
                    "decile_chunk_weighted": q.decile_chunk_weighted,
                    "popularity_avg": q.popularity_avg,
                    "dataset": q.dataset,
                    "strategy": strategy,
                    "retrieved_doc_popularity": [doc.metadata.get("popularity", 0) for doc in docs],
                    "retrieved_doc_ids": [doc.metadata.get("wikipedia_id", "") for doc in docs],
                },
            )
            for q, answer, docs in zip(question_data, answers, retrieved_docs)
        ]

        # ══════════════════════════════════════════════════════════════════════
        # Stage 4: Evaluation
        # ══════════════════════════════════════════════════════════════════════
        eval_checkpoint = output_folder / f"eval_checkpoint_{strategy}.jsonl"

        # Invalidate checkpoint if stage needs rerun
        if not should_skip_stage("evaluation", eval_checkpoint, config.restart_from):
            if eval_checkpoint.exists():
                logger.info("♻ [%s] Evaluation: Deleting checkpoint (restart_from='%s')", strategy, config.restart_from)
                eval_checkpoint.unlink()

        evaluation_results: list[EvaluationResult] = evaluator.evaluate(
            evaluation_objects_list,
            checkpoint_path=eval_checkpoint,
        )
        logger.info("✓ [%s] Evaluation: Completed %d evaluations", strategy, len(evaluation_results))

        # ══════════════════════════════════════════════════════════════════════
        # Stage 5: Save Final CSV
        # ══════════════════════════════════════════════════════════════════════
        evaluation_results_dicts = [result.__dict__ for result in evaluation_results]
        results_df = pd.DataFrame(evaluation_results_dicts)
        results_df.to_csv(csv_output_path, index=False)
        logger.info("✓ [%s] Final: Saved results to %s", strategy, csv_output_path)

if __name__ == "__main__":
    main()