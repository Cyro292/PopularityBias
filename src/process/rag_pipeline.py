from __future__ import annotations

from dataclasses import dataclass, field
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
from src.rag.elasticsearch_rag_service import ElasticsearchRagService
from src.evaluator.binary_evaluator import BinaryEvaluator
from src.evaluator.base import EvaluationObjects, EvaluationResult
from src.rag.utils import IndexingConfig
from config import DATA_DIR
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


@dataclass(frozen=True)
class PipelineConfig:
    output_dir: str = "evaluation_results_nq_turbo"  # Directory to save evaluation results
    dataset_names: list[str] = field(default_factory=lambda: ["natural_questions", "trivia_qa", "pop_qa", "fever", "trex"])  # List of HuggingFace dataset names to load questions from
    questions_per_decile: int = 100  # Number of questions to sample from each decile (set to -1 for all)
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"  # Model name for the ModalLLMService (e.g. "Qwen/Qwen3-1.7B" or a local GGUF model)
    requests_per_second_api: int = 8
    collection_name: str = "wiki_full_bil"
    es_url: str = os.getenv("ELASTICSEARCH_ENDPOINT", "")
    es_user: str = os.getenv("ELASTICSEARCH_USERNAME", "")
    es_password: str = os.getenv("ELASTICSEARCH_PASSWORD", "")
    chunk_size: int = 1000
    chunk_overlap: int = 100
    embedding_model: str = "Lajavaness/bilingual-embedding-small"
    embedding_provider: str = "huggingface"
    request_batch_size: int = 254
    gpu_batch_size: int = 64
    top_k: int = 1  # Number of top documents to retrieve for each question
    num_candidates: int = 1000  # Number of candidates to retrieve for evaluation (after re-ranking)


def main():

    # Initialize services
    config = PipelineConfig()

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
    )
    rag_service = ElasticsearchRagService(
        config=IndexingConfig(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            embedding_model=config.embedding_model,
            embedding_provider=config.embedding_provider,
            request_batch_size=config.request_batch_size,
            gpu_batch_size=config.gpu_batch_size,
            normalise_embeddings=True,
            trust_remote_code=True,
        ),
        es_url=config.es_url,
        es_user=config.es_user,
        es_password=config.es_password,
    )
    llm_service = ModalLLMService(
        model_name=config.model_name,
        temperature=0.5,
        request_batch_size=40,
    )

    llm_evaluation_service = OpenAIService(model_name="gpt-4o-mini", requests_per_second=10)
    evaluator = BinaryEvaluator(evaluation_service=llm_evaluation_service)

    question_input.load()
    question_data = question_input.get_items()

    logger.info("Loaded %d questions", len(question_data))

    rag_service.load_index(config.collection_name)

    for strategy in ["approximation", "bm25"]:
        questions = [item.question_text for item in question_data]
        retrieved_docs_with_scores = rag_service.batch_retrieve_with_scores(
            questions,
            top_k=config.top_k,
            strategy=strategy,
            search_workers=6,
            msearch_batch_size=5,
            num_candidates=config.num_candidates,
        )
        retrieved_docs = [[doc for doc, _score in docs] for docs in retrieved_docs_with_scores]

        answer_prompts = [
            f"Documents: {','.join([doc.page_content for doc in docs])}\nQuestion: {q.question_text}"
            for q, docs in zip(question_data, retrieved_docs)
        ]
        answers = llm_service.batch_generate(answer_prompts)

        evaluation_objects_list = [
            EvaluationObjects(
                id=q.question_id,
                question=q.question_text,
                answer="",
                page_content=q.page_content,
                proposed_answer=answer,
                retrieved_docs=docs,
                metadata={"wikipedia_id": q.wikipedia_id, "decile": q.decile, "dataset": q.dataset, "strategy": strategy, "retrieved_doc_ids": [doc.metadata.get("wikipedia_id", "") for doc in docs]},
            )
            for q, answer, docs in zip(
                question_data, answers, retrieved_docs
            )
        ]

        evaluation_results: list[EvaluationResult] = evaluator.evaluate(evaluation_objects_list)
        
        evaluation_results_dicts = [result.__dict__ for result in evaluation_results]
        results_df = pd.DataFrame(evaluation_results_dicts)
        results_df.to_csv(output_folder / f"evaluation_results_{strategy}.csv", index=False)
        logger.info("Saved evaluation results for strategy '%s'", strategy)

if __name__ == "__main__":
    main()