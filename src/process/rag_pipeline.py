from dataclasses import dataclass
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
logger = logging.getLogger(__name__)

from src.corpus_handler.parquet_corpus_handler import ParquetCorpusHandler
from src.question_input.huggingface_cyro_input import HuggingFaceCyroInput
from src.llm.openAi_service import OpenAIService
from src.rag.elasticsearch_rag_service import ElasticsearchRagService
from src.evaluator.binary_evaluator import BinaryEvaluator
from src.evaluator.base import EvaluationObjects, EvaluationResult
from src.rag.utils import IndexingConfig
from config import DATA_DIR
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


@dataclass
class PipelineConfig:

    output_dir: str = "evaluation_results_nq"  # Directory to save evaluation results
    dataset_names: list[str] = ["natural_questions"]  # List of datasets to load questions from
    questions_per_decile: int = 1  # Number of questions to sample from each decile (set to -1 for all)
    model_name: str = "gpt-4o-mini"
    es_url: str = os.getenv("ELASTICSEARCH_ENDPOINT", "")
    es_user: str = os.getenv("ELASTICSEARCH_USERNAME", "")
    es_password: str = os.getenv("ELASTICSEARCH_PASSWORD", "")
    collection_name: str = "wiki_full_bil"
    es_strategy: str = "vector"
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
    llm_service = OpenAIService(model_name=config.model_name)
    evaluator = BinaryEvaluator(evaluation_service=llm_service)

    question_input.load()
    question_data = question_input.get_items()

    questions = [item.question_text for item in question_data]
    page_contents = [item.page_content for item in question_data]

    logger.info("Loaded %d questions", len(questions))

    rag_service.load_index(config.collection_name)

    for strategy in ["vector", "bm25"]:
        retrieved_docs_with_scores = rag_service.batch_retrieve_with_scores(
            questions,
            top_k=config.top_k,
            strategy=strategy,
            num_candidates=config.num_candidates,
        )
        retrieved_docs = [[doc for doc, _score in docs] for docs in retrieved_docs_with_scores]

        answer_prompts = [
            f"Document: {docs[0].page_content}\nQuestion: {question}"
            for question, docs in zip(questions, retrieved_docs)
        ]
        answers = llm_service.batch_generate(answer_prompts)

        evaluation_objects_list = [
            EvaluationObjects(
                id="",
                question=question,
                proposed_answer=answer,
                answer="",
                retrieved_docs=docs,
                page_content=page_content,
            )
            for question, answer, docs, page_content in zip(
                questions, answers, retrieved_docs, page_contents
            )
        ]

        evaluation_results = evaluator.evaluate(evaluation_objects_list)
        
        evaluation_results_dicts = [result.__dict__ for result in evaluation_results]
        results_df = pd.DataFrame(evaluation_results_dicts)
        results_df.to_csv(output_folder / f"evaluation_results_{strategy}.csv", index=False)
        logger.info("Saved evaluation results for strategy '%s'", strategy)

if __name__ == "__main__":
    main()