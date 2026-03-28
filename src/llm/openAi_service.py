"""
OpenAI Service using LangChain for LLM interactions and dataframe expansion.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
from langchain_openai import ChatOpenAI, OpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.rate_limiters import BaseRateLimiter, InMemoryRateLimiter
from pydantic import BaseModel, Field
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

# Handle imports for both module usage and direct script execution
try:
    from config import DATA_DIR
except ModuleNotFoundError:
    # If running as a script, add parent directory to path and try again
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from config import DATA_DIR

logger = logging.getLogger(__name__)


class OpenAIService:
    """Service for interacting with OpenAI models using LangChain."""

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        api_key: Optional[str] = None,
        use_chat: bool = True,
        request_timeout: Optional[float] = 30.0,
        rate_limiter: Optional[BaseRateLimiter] = None,
        rate_limit_rps: Optional[float] = 500 / 60,
        rate_limit_check_every: float = 0.1,
        rate_limit_max_burst: int = 10,
    ):
        """Initialize the OpenAI service.
        
        Args:
            model_name: Name of the OpenAI model to use (e.g., 'gpt-3.5-turbo', 'gpt-4', 'gpt-3.5-turbo-instruct').
            temperature: Sampling temperature (0-2). Higher values make output more random.
            max_tokens: Maximum number of tokens to generate.
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY environment variable.
            use_chat: If True, uses ChatOpenAI (for chat models). If False, uses OpenAI (for completion models).
            request_timeout: Request timeout passed to LangChain's OpenAI client.
            rate_limiter: Optional LangChain rate limiter (e.g., InMemoryRateLimiter).
            rate_limit_rps: Requests per second when auto-creating a limiter (None to disable).
            rate_limit_check_every: How often to check the limiter bucket (seconds).
            rate_limit_max_burst: Maximum burst size for the limiter bucket.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_chat = use_chat
        self.request_timeout = request_timeout
        if rate_limiter is None and rate_limit_rps is not None:
            rate_limiter = InMemoryRateLimiter(
                requests_per_second=rate_limit_rps,
                check_every_n_seconds=rate_limit_check_every,
                max_bucket_size=rate_limit_max_burst,
            )

        self.rate_limiter = rate_limiter

        # Get API key from parameter or environment
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Initialize the appropriate model
        if use_chat:
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
                request_timeout=request_timeout,
                rate_limiter=rate_limiter,
            )
        else:
            self.llm = OpenAI(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                openai_api_key=api_key,
                request_timeout=request_timeout,
                rate_limiter=rate_limiter,
            )

        logger.info(f"Initialized OpenAI service with model: {model_name}")

    def invoke(self, prompt: str, **kwargs) -> str:
        """Invoke the LLM with a prompt.
        
        Args:
            prompt: The input prompt/question.
            **kwargs: Additional arguments to pass to the model.
            
        Returns:
            The model's response as a string.
        """
        start = time.perf_counter()
        try:
            response = self.llm.invoke(prompt, **kwargs)
            if hasattr(response, "content"):
                return response.content
            return str(response)
        except Exception as e:
            logger.error(f"Error invoking LLM: {e}")
            raise
        finally:
            elapsed = time.perf_counter() - start
            logger.info("LLM request completed in %.2fs", elapsed)

    async def ainvoke(self, prompt: str, **kwargs) -> str:
        """Async invoke with basic metrics."""
        start = time.perf_counter()
        try:
            response = await self.llm.ainvoke(prompt, **kwargs)
            if hasattr(response, "content"):
                return response.content
            return str(response)
        except Exception as e:
            logger.error(f"Error invoking LLM async: {e}")
            raise

    def invoke_with_template(
        self,
        template: str,
        input_variables: dict[str, Any],
        **kwargs
    ) -> str:
        """Invoke the LLM with a prompt template.
        
        Args:
            template: Prompt template string with placeholders (e.g., "How to say {input} in {language}?").
            input_variables: Dictionary of variables to fill in the template.
            **kwargs: Additional arguments to pass to the model.
            
        Returns:
            The model's response as a string.
        """
        if self.use_chat:
            prompt = ChatPromptTemplate.from_template(template)
        else:
            prompt = PromptTemplate.from_template(template)
        
        chain = prompt | self.llm
        response = chain.invoke(input_variables, **kwargs)
        
        # Handle both ChatOpenAI (returns AIMessage) and OpenAI (returns string)
        if hasattr(response, "content"):
            return response.content
        return str(response)

    def is_text_relevant(
        self,
        question: str,
        text: str,
        prompt_template: Optional[str] = None,
        **kwargs,
    ) -> dict:
        """Determine if `text` is relevant to `question` using structured output.

        Uses LangChain's with_structured_output() to get a properly formatted response.

        Args:
            question: The question to evaluate relevance against.
            text: The text/passage to check for relevance.
            prompt_template: Optional custom prompt template. If None, loads from
                           DATA_DIR/prompts/binary_relevance_promt.txt
            **kwargs: Additional arguments to pass to the LLM.

        Returns:
            Dictionary with keys:
                - relevant (bool): Whether the text is relevant
                - explanation (str): Explanation of the relevance judgment
        """
        # Define structured output schema
        class RelevanceSchema(BaseModel):
            relevant: bool = Field(description="Whether the passage is relevant to answer the question")
            explanation: str = Field(description="Brief explanation of the relevance judgment")

        # Load prompt template from file if not provided
        if prompt_template is None:
            prompt_file = DATA_DIR / "prompts" / "binary_relevance_promt.txt"
            try:
                with open(prompt_file, 'r') as f:
                    prompt_template = f.read().strip()
            except FileNotFoundError:
                logger.warning(f"Prompt file not found: {prompt_file}. Using default template.")
                prompt_template = (
                    "Instruction: Indicate if the passage is relevant for the question.\n\n"
                    "Question: {question}\nPassage: {text}"
                )

        # Create structured output LLM
        structured_llm = self.llm.with_structured_output(RelevanceSchema)

        # Create prompt
        if self.use_chat:
            prompt = ChatPromptTemplate.from_template(prompt_template)
        else:
            raise NotImplementedError("Structured output is only implemented for chat models.")

        # Create chain with structured output
        chain = prompt | structured_llm

        # Map 'text' to 'passage' if the prompt uses 'passage'
        input_vars = {"question": question}
        if "{passage}" in prompt_template:
            input_vars["passage"] = text
        else:
            input_vars["text"] = text

        try:
            # Invoke chain and get structured output
            result = chain.invoke(input_vars, **kwargs)

            # Convert Pydantic model to dict (Pydantic v2 uses model_dump)
            if hasattr(result, 'model_dump'):
                return result.model_dump()
            else:
                return dict(result)

        except Exception as e:
            logger.error(f"Error in is_text_relevant: {e}")
            return {
                "relevant": False,
                "explanation": f"Error during evaluation: {str(e)}",
            }

    def is_text_relevant_batch(
        self,
        question: str,
        texts: list[str],
        prompt_template: Optional[str] = None,
        progress_bar: bool = True,
        **kwargs,
    ) -> list[dict]:
        """Determine if multiple texts are relevant to a question (batched processing).

        Makes multiple API calls to evaluate each text independently.

        Args:
            question: The question to evaluate relevance against.
            texts: List of texts/passages to check for relevance.
            prompt_template: Optional custom prompt template passed to is_text_relevant.
            progress_bar: Whether to show a progress bar.
            **kwargs: Additional arguments to pass to is_text_relevant.

        Returns:
            List of dictionaries, one for each text, with keys:
                - relevant (bool): Whether the text is relevant
                - explanation (str): Explanation of the relevance judgment
        """
        results = []

        iterator = texts
        if progress_bar:
            iterator = tqdm(texts, desc="Evaluating relevance")

        for text in iterator:
            result = self.is_text_relevant(
                question=question,
                text=text,
                prompt_template=prompt_template,
                **kwargs
            )
            results.append(result)

        return results

    async def is_text_relevant_async(
        self,
        question: str,
        text: str,
        prompt_template: Optional[str] = None,
        **kwargs,
    ) -> dict:
        """Async version of is_text_relevant for parallel processing.

        Args:
            question: The question to evaluate relevance against.
            text: The text/passage to check for relevance.
            prompt_template: Optional custom prompt template.
            **kwargs: Additional arguments to pass to the LLM.

        Returns:
            Dictionary with keys:
                - relevant (bool): Whether the text is relevant
                - explanation (str): Explanation of the relevance judgment
        """
        # Define structured output schema
        class RelevanceSchema(BaseModel):
            relevant: bool = Field(description="Whether the passage is relevant to answer the question")
            explanation: str = Field(description="Brief explanation of the relevance judgment")

        # Load prompt template from file if not provided
        if prompt_template is None:
            prompt_file = DATA_DIR / "prompts" / "binary_relevance_promt.txt"
            try:
                with open(prompt_file, 'r') as f:
                    prompt_template = f.read().strip()
            except FileNotFoundError:
                logger.warning(f"Prompt file not found: {prompt_file}. Using default template.")
                prompt_template = (
                    "Instruction: Indicate if the passage is relevant for the question.\n\n"
                    "Question: {question}\nPassage: {text}"
                )

        # Create structured output LLM
        structured_llm = self.llm.with_structured_output(RelevanceSchema)

        # Create prompt
        if self.use_chat:
            prompt = ChatPromptTemplate.from_template(prompt_template)
        else:
            raise NotImplementedError("Structured output is only implemented for chat models.")

        # Create chain with structured output
        chain = prompt | structured_llm

        # Map 'text' to 'passage' if the prompt uses 'passage'
        input_vars = {"question": question}
        if "{passage}" in prompt_template:
            input_vars["passage"] = text
        else:
            input_vars["text"] = text

        try:
            # Invoke chain asynchronously
            result = await chain.ainvoke(input_vars, **kwargs)

            # Convert Pydantic model to dict
            if hasattr(result, 'model_dump'):
                return result.model_dump()
            else:
                return dict(result)

        except Exception as e:
            logger.error(f"Error in is_text_relevant_async: {e}")
            return {
                "relevant": False,
                "explanation": f"Error during evaluation: {str(e)}",
            }

    async def is_text_relevant_batch_async(
        self,
        question: str,
        texts: list[str],
        prompt_template: Optional[str] = None,
        progress_bar: bool = True,
        max_concurrent: int = 50,
        **kwargs,
    ) -> list[dict]:
        """Async batched relevance checking with true parallelism.

        Processes all texts concurrently with controlled concurrency for optimal speed.

        Args:
            question: The question to evaluate relevance against.
            texts: List of texts/passages to check for relevance.
            prompt_template: Optional custom prompt template.
            progress_bar: Whether to show a progress bar.
            max_concurrent: Maximum number of concurrent API calls (default: 50).
            **kwargs: Additional arguments to pass to is_text_relevant_async.

        Returns:
            List of dictionaries, one for each text, with keys:
                - relevant (bool): Whether the text is relevant
                - explanation (str): Explanation of the relevance judgment
        """
        if not texts:
            return []

        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_semaphore(text: str) -> dict:
            async with semaphore:
                return await self.is_text_relevant_async(
                    question=question,
                    text=text,
                    prompt_template=prompt_template,
                    **kwargs
                )

        # Create all tasks
        tasks = [process_with_semaphore(text) for text in texts]

        # Execute with progress bar if requested
        if progress_bar:
            results = []
            for coro in atqdm.as_completed(tasks, desc="Evaluating relevance", total=len(tasks)):
                result = await coro
                results.append(result)
            # Reorder results to match input order
            # Note: as_completed doesn't preserve order, so we need to track indices
            # For simplicity, let's use gather which preserves order
            results = await atqdm.gather(*tasks, desc="Evaluating relevance")
        else:
            results = await asyncio.gather(*tasks)

        return results

    def expand_dataframe(
        self,
        df: pd.DataFrame,
        expansion_prompt: str | Callable[[pd.Series], str],
        batch_size: int = 10,
        output_column: str = "expanded",
        progress_bar: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """Expand a dataframe by generating additional content for each row using the LLM.
        
        Args:
            df: Input dataframe to expand.
            expansion_prompt: Either a string template with {column_name} placeholders,
                            or a callable that takes a pd.Series (row) and returns a prompt string.
            batch_size: Number of rows to process in each batch.
            output_column: Name of the column to store the expanded content.
            progress_bar: Whether to show a progress bar.
            **kwargs: Additional arguments to pass to the LLM invoke method.
            
        Returns:
            A new dataframe with the expanded content in the specified column.
            
        Example:
            >>> service = OpenAIService()
            >>> df = pd.DataFrame({"question": ["What is AI?", "What is ML?"]})
            >>> expanded = service.expand_dataframe(
            ...     df,
            ...     expansion_prompt="Generate a detailed explanation for: {question}",
            ...     output_column="explanation"
            ... )
        """
        result_df = df.copy()
        result_df[output_column] = None

        # Determine if expansion_prompt is a template string or a callable
        is_template = isinstance(expansion_prompt, str)
        
        # Process in batches
        total_batches = (len(df) + batch_size - 1) // batch_size
        iterator = range(0, len(df), batch_size)
        
        if progress_bar:
            iterator = tqdm(iterator, desc="Expanding dataframe", total=total_batches)

        for batch_start in iterator:
            batch_end = min(batch_start + batch_size, len(df))
            batch_df = df.iloc[batch_start:batch_end]

            for idx, row in batch_df.iterrows():
                try:
                    # Generate prompt for this row
                    if is_template:
                        # Replace template variables with row values
                        prompt = expansion_prompt
                        for col in df.columns:
                            prompt = prompt.replace(f"{{{col}}}", str(row[col]))
                    else:
                        # Use callable to generate prompt
                        prompt = expansion_prompt(row)

                    # Invoke LLM
                    expanded_content = self.invoke(prompt, **kwargs)
                    result_df.at[idx, output_column] = expanded_content

                except Exception as e:
                    logger.warning(f"Error expanding row {idx}: {e}")
                    result_df.at[idx, output_column] = None

        return result_df

    def expand_dataframe_batch(
        self,
        df: pd.DataFrame,
        expansion_prompt: str | Callable[[pd.DataFrame], str],
        batch_size: int = 10,
        output_column: str = "expanded",
        progress_bar: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """Expand a dataframe by processing entire batches at once (more efficient for structured outputs).
        
        Args:
            df: Input dataframe to expand.
            expansion_prompt: Either a string template or a callable that takes a pd.DataFrame (batch)
                            and returns a prompt string for the entire batch.
            batch_size: Number of rows to process in each batch.
            output_column: Name of the column to store the expanded content.
            progress_bar: Whether to show a progress bar.
            **kwargs: Additional arguments to pass to the LLM invoke method.
            
        Returns:
            A new dataframe with the expanded content in the specified column.
            
        Example:
            >>> service = OpenAIService()
            >>> df = pd.DataFrame({"question": ["What is AI?", "What is ML?"]})
            >>> expanded = service.expand_dataframe_batch(
            ...     df,
            ...     expansion_prompt=lambda batch: f"Generate explanations for: {batch['question'].tolist()}",
            ...     output_column="explanations"
            ... )
        """
        result_df = df.copy()
        result_df[output_column] = None

        # Process in batches
        total_batches = (len(df) + batch_size - 1) // batch_size
        iterator = range(0, len(df), batch_size)
        
        if progress_bar:
            iterator = tqdm(iterator, desc="Expanding dataframe (batch mode)", total=total_batches)

        for batch_start in iterator:
            batch_end = min(batch_start + batch_size, len(df))
            batch_df = df.iloc[batch_start:batch_end]

            try:
                # Generate prompt for this batch
                if isinstance(expansion_prompt, str):
                    # If it's a template, we'll need to format it with batch data
                    prompt = expansion_prompt
                    # Replace with batch representation
                    for col in df.columns:
                        values = batch_df[col].tolist()
                        prompt = prompt.replace(f"{{{col}}}", str(values))
                else:
                    # Use callable to generate prompt for entire batch
                    prompt = expansion_prompt(batch_df)

                # Invoke LLM for the batch
                expanded_content = self.invoke(prompt, **kwargs)
                
                # For batch mode, we assume the LLM returns structured output
                # that can be split or parsed. For simplicity, we'll assign
                # the same content to all rows in the batch, but you can customize
                # this to parse the response into individual values.
                # This is a design choice - you may want to modify this based on
                # your specific use case.
                for idx in batch_df.index:
                    result_df.at[idx, output_column] = expanded_content

            except Exception as e:
                logger.warning(f"Error expanding batch {batch_start}-{batch_end}: {e}")
                for idx in batch_df.index:
                    result_df.at[idx, output_column] = None

        return result_df


def main():
    """Test function to verify is_text_relevant with structured output."""
    # Initialize the service
    print("Initializing OpenAI service...")
    service = OpenAIService(model_name="gpt-3.5-turbo", temperature=0.0)

    # Test case 1: Relevant passage
    question1 = "What is the capital of France?"
    passage1 = "Paris is the capital and most populous city of France. Located in northern France, Paris has been one of Europe's major centers of finance, diplomacy, commerce, fashion, science and arts."

    print("\n" + "="*80)
    print("Test Case 1: Relevant passage")
    print("="*80)
    print(f"Question: {question1}")
    print(f"Passage: {passage1}")
    print("\nEvaluating relevance...")

    result1 = service.is_text_relevant(question1, passage1)
    print(f"\nResult:")
    print(f"  Relevant: {result1['relevant']}")
    print(f"  Explanation: {result1['explanation']}")

    # Test case 2: Irrelevant passage
    question2 = "What is the capital of France?"
    passage2 = "The Amazon rainforest is the largest tropical rainforest in the world, covering much of northwestern Brazil and extending into Colombia, Peru and other South American countries."

    print("\n" + "="*80)
    print("Test Case 2: Irrelevant passage")
    print("="*80)
    print(f"Question: {question2}")
    print(f"Passage: {passage2}")
    print("\nEvaluating relevance...")

    result2 = service.is_text_relevant(question2, passage2)
    print(f"\nResult:")
    print(f"  Relevant: {result2['relevant']}")
    print(f"  Explanation: {result2['explanation']}")

    # Test case 3: Partially relevant passage
    question3 = "When was the Eiffel Tower built?"
    passage3 = "Paris is home to many famous landmarks including the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral. The city attracts millions of tourists every year."

    print("\n" + "="*80)
    print("Test Case 3: Partially relevant passage")
    print("="*80)
    print(f"Question: {question3}")
    print(f"Passage: {passage3}")
    print("\nEvaluating relevance...")

    result3 = service.is_text_relevant(question3, passage3)
    print(f"\nResult:")
    print(f"  Relevant: {result3['relevant']}")
    print(f"  Explanation: {result3['explanation']}")

    print("\n" + "="*80)
    print("All tests completed!")
    print("="*80)


if __name__ == "__main__":
    main()
