"""LLM service subpackage."""

from __future__ import annotations

from src.llm.base import LLMBase

# ModalLLMService is imported lazily to avoid requiring `modal` at import time
# when only OpenAIService is needed.  Import explicitly when required:
#   from src.llm.ModalLLM import ModalLLMService

__all__ = ["LLMBase"]
