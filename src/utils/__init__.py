"""Utility modules."""

from .llm_client import LLMClient, LLMProvider
from .embeddings import EmbeddingClient, deduplicate_facts

__all__ = [
    "LLMClient",
    "LLMProvider",
    "EmbeddingClient",
    "deduplicate_facts",
]

