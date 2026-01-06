"""
Embedding client and deduplication utilities.

Uses OpenRouter for embeddings and provides deduplication logic for facts.
"""

import os
import asyncio
import logging
from typing import TYPE_CHECKING

import httpx
import numpy as np
from dotenv import load_dotenv

if TYPE_CHECKING:
    from src.storage.models import Fact

load_dotenv()

logger = logging.getLogger(__name__)


class EmbeddingClientError(Exception):
    """Base exception for embedding client errors."""
    pass


class EmbeddingClient:
    """
    Client for generating embeddings via OpenRouter.
    
    Usage:
        client = EmbeddingClient()
        embeddings = await client.get_embeddings(["text1", "text2"])
    """
    
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    DEFAULT_MODEL = "openai/text-embedding-3-small"
    
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        batch_size: int = 100,
    ):
        """
        Initialize the embedding client.
        
        Args:
            api_key: OpenRouter API key (defaults to environment variable)
            model: Embedding model to use
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            batch_size: Maximum texts per API call
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise EmbeddingClientError(
                "No API key provided. Set OPENROUTER_API_KEY environment variable."
            )
        
        self.model = model or os.getenv("DEFAULT_EMBEDDING_MODEL", self.DEFAULT_MODEL)
        self.timeout = timeout
        self.max_retries = max_retries
        self.batch_size = batch_size
        self._client: httpx.AsyncClient | None = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://factor.innovera.ai",
                    "X-Title": "Factor Fact Bank",
                },
            )
        return self._client
    
    async def get_embedding(self, text: str) -> list[float]:
        """
        Get embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        embeddings = await self.get_embeddings([text])
        return embeddings[0]
    
    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Get embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Process in batches
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = await self._get_embeddings_batch(batch)
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    async def _get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for a batch of texts."""
        client = await self._get_client()
        url = f"{self.OPENROUTER_BASE_URL}/embeddings"
        
        payload = {
            "model": self.model,
            "input": texts,
        }
        
        last_error: Exception | None = None
        
        for attempt in range(self.max_retries):
            try:
                response = await client.post(url, json=payload)
                
                if response.status_code == 429:
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limited, waiting {wait_time}s before retry")
                    await asyncio.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                data = response.json()
                
                # Sort by index to ensure correct order
                embeddings_data = sorted(data["data"], key=lambda x: x["index"])
                return [item["embedding"] for item in embeddings_data]
                
            except httpx.HTTPStatusError as e:
                last_error = e
                logger.error(f"HTTP error on attempt {attempt + 1}: {e}")
                if e.response.status_code >= 500:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise EmbeddingClientError(f"API error: {e}") from e
                
            except httpx.RequestError as e:
                last_error = e
                logger.error(f"Request error on attempt {attempt + 1}: {e}")
                await asyncio.sleep(2 ** attempt)
                continue
        
        raise EmbeddingClientError(
            f"Failed after {self.max_retries} attempts: {last_error}"
        )
    
    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity score (0.0 to 1.0)
    """
    a = np.array(vec1)
    b = np.array(vec2)
    
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return float(dot_product / (norm_a * norm_b))


def cosine_similarity_matrix(embeddings: list[list[float]]) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix.
    
    Args:
        embeddings: List of embedding vectors
        
    Returns:
        NxN similarity matrix
    """
    matrix = np.array(embeddings)
    
    # Normalize rows
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    normalized = matrix / norms
    
    # Compute similarity matrix
    return np.dot(normalized, normalized.T)


def find_duplicate_pairs(
    embeddings: list[list[float]],
    threshold: float = 0.85,
) -> list[tuple[int, int, float]]:
    """
    Find pairs of indices with similarity above threshold.
    
    Args:
        embeddings: List of embedding vectors
        threshold: Similarity threshold for duplicates
        
    Returns:
        List of (idx1, idx2, similarity) tuples
    """
    if len(embeddings) < 2:
        return []
    
    sim_matrix = cosine_similarity_matrix(embeddings)
    n = len(embeddings)
    
    duplicates = []
    for i in range(n):
        for j in range(i + 1, n):
            similarity = sim_matrix[i, j]
            if similarity >= threshold:
                duplicates.append((i, j, float(similarity)))
    
    return duplicates


async def deduplicate_facts(
    facts: list["Fact"],
    embedding_client: EmbeddingClient | None = None,
    threshold: float = 0.85,
) -> list["Fact"]:
    """
    Deduplicate facts based on embedding similarity.
    
    Merges duplicate facts by:
    - Keeping the more detailed content
    - Combining source documents
    - Averaging confidence scores
    - Keeping the highest importance rating
    
    Args:
        facts: List of facts to deduplicate
        embedding_client: Optional pre-configured client
        threshold: Similarity threshold for duplicates
        
    Returns:
        Deduplicated list of facts
    """
    from src.storage.models import Fact, RelevanceLevel, Believability
    
    if len(facts) < 2:
        return facts
    
    # Get or create embedding client
    close_client = False
    if embedding_client is None:
        embedding_client = EmbeddingClient()
        close_client = True
    
    try:
        # Get embeddings for facts that don't have them
        facts_needing_embeddings = [
            (i, f) for i, f in enumerate(facts) if f.embedding is None
        ]
        
        if facts_needing_embeddings:
            indices, facts_to_embed = zip(*facts_needing_embeddings)
            texts = [f.content for f in facts_to_embed]
            new_embeddings = await embedding_client.get_embeddings(texts)
            
            for idx, embedding in zip(indices, new_embeddings):
                facts[idx].embedding = embedding
        
        # Get all embeddings
        embeddings = [f.embedding for f in facts]
        
        # Find duplicates
        duplicate_pairs = find_duplicate_pairs(embeddings, threshold)
        
        if not duplicate_pairs:
            return facts
        
        # Build union-find structure for merging groups
        parent = list(range(len(facts)))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        for i, j, _ in duplicate_pairs:
            union(i, j)
        
        # Group facts by their root
        groups: dict[int, list[int]] = {}
        for i in range(len(facts)):
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(i)
        
        # Merge each group
        result = []
        
        # Priority ordering for relevance (higher = more important)
        relevance_order = {
            RelevanceLevel.CRITICAL.value: 4,
            RelevanceLevel.CHAPTER_SPECIFIC.value: 3,
            RelevanceLevel.ADDITIONAL_CONTEXT.value: 2,
            RelevanceLevel.NOISE.value: 1,
        }
        
        # Priority ordering for believability (higher = more believable)
        believability_order = {
            Believability.VERIFIED.value: 3,
            Believability.NEEDS_VERIFICATION.value: 2,
            Believability.OPINION.value: 1,
        }
        
        for indices in groups.values():
            group_facts = [facts[i] for i in indices]
            
            if len(group_facts) == 1:
                result.append(group_facts[0])
                continue
            
            # Keep the most detailed content (longest)
            best_fact = max(group_facts, key=lambda f: len(f.content))
            
            # Combine source documents
            all_sources = []
            for f in group_facts:
                all_sources.extend(f.source_documents)
            unique_sources = list(dict.fromkeys(all_sources))  # Preserve order
            
            # Keep best believability (most believable)
            best_believability = max(
                group_facts,
                key=lambda f: believability_order.get(f.believability, 0)
            ).believability
            
            # Keep highest relevance
            best_relevance = max(
                group_facts,
                key=lambda f: relevance_order.get(f.relevance, 0)
            ).relevance
            
            # Merge chapter relevance (take max for each chapter)
            merged_relevance = {}
            for f in group_facts:
                for chapter, score in f.chapter_relevance.items():
                    if chapter not in merged_relevance:
                        merged_relevance[chapter] = score
                    else:
                        merged_relevance[chapter] = max(merged_relevance[chapter], score)
            
            # Create merged fact
            merged = Fact(
                id=best_fact.id,
                content=best_fact.content,
                source_quote=best_fact.source_quote,
                source_documents=unique_sources,
                source_type=best_fact.source_type,
                believability=best_believability,
                relevance=best_relevance,
                chapter_relevance=merged_relevance,
                extraction_timestamp=best_fact.extraction_timestamp,
                usage_count=0,
                used_in_chapters=[],
                embedding=best_fact.embedding,
            )
            result.append(merged)
        
        logger.info(
            f"Deduplication: {len(facts)} facts -> {len(result)} unique "
            f"({len(facts) - len(result)} duplicates merged)"
        )
        
        return result
        
    finally:
        if close_client:
            await embedding_client.close()


# Synchronous wrapper
class SyncEmbeddingClient:
    """Synchronous wrapper for EmbeddingClient."""
    
    def __init__(self, **kwargs):
        self._async_client = EmbeddingClient(**kwargs)
        self._loop: asyncio.AbstractEventLoop | None = None
    
    def _get_loop(self) -> asyncio.AbstractEventLoop:
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            if self._loop is None or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
            return self._loop
    
    def get_embedding(self, text: str) -> list[float]:
        loop = self._get_loop()
        return loop.run_until_complete(self._async_client.get_embedding(text))
    
    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        loop = self._get_loop()
        return loop.run_until_complete(self._async_client.get_embeddings(texts))
    
    def close(self):
        if self._loop and not self._loop.is_closed():
            self._loop.run_until_complete(self._async_client.close())
            self._loop.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def deduplicate_facts_sync(
    facts: list["Fact"],
    threshold: float = 0.85,
) -> list["Fact"]:
    """Synchronous wrapper for deduplicate_facts."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            deduplicate_facts(facts, threshold=threshold)
        )
    finally:
        loop.close()

