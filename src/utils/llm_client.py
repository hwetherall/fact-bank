"""
LLM Client abstraction for OpenRouter and Groq.

Provides a unified interface for making LLM calls across different providers.
"""

import os
import json
import asyncio
import logging
from enum import Enum
from typing import Any
from dataclasses import dataclass

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENROUTER = "openrouter"
    GROQ = "groq"


@dataclass
class LLMResponse:
    """Response from an LLM call."""
    content: str
    model: str
    provider: LLMProvider
    usage: dict[str, int] | None = None
    raw_response: dict | None = None


class LLMClientError(Exception):
    """Base exception for LLM client errors."""
    pass


class LLMClient:
    """
    Unified LLM client supporting OpenRouter and Groq.
    
    Usage:
        client = LLMClient(provider=LLMProvider.OPENROUTER)
        response = await client.chat_completion(
            messages=[{"role": "user", "content": "Hello"}],
            model="anthropic/claude-3.5-sonnet"
        )
    """
    
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    GROQ_BASE_URL = "https://api.groq.com/openai/v1"
    
    # Default models for each provider
    DEFAULT_MODELS = {
        LLMProvider.OPENROUTER: "anthropic/claude-3.5-sonnet",
        LLMProvider.GROQ: "openai/gpt-oss-120b",
    }
    
    def __init__(
        self,
        provider: LLMProvider | str = LLMProvider.OPENROUTER,
        api_key: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        """
        Initialize the LLM client.
        
        Args:
            provider: The LLM provider to use (openrouter or groq)
            api_key: API key (defaults to environment variable)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        if isinstance(provider, str):
            provider = LLMProvider(provider.lower())
        
        self.provider = provider
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Set up API key and base URL based on provider
        if provider == LLMProvider.OPENROUTER:
            self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
            self.base_url = self.OPENROUTER_BASE_URL
        else:
            self.api_key = api_key or os.getenv("GROQ_API_KEY")
            self.base_url = self.GROQ_BASE_URL
        
        if not self.api_key:
            raise LLMClientError(
                f"No API key provided for {provider.value}. "
                f"Set {provider.value.upper()}_API_KEY environment variable."
            )
        
        self._client: httpx.AsyncClient | None = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                headers=self._get_headers(),
            )
        return self._client
    
    def _get_headers(self) -> dict[str, str]:
        """Get headers for API requests."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        if self.provider == LLMProvider.OPENROUTER:
            headers["HTTP-Referer"] = "https://factor.innovera.ai"
            headers["X-Title"] = "Factor Fact Bank"
        
        return headers
    
    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> LLMResponse:
        """
        Make a chat completion request.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model identifier (uses default if not specified)
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens in response
            response_format: Optional response format (e.g., {"type": "json_object"})
        
        Returns:
            LLMResponse with the generated content
        """
        model = model or self.DEFAULT_MODELS[self.provider]
        
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
        
        if response_format:
            payload["response_format"] = response_format
        
        return await self._make_request("/chat/completions", payload)
    
    async def _make_request(
        self,
        endpoint: str,
        payload: dict[str, Any],
    ) -> LLMResponse:
        """Make an API request with retry logic."""
        client = await self._get_client()
        url = f"{self.base_url}{endpoint}"
        
        last_error: Exception | None = None
        
        for attempt in range(self.max_retries):
            try:
                response = await client.post(url, json=payload)
                
                if response.status_code == 429:
                    # Rate limited - wait and retry
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limited, waiting {wait_time}s before retry")
                    await asyncio.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                data = response.json()
                
                return LLMResponse(
                    content=data["choices"][0]["message"]["content"],
                    model=data.get("model", payload.get("model", "unknown")),
                    provider=self.provider,
                    usage=data.get("usage"),
                    raw_response=data,
                )
                
            except httpx.HTTPStatusError as e:
                last_error = e
                # Try to extract detailed error message from response
                error_detail = ""
                try:
                    error_body = e.response.json()
                    if "error" in error_body:
                        error_info = error_body["error"]
                        if isinstance(error_info, dict):
                            error_detail = error_info.get("message", str(error_info))
                        else:
                            error_detail = str(error_info)
                except Exception:
                    error_detail = e.response.text[:500] if e.response.text else ""
                
                logger.error(f"HTTP error on attempt {attempt + 1}: {e}. Detail: {error_detail}")
                
                if e.response.status_code >= 500:
                    # Server error - retry with backoff
                    await asyncio.sleep(2 ** attempt)
                    continue
                
                # For 4xx errors, include the detail in the error message
                error_msg = f"API error: {e}"
                if error_detail:
                    error_msg += f" - {error_detail}"
                raise LLMClientError(error_msg) from e
                
            except httpx.RequestError as e:
                last_error = e
                logger.error(f"Request error on attempt {attempt + 1}: {e}")
                await asyncio.sleep(2 ** attempt)
                continue
        
        raise LLMClientError(
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


# Synchronous wrapper for convenience
class SyncLLMClient:
    """
    Synchronous wrapper for LLMClient.
    
    Usage:
        client = SyncLLMClient(provider="openrouter")
        response = client.chat_completion(
            messages=[{"role": "user", "content": "Hello"}]
        )
    """
    
    def __init__(self, **kwargs):
        self._async_client = LLMClient(**kwargs)
        self._loop: asyncio.AbstractEventLoop | None = None
    
    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create an event loop."""
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            if self._loop is None or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
            return self._loop
    
    def chat_completion(self, **kwargs) -> LLMResponse:
        """Synchronous chat completion."""
        loop = self._get_loop()
        return loop.run_until_complete(
            self._async_client.chat_completion(**kwargs)
        )
    
    def close(self):
        """Close the client."""
        if self._loop and not self._loop.is_closed():
            self._loop.run_until_complete(self._async_client.close())
            self._loop.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def get_default_client(
    provider: str | None = None,
    **kwargs
) -> LLMClient:
    """
    Get a default LLM client based on environment configuration.
    
    Args:
        provider: Override the default provider
        **kwargs: Additional arguments for LLMClient
    
    Returns:
        Configured LLMClient instance
    """
    if provider is None:
        provider = os.getenv("DEFAULT_LLM_PROVIDER", "openrouter")
    
    return LLMClient(provider=LLMProvider(provider.lower()), **kwargs)

