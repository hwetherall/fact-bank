"""
Configuration management for Factor.

Centralizes environment variable loading and application settings.
"""

import os
import logging
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Application configuration."""
    
    # API Keys
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    
    # Database
    DATABASE_PATH: str = os.getenv("DATABASE_PATH", "data/factor.db")
    
    # LLM Settings
    DEFAULT_LLM_PROVIDER: Literal["openrouter", "groq"] = os.getenv(
        "DEFAULT_LLM_PROVIDER", "openrouter"
    )  # type: ignore
    DEFAULT_EXTRACTION_MODEL: str = os.getenv(
        "DEFAULT_EXTRACTION_MODEL", "anthropic/claude-3.5-sonnet"
    )
    DEFAULT_EMBEDDING_MODEL: str = os.getenv(
        "DEFAULT_EMBEDDING_MODEL", "openai/text-embedding-3-small"
    )
    
    # Extraction Settings
    DEDUP_SIMILARITY_THRESHOLD: float = float(os.getenv("DEDUP_SIMILARITY_THRESHOLD", "0.85"))
    MAX_FACTS_PER_DOCUMENT: int = int(os.getenv("MAX_FACTS_PER_DOCUMENT", "500"))
    # Chunk size for document processing (in characters, ~4 chars = 1 token)
    CHUNK_SIZE_CHARS: int = int(os.getenv("CHUNK_SIZE_CHARS", "40000"))
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @classmethod
    def validate(cls) -> list[str]:
        """
        Validate configuration and return any issues.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        issues = []
        
        if not cls.OPENROUTER_API_KEY:
            issues.append("OPENROUTER_API_KEY is not set")
        
        return issues
    
    @classmethod
    def is_valid(cls) -> bool:
        """Check if configuration is valid for operation."""
        return len(cls.validate()) == 0
    
    @classmethod
    def setup_logging(cls, level: str | None = None):
        """
        Configure application logging.
        
        Args:
            level: Optional override for log level
        """
        log_level = level or cls.LOG_LEVEL
        
        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, log_level.upper(), logging.INFO),
            format=cls.LOG_FORMAT,
        )
        
        # Reduce noise from third-party libraries
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)


# Singleton config instance
config = Config()


def get_config() -> Config:
    """Get the configuration instance."""
    return config

