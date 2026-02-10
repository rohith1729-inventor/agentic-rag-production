"""
Unified configuration for the Universal RAG Core.

Single entry point for all configurable values.
Defaults reproduce current pipeline behavior with zero config changes.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class RAGConfig(BaseSettings):
    """RAG system configuration loaded from environment variables."""

    # Embedding model
    model_path: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="MODEL_PATH",
    )

    # Qdrant
    qdrant_host: str = Field(default="localhost", alias="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, alias="QDRANT_PORT")
    collection_name: str = Field(default="knowledge_base", alias="COLLECTION_NAME")

    # Groq LLM
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.1-8b-instant", alias="GROQ_MODEL")

    # Ingestion
    chunk_size: int = Field(default=500, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, alias="CHUNK_OVERLAP")
    min_chunk_size: int = Field(default=100, alias="MIN_CHUNK_SIZE")
    default_data_dir: str = Field(default="data", alias="DEFAULT_DATA_DIR")

    # Prompts
    prompts_dir: str = Field(default="prompts", alias="PROMPTS_DIR")

    # Retrieval defaults
    default_top_k: int = Field(default=20, alias="DEFAULT_TOP_K")
    default_top_n: int = Field(default=5, alias="DEFAULT_TOP_N")
    default_token_budget: int = Field(default=6000, alias="DEFAULT_TOKEN_BUDGET")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "populate_by_name": True,
    }

    def prompts_path(self) -> Path:
        """Resolve the prompts directory relative to the project root."""
        p = Path(self.prompts_dir)
        if p.is_absolute():
            return p
        # Relative to the project root (parent of rag_core/)
        return Path(__file__).parent.parent / p


# Singleton instance — import this everywhere
_config: Optional[RAGConfig] = None


def get_config() -> RAGConfig:
    """Return the singleton RAGConfig instance."""
    global _config
    if _config is None:
        _config = RAGConfig()
    return _config
