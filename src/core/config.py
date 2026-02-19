"""
Application configuration and settings management.
Uses Pydantic Settings for type-safe environment variable handling.

Provider Priority:
  - LLM: OpenRouter (primary) → Groq (fallback)
  - Embeddings: OpenRouter (primary) → NVIDIA NIM (fallback)
  - Reranking: NVIDIA NIM (only provider)
"""

from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Application
    app_env: str = Field(default="development", alias="APP_ENV")
    debug: bool = Field(default=False, alias="DEBUG")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # OpenRouter Configuration (Primary LLM + Embeddings)
    openrouter_api_key: Optional[str] = Field(default=None, alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1", alias="OPENROUTER_BASE_URL"
    )
    openrouter_model: str = Field(
        default="meta-llama/llama-3.3-70b-instruct", alias="OPENROUTER_MODEL"
    )

    # OpenRouter Embedding Configuration
    openrouter_embedding_model: str = Field(
        default="qwen/qwen3-embedding-8b", alias="OPENROUTER_EMBEDDING_MODEL"
    )
    openrouter_embedding_dimension: int = Field(
        default=4096, alias="OPENROUTER_EMBEDDING_DIMENSION"
    )

    # NVIDIA NIM Configuration (Fallback Embeddings + Reranking)
    nvidia_api_key: Optional[str] = Field(default=None, alias="NVIDIA_API_KEY")
    nvidia_embedding_api_key: Optional[str] = Field(
        default=None, alias="NVIDIA_EMBEDDING_API_KEY"
    )
    nvidia_rerank_api_key: Optional[str] = Field(
        default=None, alias="NVIDIA_RERANK_API_KEY"
    )
    nvidia_base_url: str = Field(
        default="https://integrate.api.nvidia.com/v1", alias="NVIDIA_BASE_URL"
    )
    nvidia_model: str = Field(default="meta-llama/llama-3.3-70b-instruct", alias="NVIDIA_MODEL")
    nvidia_embedding_model: str = Field(
        default="nvidia/llama-3.2-nv-embedqa-1b-v2", alias="NVIDIA_EMBEDDING_MODEL"
    )
    nvidia_embedding_dimension: int = Field(
        default=2048, alias="NVIDIA_EMBEDDING_DIMENSION"
    )

    # NVIDIA Reranker Configuration (Primary)
    rerank_model: str = Field(
        default="nvidia/llama-3.2-nv-rerankqa-1b-v2", alias="RERANK_MODEL"
    )
    nvidia_rerank_base_url: str = Field(
        default="https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-3_2-nv-rerankqa-1b-v2/reranking",
        alias="NVIDIA_RERANK_BASE_URL",
    )

    # Groq Configuration (Fallback LLM)
    groq_api_key: Optional[str] = Field(default=None, alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_MODEL")

    # Active Embedding Configuration (computed from available keys)
    @property
    def embedding_model(self) -> str:
        """Return active embedding model based on available API keys."""
        if self.openrouter_api_key:
            return self.openrouter_embedding_model
        return self.nvidia_embedding_model

    @property
    def embedding_dimension(self) -> int:
        """Return active embedding dimension based on provider."""
        if self.openrouter_api_key:
            return self.openrouter_embedding_dimension
        return self.nvidia_embedding_dimension

    # Embedding Batch Settings
    embedding_batch_size: int = Field(default=50, alias="EMBEDDING_BATCH_SIZE")

    # Vector Database Paths
    data_dir: Path = Field(default=Path("./data"), alias="DATA_DIR")
    faiss_index_path: Path = Field(
        default=Path("./data/faiss_index.bin"), alias="FAISS_INDEX_PATH"
    )
    bm25_index_path: Path = Field(
        default=Path("./data/bm25_index.pkl"), alias="BM25_INDEX_PATH"
    )
    sqlite_db_path: Path = Field(
        default=Path("./data/metadata.db"), alias="SQLITE_DB_PATH"
    )

    # Document Processing
    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")
    max_file_size_mb: int = Field(default=50, alias="MAX_FILE_SIZE_MB")

    # Memory Settings
    memory_decision_temperature: float = Field(
        default=0.3, alias="MEMORY_DECISION_TEMPERATURE"
    )
    memory_confidence_threshold: float = Field(
        default=0.7, alias="MEMORY_CONFIDENCE_THRESHOLD"
    )

    # Sandbox Configuration
    sandbox_timeout: int = Field(default=30, alias="SANDBOX_TIMEOUT")
    sandbox_max_memory_mb: int = Field(default=512, alias="SANDBOX_MAX_MEMORY_MB")
    sandbox_max_output_length: int = Field(
        default=10000, alias="SANDBOX_MAX_OUTPUT_LENGTH"
    )

    # Open-Meteo
    open_meteo_base_url: str = Field(
        default="https://api.open-meteo.com/v1", alias="OPEN_METEO_BASE_URL"
    )

    # Rate Limiting (NVIDIA NIM fallback only - OpenRouter is credit-based)
    api_requests_per_minute: int = Field(default=40, alias="API_REQUESTS_PER_MINUTE")
    api_batch_delay_seconds: float = Field(default=1.5, alias="API_BATCH_DELAY_SECONDS")

    # Security
    allowed_file_types: List[str] = Field(
        default=["pdf", "txt", "md", "html"], alias="ALLOWED_FILE_TYPES"
    )
    max_upload_size: int = Field(default=52428800, alias="MAX_UPLOAD_SIZE")

    # Reranking Configuration
    rerank_top_k: int = Field(default=5, alias="RERANK_TOP_K")
    rerank_candidates: int = Field(default=20, alias="RERANK_CANDIDATES")

    # Feature Flags
    enable_hybrid_search: bool = Field(default=True, alias="ENABLE_HYBRID_SEARCH")
    enable_citations: bool = Field(default=True, alias="ENABLE_CITATIONS")
    enable_memory: bool = Field(default=True, alias="ENABLE_MEMORY")
    enable_sandbox: bool = Field(default=True, alias="ENABLE_SANDBOX")
    enable_reranking: bool = Field(default=True, alias="ENABLE_RERANKING")

    # Provider Selection
    embedding_provider: str = Field(
        default="openrouter", alias="EMBEDDING_PROVIDER"
    )  # "openrouter" or "nvidia"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.faiss_index_path.parent.mkdir(parents=True, exist_ok=True)
        self.sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
