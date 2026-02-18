"""
Application configuration and settings management.
Uses Pydantic Settings for type-safe environment variable handling.
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

    # NVIDIA NIM Configuration
    nvidia_api_key: str = Field(alias="NVIDIA_API_KEY")
    nvidia_embedding_api_key: Optional[str] = Field(
        default=None, alias="NVIDIA_EMBEDDING_API_KEY"
    )
    nvidia_base_url: str = Field(
        default="https://integrate.api.nvidia.com/v1", alias="NVIDIA_BASE_URL"
    )
    nvidia_model: str = Field(
        default="moonshotai/kimi-k2-thinking", alias="NVIDIA_MODEL"
    )

    # Groq Configuration
    groq_api_key: Optional[str] = Field(default=None, alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.1-70b-versatile", alias="GROQ_MODEL")

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

    # Security
    allowed_file_types: List[str] = Field(
        default=["pdf", "txt", "md", "html"], alias="ALLOWED_FILE_TYPES"
    )
    max_upload_size: int = Field(default=52428800, alias="MAX_UPLOAD_SIZE")

    # Feature Flags
    enable_hybrid_search: bool = Field(default=True, alias="ENABLE_HYBRID_SEARCH")
    enable_citations: bool = Field(default=True, alias="ENABLE_CITATIONS")
    enable_memory: bool = Field(default=True, alias="ENABLE_MEMORY")
    enable_sandbox: bool = Field(default=True, alias="ENABLE_SANDBOX")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure data directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.faiss_index_path.parent.mkdir(parents=True, exist_ok=True)
        self.sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
