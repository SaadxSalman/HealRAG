"""Application configuration loaded from environment variables / .env."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = two levels up from this file (app/config.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

# Load .env into the process environment if it exists (never fail if missing).
if ENV_PATH.exists():
    load_dotenv(ENV_PATH, override=False)


class Settings(BaseSettings):
    """Central settings object. Values come from .env / environment."""

    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH) if ENV_PATH.exists() else None,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- General ----
    app_name: str = "HealRAG"
    app_env: str = "development"
    debug: bool = True
    log_level: str = "INFO"

    # ---- Ollama ----
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "qwen3:8b"
    ollama_generation_model: str = "qwen3:8b"
    ollama_embed_model: str = "nomic-embed-text"
    generation_temperature: float = 0.4
    grader_temperature: float = 0.0
    rewriter_temperature: float = 0.2
    ollama_timeout: int = 120

    # ---- Embeddings ----
    embedding_backend: str = "ollama"  # ollama | sentence_transformers
    embedding_dimension: int = 768
    sentence_transformer_model: str = "all-MiniLM-L6-v2"

    # ---- Vector store ----
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "healrag_documents"

    # ---- Retrieval & grading ----
    retrieval_k: int = 6
    bm25_k: int = 4
    relevance_threshold: float = 0.5
    max_query_rewrites: int = 2
    max_hallucination_retries: int = 2

    # ---- Ingestion ----
    chunk_size: int = 800
    chunk_overlap: int = 120
    max_file_size_mb: int = 20

    # ---- API / web ----
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    api_bearer_token: Optional[str] = None
    cors_origins: List[str] = Field(
        default_factory=lambda: ["http://localhost:8501", "http://localhost:3000"]
    )
    streamlit_port: int = 8501
    streamlit_api_url: str = "http://localhost:8000"

    # ---- Observability ----
    trace_dir: str = "./traces"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_cors(cls, v):
        if isinstance(v, str):
            # Accept both JSON-ish lists and comma separated strings.
            cleaned = v.strip()
            if cleaned.startswith("["):
                return [x.strip().strip('"').strip("'") for x in cleaned[1:-1].split(",")]
            return [x.strip() for x in cleaned.split(",") if x.strip()]
        return v

    @property
    def chroma_path(self) -> Path:
        p = Path(self.chroma_persist_dir)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def trace_path(self) -> Path:
        p = Path(self.trace_dir)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def data_dir(self) -> Path:
        return PROJECT_ROOT / "data" / "documents"


def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    if not hasattr(get_settings, "_cached"):
        get_settings._cached = Settings()
    return get_settings._cached


settings = get_settings()
