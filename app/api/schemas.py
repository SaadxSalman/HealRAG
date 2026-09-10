"""Pydantic request/response schemas for the API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    app: str = "HealRAG"
    version: str = "1.0.0"
    ollama_connected: bool = False
    ollama_models: List[str] = Field(default_factory=list)
    documents_indexed: int = 0


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="The user's question")
    stream: bool = False


class AskResponse(BaseModel):
    query: str
    answer: str
    steps: List[str] = Field(default_factory=list)
    corrections: List[str] = Field(default_factory=list)
    rewritten_queries: List[str] = Field(default_factory=list)
    confidence: Optional[float] = None
    hallucination_grade: Optional[float] = None
    retries: int = 0
    accepted_chunks: List[Dict[str, Any]] = Field(default_factory=list)
    trace_id: Optional[str] = None
    error: Optional[str] = None


class IngestResponse(BaseModel):
    added: int
    total_documents: int


class CorpusInfo(BaseModel):
    total_chunks: int
    source_files: List[str] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    error: str
    message: str