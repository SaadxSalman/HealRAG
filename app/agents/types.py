"""Shared data structures for the agent pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Chunk:
    """A single retrieved text chunk with provenance + a relevance grade."""

    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: Optional[float] = None          # raw retrieval score (vector/bm25)
    source: str = "vector"                 # vector | bm25 | blended
    relevance: Optional[float] = None      # LLM grader score in [0,1]
    relevance_reason: Optional[str] = None
    accepted: bool = False                 # passed the relevance threshold

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
            "score": self.score,
            "source": self.source,
            "relevance": self.relevance,
            "relevance_reason": self.relevance_reason,
            "accepted": self.accepted,
        }


@dataclass
class PipelineResult:
    """Everything produced by one query run (used for API + UI + tracing)."""

    query: str
    rewritten_queries: List[str] = field(default_factory=list)
    chunks: List[Chunk] = field(default_factory=list)
    accepted_chunks: List[Chunk] = field(default_factory=list)
    answer: str = ""
    grade: Optional[float] = None
    hallucination_grade: Optional[float] = None
    retries: int = 0
    steps: List[str] = field(default_factory=list)
    corrections: List[str] = field(default_factory=list)
    confidence: Optional[float] = None
    error: Optional[str] = None
    trace: Dict[str, Any] = field(default_factory=dict)

    def add_step(self, name: str) -> None:
        self.steps.append(name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "rewritten_queries": self.rewritten_queries,
            "chunks": [c.to_dict() for c in self.chunks],
            "accepted_chunks": [c.to_dict() for c in self.accepted_chunks],
            "answer": self.answer,
            "grade": self.grade,
            "hallucination_grade": self.hallucination_grade,
            "retries": self.retries,
            "steps": self.steps,
            "corrections": self.corrections,
            "confidence": self.confidence,
            "error": self.error,
            "trace": self.trace,
        }
