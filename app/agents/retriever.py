"""Retrieval agent — dense vector + BM25 keyword + blended retrieval."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.config import settings
from app.core.bm25_index import get_bm25_index
from app.core.error_handling import EmptyCorpusError, logger
from app.core.vector_store import get_vector_store
from app.agents.types import Chunk


def _to_chunk(payload: Dict[str, Any], source: str = "vector") -> Chunk:
    return Chunk(
        id=payload.get("id", ""),
        text=payload.get("text", ""),
        metadata=payload.get("metadata", {}),
        score=payload.get("score"),
        source=source,
    )


class Retriever:
    """Coordinates vector + BM25 retrieval and merges results."""

    def __init__(self) -> None:
        self.vector_store = get_vector_store()
        self.bm25 = get_bm25_index()

    def sync_bm25(self) -> None:
        """Rebuild the BM25 index from the current vector-store contents.

        ChromaDB exposes the full corpus only via ``get()``, so we pull all
        documents and refresh the in-memory keyword index. Called once at
        startup (and after ingestion) to keep both indexes aligned.
        """
        try:
            all_data = self.vector_store._collection.get(include=["documents", "metadatas"])
        except Exception:
            return
        chunks = [
            {"id": i, "text": t, "metadata": m or {}}
            for i, t, m in zip(
                all_data.get("ids", []),
                all_data.get("documents", []),
                all_data.get("metadatas", []),
            )
        ]
        self.bm25.rebuild_from_chunks(chunks)

    # ------------------------------------------------------------------ #
    def retrieve_vector(self, query: str, k: Optional[int] = None) -> List[Chunk]:
        try:
            payloads = self.vector_store.query(query, k=k)
        except EmptyCorpusError:
            logger.info("Vector store empty during retrieval.")
            return []
        return [_to_chunk(p, "vector") for p in payloads]

    def retrieve_bm25(self, query: str, k: Optional[int] = None) -> List[Chunk]:
        try:
            payloads = self.bm25.query(query, k=k)
        except EmptyCorpusError:
            return []
        return [_to_chunk(p, "bm25") for p in payloads]

    def retrieve_hybrid(
        self,
        query: str,
        vector_k: Optional[int] = None,
        bm25_k: Optional[int] = None,
    ) -> List[Chunk]:
        """Blend dense + lexical results, de-duped, vector-ranked first."""
        vector_chunks = self.retrieve_vector(query, k=vector_k or settings.retrieval_k)
        bm25_chunks = self.retrieve_bm25(query, k=bm25_k or settings.bm25_k)

        seen: Dict[str, Chunk] = {}
        for c in vector_chunks + bm25_chunks:
            if c.text not in seen:
                seen[c.text] = c
            else:
                # prefer the vector copy but keep a bm25 score hint if present
                existing = seen[c.text]
                if c.source == "bm25" and getattr(existing, "bm25_score", None) is None:
                    existing.metadata.setdefault("bm25_score", c.metadata.get("bm25_score"))
        merged = list(seen.values())
        # Keep vector chunks first, then bm25-only chunks (which are fallback).
        return merged


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


_retriever: Optional[Retriever] = None