"""ChromaDB vector store wrapper.

Provides document ingestion + retrieval with persistent local storage. Stores
embedding model + dimension in collection metadata so that a change in the
embedding configuration triggers a transparent rebuild instead of silent
mismatched-vector bugs.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from app.config import settings
from app.core.embeddings import get_embedding_service
from app.core.error_handling import EmptyCorpusError, RetrievalError, logger


class VectorStore:
    def __init__(self, persist_dir: Optional[str] = None) -> None:
        import chromadb

        self._dir = persist_dir or str(settings.chroma_path)
        settings.chroma_path.mkdir(parents=True, exist_ok=True)
        # Client-side (persistent) Chroma; no credentials involved.
        self._client = chromadb.PersistentClient(path=self._dir)
        self._collection = self._get_or_create_collection()
        self._ensure_dimension_matches()

    # ------------------------------------------------------------------ #
    def _get_or_create_collection(self):
        import chromadb

        try:
            return self._client.get_collection(settings.chroma_collection_name)
        except Exception:
            emb = get_embedding_service()
            return self._client.create_collection(
                name=settings.chroma_collection_name,
                metadata={
                    "embedding_model": emb.model_name,
                    "embedding_dimension": emb.dimension,
                    "created_for": "healrag",
                },
            )

    def _ensure_dimension_matches(self) -> None:
        """Rebuild the collection if the embedding model changed."""
        emb = get_embedding_service()
        meta = self._collection.metadata or {}
        stored_model = meta.get("embedding_model")
        if stored_model and stored_model != emb.model_name:
            logger.warning(
                "Embedding model changed (%s -> %s). Rebuilding vector store.",
                stored_model,
                emb.model_name,
            )
            self._client.delete_collection(settings.chroma_collection_name)
            self._collection = self._get_or_create_collection()

    # ------------------------------------------------------------------ #
    @property
    def count(self) -> int:
        return self._collection.count()

    def is_empty(self) -> bool:
        return self.count == 0

    def reset(self) -> None:
        """Delete all documents in the collection."""
        self._client.delete_collection(settings.chroma_collection_name)
        self._collection = self._get_or_create_collection()

    # ------------------------------------------------------------------ #
    def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Embed and store a list of chunk texts. Returns assigned ids."""
        if not texts:
            return []
        emb = get_embedding_service()
        vectors = emb.embed(texts)
        if metadatas is None:
            metadatas = [{} for _ in texts]
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        self._collection.add(
            ids=ids,
            documents=texts,
            embeddings=vectors,
            metadatas=metadatas,
        )
        logger.info("Added %d chunk(s) to vector store (total=%d)", len(texts), self.count)
        return ids

    # ------------------------------------------------------------------ #
    def query(
        self, query_text: str, k: Optional[int] = None, where: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Vector-similarity retrieval. Returns list of chunk dicts."""
        if self.is_empty():
            raise EmptyCorpusError(
                "The vector store is empty. Run seeding/ingestion first."
            )
        k = k or settings.retrieval_k
        emb = get_embedding_service().embed([query_text])
        try:
            res = self._collection.query(
                query_embeddings=emb,
                n_results=min(k, self.count),
                where=where,
            )
        except Exception as exc:
            raise RetrievalError(f"Chromadb query failed: {exc}") from exc

        chunks: List[Dict[str, Any]] = []
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        distances = res.get("distances", [[]])[0]
        for i, cid in enumerate(ids):
            chunks.append(
                {
                    "id": cid,
                    "text": docs[i] if i < len(docs) else "",
                    "metadata": metas[i] if i < len(metas) else {},
                    "distance": distances[i] if i < len(distances) else None,
                    "score": _similarity(distances[i]) if i < len(distances) else None,
                    "source": "vector",
                }
            )
        chunks.sort(key=lambda c: c.get("score") or 0, reverse=True)
        return chunks


def _similarity(distance: Optional[float]) -> Optional[float]:
    """Cosine distance -> similarity. Chroma uses cosine distance by default."""
    if distance is None:
        return None
    return max(0.0, 1.0 - float(distance))


_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store