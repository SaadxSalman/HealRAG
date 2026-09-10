"""Embedding service.

Abstracts two local embedding backends:
  * ``ollama``              -> nomic-embed-text (recommended, zero install)
  * ``sentence_transformers`` -> all-MiniLM-L6-v2 (optional pip install)

The ChromaDB collection metadata stores the embedding model + dimension so a
stale model change is detected and the store rebuilt automatically.
"""

from __future__ import annotations

from typing import List

from app.config import settings
from app.core.error_handling import ConfigError, logger


class EmbeddingService:
    def __init__(self, backend: str = "ollama") -> None:
        self.backend = backend.lower()
        self._st_client = None  # lazy import for sentence_transformers

    # ------------------------------------------------------------------ #
    @property
    def model_name(self) -> str:
        if self.backend == "ollama":
            return settings.ollama_embed_model
        if self.backend == "sentence_transformers":
            return settings.sentence_transformer_model
        raise ConfigError(f"Unknown embedding backend: {self.backend}")

    @property
    def dimension(self) -> int:
        if self.backend == "ollama":
            return settings.embedding_dimension
        return 384  # all-MiniLM-L6-v2

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts into a list of dense vectors."""
        if not texts:
            return []
        if self.backend == "ollama":
            from app.core.llm import get_ollama_client

            vectors = get_ollama_client().embed(texts)
            logger.info("Embedded %d text(s) via Ollama/%s", len(texts), self.model_name)
            return vectors

        if self.backend == "sentence_transformers":
            if self._st_client is None:
                from sentence_transformers import SentenceTransformer

                logger.info("Loading sentence-transformer model %s", self.model_name)
                self._st_client = SentenceTransformer(self.model_name)
            return self._st_client.encode(texts, normalize_embeddings=True).tolist()

        raise ConfigError(f"Unknown embedding backend: {self.backend}")

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]


_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    global _service
    if _service is None:
        _service = EmbeddingService(settings.embedding_backend)
    return _service