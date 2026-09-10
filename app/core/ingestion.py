"""Ingestion service.

Coordinates document extraction -> chunking -> embedding -> vector storage,
and keeps the BM25 index synchronized with the vector store after every write.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from app.config import settings
from app.core.error_handling import logger
from app.core.loader import (  # noqa: F401  re-export for convenience
    extract_text,
    load_directory,
    load_document,
)
from app.core.vector_store import get_vector_store
from app.agents.retriever import get_retriever


def ingest_chunks(chunks: List[Tuple[str, dict]]) -> int:
    """Embed + store a list of (text, metadata) chunks. Returns count added."""
    if not chunks:
        return 0
    texts = [c[0] for c in chunks]
    metadatas = [c[1] for c in chunks]
    vector_store = get_vector_store()
    vector_store.add_documents(texts, metadatas)
    # Keep the BM25 lexical index in sync.
    get_retriever().sync_bm25()
    return len(texts)


def ingest_file(path: Path) -> int:
    return ingest_chunks(load_document(path))


def ingest_directory(directory: Path) -> int:
    return ingest_chunks(load_directory(directory))


def ingest_seed_bundle(bundle_path: Path) -> int:
    """Ingest a directory of bundled demo documents (from data/documents)."""
    data_dir = settings.data_dir
    if bundle_path and bundle_path != data_dir:
        # Merge the provided dir with the default demo dir.
        return ingest_directory(bundle_path) + ingest_directory(data_dir)
    return ingest_directory(data_dir)