"""Seed the knowledge base with the bundled demo documents.

Usage:  python -m app.scripts.seed_data
"""

from __future__ import annotations

import sys
from pathlib import Path

from app.config import settings, PROJECT_ROOT
from app.core.error_handling import setup_logging
from app.core.ingestion import ingest_directory
from app.core.vector_store import get_vector_store

logger = setup_logging()


def main() -> int:
    data_dir = settings.data_dir
    logger.info("Seeding corpus from %s", data_dir)
    store = get_vector_store()
    added = ingest_directory(data_dir)
    logger.info("Done. Added %d chunk(s); vector store now holds %d document(s).", added, store.count)
    return 0


if __name__ == "__main__":
    sys.exit(main())