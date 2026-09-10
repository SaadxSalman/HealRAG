"""Ingest your own documents.

Place files (.txt/.md/.pdf/.docx/.html) in ./data/documents (or pass --dir) and run:

    python -m app.scripts.ingest --dir ./data/documents

Usage:
    python -m app.scripts.ingest [--dir PATH] [--reset]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.config import settings
from app.core.error_handling import setup_logging
from app.core.ingestion import ingest_directory, ingest_file
from app.core.vector_store import get_vector_store

logger = setup_logging()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Ingest documents into HealRAG.")
    parser.add_argument("--dir", type=Path, default=None, help="Directory of documents.")
    parser.add_argument("--file", type=Path, default=None, help="A single document file.")
    parser.add_argument("--reset", action="store_true", help="Wipe the vector store first.")
    args = parser.parse_args(argv)

    if args.reset:
        get_vector_store().reset()
        logger.info("Vector store reset.")

    if args.file:
        added = ingest_file(args.file)
    else:
        target = args.dir or settings.data_dir
        added = ingest_directory(target)

    store = get_vector_store()
    logger.info("Ingested %d chunk(s). Total in store: %d", added, store.count)
    return 0


if __name__ == "__main__":
    sys.exit(main())