"""One-command launcher: starts the FastAPI backend and Streamlit UI.

Usage:  python scripts/launch.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from app.config import settings
from app.core.error_handling import setup_logging

logger = setup_logging()
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _spawn(cmd: list) -> subprocess.Popen:
    return subprocess.Popen(cmd, cwd=str(PROJECT_ROOT))


def main() -> int:
    print("=" * 64)
    print("  HealRAG — Agentic Self-Correction CRAG Pipeline")
    print("=" * 64)

    # Optionally auto-seed if the store is empty.
    from app.core.vector_store import get_vector_store

    store = get_vector_store()
    if store.is_empty():
        logger.info("Vector store empty -> seeding demo corpus.")
        from app.core.ingestion import ingest_directory

        ingest_directory(settings.data_dir)

    api = _spawn(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.api.main:app",
            "--host",
            str(settings.api_host),
            "--port",
            str(settings.api_port),
        ]
    )
    logger.info("API starting at http://%s:%s", settings.api_host, settings.api_port)
    time.sleep(3.0)

    ui = _spawn(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app/ui/streamlit_app.py",
            "--server.port",
            str(settings.streamlit_port),
            "--server.headless",
            "true",
        ]
    )
    logger.info("UI starting at http://localhost:%s", settings.streamlit_port)

    print("\nPress Ctrl+C to stop both services.\n")
    try:
        api.wait()
    except KeyboardInterrupt:
        pass
    finally:
        api.terminate()
        ui.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())