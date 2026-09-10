"""Pytest fixtures. Ensures the project root is importable and config is isolated."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(autouse=True)
def isolate_settings(tmp_path, monkeypatch):
    """Point persistence to a temp dir for every test run."""
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("TRACE_DIR", str(tmp_path / "traces"))
    # Reload cached settings singleton for the test process.
    from app.config import get_settings

    if hasattr(get_settings, "_cached"):
        del get_settings._cached
    yield get_settings()