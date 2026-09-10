"""Unit tests for document loading and chunking."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.loader import extract_text, load_document
from app.core.ingestion import ingest_chunks


def test_chunk_overlap(tmp_path):
    doc = tmp_path / "doc.md"
    doc.write_text("word " * 1000, encoding="utf-8")
    chunks = load_document(doc)
    assert len(chunks) > 1
    # Chunks should overlap slightly instead of losing content between them.
    first_end = chunks[0][0][-60:]
    assert any(first_end in c[0] for c in chunks[1:2])


def test_metadata_carries_source(tmp_path):
    doc = tmp_path / "notes.txt"
    doc.write_text("Experiment A produced 42 widgets.", encoding="utf-8")
    chunks = load_document(doc)
    assert chunks[0][1]["source_file"] == "notes.txt"


def test_unsupported_extension_raises(tmp_path):
    doc = tmp_path / "x.bin"
    doc.write_bytes(b"abc")
    with pytest.raises(Exception):
        extract_text(doc)


def test_ingest_chunks_empty():
    assert ingest_chunks([]) == 0