"""Unit tests for the BM25 keyword fallback index."""

from __future__ import annotations

import pytest

from app.core.bm25_index import BM25Index, _tokenize


@pytest.fixture()
def index():
    idx = BM25Index()
    idx.rebuild_from_chunks(
        [
            {"id": "1", "text": "Hypertension treatment includes reducing sodium intake.", "metadata": {}},
            {"id": "2", "text": "Type 2 diabetes requires glucose monitoring and metformin.", "metadata": {}},
            {"id": "3", "text": "Sleep disorders respond well to cognitive behavioral therapy.", "metadata": {}},
        ]
    )
    return idx


def test_tokenize_removes_stopwords():
    toks = _tokenize("The Hypertension Guide for sodium")
    assert "the" not in toks
    assert "for" not in toks
    assert "hypertension" in toks
    assert "sodium" in toks


def test_rebuild_and_count(index):
    assert index.count == 3
    assert not index.is_empty()


def test_relevant_doc_ranks_first(index):
    results = index.query("hypertension sodium treatment")
    assert results, "expected at least one hit"
    assert results[0]["id"] == "1"
    assert results[0]["source"] == "bm25"
    assert results[0]["score"] is not None


def test_query_emptiness():
    idx = BM25Index()
    assert idx.is_empty()
    with pytest.raises(Exception):
        idx.query("anything")