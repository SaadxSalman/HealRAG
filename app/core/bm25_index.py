"""BM25 lexical keyword index.

A lightweight in-memory fallback index used by the corrective loop when the
dense vector retrieval is graded as low-relevance. Implemented with
``rank-bm25`` (Okapi-BM25). Documents are kept in sync with the vector store:
each time the vector store is (re)built, ``rebuild_from_chunks`` is called so
both indexes see identical content.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from app.config import settings
from app.core.error_handling import EmptyCorpusError, logger

# Small default stopword set (avoid importing heavy NLTK for this demo).
_STOPWORDS = set(
    """
    a an the and or but if then else for with without from by of in on at to is
    are was were be been being do does did have has had will would shall should
    can could may might must not no nor only just very how what which who whom
    this that these those it its their our your my his her about into over under
    again further once here there when where why while all any both each few more
    most other some such than too so as up down out off above below
    """.split()
)


def _tokenize(text: str) -> List[str]:
    """Split into lowercase alphanumeric tokens."""
    tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


class BM25Index:
    def __init__(self) -> None:
        self._docs: List[Dict[str, Any]] = []
        self._corpus: List[List[str]] = []
        self._bm25 = None
        self._k1 = 1.5
        self._b = 0.75

    @property
    def count(self) -> int:
        return len(self._docs)

    def is_empty(self) -> bool:
        return self.count == 0

    # ------------------------------------------------------------------ #
    def rebuild_from_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """(Re)build the index from a list of chunk dicts (must have 'text')."""
        from rank_bm25 import BM25Okapi

        self._docs = list(chunks)
        self._corpus = [_tokenize(c["text"]) for c in chunks]
        if self._corpus:
            self._bm25 = BM25Okapi(self._corpus, k1=self._k1, b=self._b)
        else:
            self._bm25 = None
        logger.info("BM25 index rebuilt with %d document(s)", self.count)

    # ------------------------------------------------------------------ #
    def query(self, query_text: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return top-k BM25-scored chunks."""
        if self.is_empty() or self._bm25 is None:
            raise EmptyCorpusError("BM25 index is empty. Run ingestion first.")
        k = k or settings.bm25_k
        tokens = _tokenize(query_text)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results: List[Dict[str, Any]] = []
        for idx in ranked[:k]:
            if scores[idx] <= 0:
                continue
            doc = self._docs[idx]
            results.append(
                {
                    **doc,
                    "bm25_score": float(scores[idx]),
                    "source": "bm25",
                }
            )
        # Normalize scores to [0,1] for a consistent grading surface.
        if results:
            max_s = max(r["bm25_score"] for r in results)
            if max_s > 0:
                for r in results:
                    r["score"] = r["bm25_score"] / max_s
        return results


_index: Optional[BM25Index] = None


def get_bm25_index() -> BM25Index:
    global _index
    if _index is None:
        _index = BM25Index()
    return _index
