"""LLM-based grading agents.

Two graders power the evaluator–optimizer loops:

1. ``RelevanceGrader`` — grades each retrieved chunk against the query on a
   ``0..10`` scale and decides "accept | reject". Output is forced to JSON via
   Ollama's ``format=json`` support, so downstream routing is deterministic.

2. ``HallucinationGrader`` — after generation, verifies that every factual
   claim in the answer is grounded in the accepted context. Answers that fail
   are rejected and trigger a corrective re-retrieval + regeneration cycle.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.config import settings
from app.core.error_handling import logger
from app.core.llm import get_ollama_client

RELEVANCE_SYSTEM = """You are a meticulous retrieval evaluator. Your only job is to judge
whether a retrieved text chunk is RELEVANT to answering a user's question.

Rules:
- Return STRICT JSON with EXACTLY these keys:
  {"score": <integer 0..10>, "verdict": "accept"|"reject", "reason": "<short reason>"}
- "accept" means the chunk contains information that helps answer the question.
- "reject" means the chunk is unrelated, redundant, or offers no usable signal.
- Be strict: borderline chunks should be "reject".
- The score reflects the degree of usefulness (10 = perfectly answers it).
"""

HALLUCINATION_SYSTEM = """You are an expert fact-checker. You are given:
1. A set of "context" passages retrieved from a knowledge base.
2. A generated "answer".

Your job is to detect whether the answer is grounded in the context or contains
hallucinated/unverifiable claims.

Return STRICT JSON with EXACTLY these keys:
{
  "score": <integer 0..10>,
  "verdict": "pass"|"fail",
  "unsupported_claims": ["<claim not supported by context>", ...],
  "reason": "<one sentence explanation>"
}
- "pass" (score >= 7) means every claim in the answer is supported by context.
- "fail" (score < 7) means part of the answer is ungrounded.
- List each unsupported claim verbatim or near-verbatim.
"""

_FENCE = "'''"

# Keys some SLMs emit instead of the canonical "score".
_SCORE_KEYS = ("score", "relevance", "rating", "usefulness")


def _extract_score(raw: Dict[str, Any]) -> Optional[float]:
    """Pull a 0..10 score out of possibly off-spec JSON, clamped to [0, 1].

    Returns ``None`` when no numeric score-like field is present.
    """
    for key in _SCORE_KEYS:
        value = raw.get(key)
        if value is None:
            continue
        try:
            return _clamp01(value)
        except (TypeError, ValueError):
            continue
    return None


def _derive_verdict(
    raw: Dict[str, Any],
    score: Optional[float],
    positive: str,
    negative: str,
    positive_cutoff: float,
) -> Optional[str]:
    """Normalize the verdict; infer it from the score when missing/invalid.

    Returns ``None`` only when neither a usable verdict nor a score exists.
    """
    verdict = str(raw.get("verdict", "")).lower().strip()
    if verdict in (positive, negative):
        return verdict
    if score is not None:
        return positive if score >= positive_cutoff else negative
    return None


def _build_relevance_prompt(query: str, chunk_id: str, chunk_text: str) -> str:
    return (
        f"QUESTION: {query}\n\n"
        f"CHUNK ID: {chunk_id}\n"
        f"CHUNK CONTENT:\n{_FENCE}\n{chunk_text[:1800]}\n{_FENCE}\n\n"
        "Grade the relevance of this chunk to the question."
    )


def _build_hallucination_prompt(query: str, answer: str, context: str) -> str:
    return (
        f"QUESTION: {query}\n\n"
        f"CONTEXT PASSAGES:\n{_FENCE}\n{context[:6000]}\n{_FENCE}\n\n"
        f"ANSWER TO VERIFY:\n{_FENCE}\n{answer}\n{_FENCE}\n\n"
        "Check whether the answer is fully grounded in the CONTEXT PASSAGES."
    )


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, v / 10.0))


class RelevanceGrader:
    """Grades chunk relevance using a fast local SLM via Ollama."""

    def __init__(self, model: Optional[str] = None) -> None:
        self.model = model or settings.ollama_chat_model
        self.client = get_ollama_client()

    def grade_chunk(self, query: str, chunk_id: str, chunk_text: str) -> Dict[str, Any]:
        prompt = _build_relevance_prompt(query, chunk_id, chunk_text)
        raw = self.client.chat_json(
            prompt,
            system=RELEVANCE_SYSTEM,
            model=self.model,
            temperature=settings.grader_temperature,
            max_tokens=300,
        )
        score = _extract_score(raw)
        verdict = _derive_verdict(raw, score, "accept", "reject", 0.5)
        if verdict == "accept" and score is None:
            # Accept-without-score: assume moderately relevant.
            score = 0.7
        if verdict is None:
            raise ValueError(f"unusable grader JSON: {str(raw)[:200]}")
        reason = str(raw.get("reason") or raw.get("explanation") or "").strip()
        return {
            "relevance": score if score is not None else 0.0,
            "verdict": verdict,
            "reason": reason,
        }

    def grade_batch(
        self, query: str, chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Grade every chunk, returning merged payloads."""
        graded: List[Dict[str, Any]] = []
        for c in chunks:
            try:
                result = self.grade_chunk(query, c["id"], c["text"])
            except Exception as exc:  # a grading failure should not kill the run
                logger.warning("Grading chunk %s failed: %s", c["id"], exc)
                result = {
                    "relevance": 0.0,
                    "verdict": "reject",
                    "reason": f"grading error: {exc}",
                }
            c = {**c, **result}
            c["accepted"] = result["verdict"] == "accept"
            graded.append(c)
        return graded


class HallucinationGrader:
    """Verifies that a generated answer is grounded in the retrieved context."""

    def __init__(self, model: Optional[str] = None) -> None:
        self.model = model or settings.ollama_chat_model
        self.client = get_ollama_client()

    def verify(
        self, query: str, answer: str, context_passages: List[str]
    ) -> Dict[str, Any]:
        """Return pass/fail decision plus unsupported claims."""
        context = "\n\n---\n\n".join(context_passages)
        prompt = _build_hallucination_prompt(query, answer, context)
        try:
            raw = self.client.chat_json(
                prompt,
                system=HALLUCINATION_SYSTEM,
                model=self.model,
                temperature=0.0,
                max_tokens=500,
            )
        except Exception as exc:  # conservative default: fail closed
            logger.warning("Hallucination check failed: %s", exc)
            return {
                "score": 0.0,
                "verdict": "fail",
                "unsupported_claims": ["[unable to verify: grader error]"],
                "reason": f"grader error: {exc}",
            }

        claims = raw.get("unsupported_claims", [])
        score = _extract_score(raw)
        verdict = _derive_verdict(raw, score, "pass", "fail", 0.7)
        if verdict == "pass" and score is None:
            score = 0.8
        if verdict is None:
            # Nothing usable in the response — fail closed.
            score, verdict = 0.0, "fail"
        return {
            "score": score if score is not None else 0.0,
            "verdict": verdict,
            "unsupported_claims": claims if isinstance(claims, list) else [str(claims)],
            "reason": str(raw.get("reason") or raw.get("explanation") or "").strip(),
        }


def build_context_block(chunks: List[Dict[str, Any]]) -> str:
    """Format accepted chunks into a numbered context block for generation."""
    parts = []
    for i, c in enumerate(chunks, start=1):
        src = c.get("metadata", {}).get("source_file", c.get("source", "unknown"))
        parts.append(f"[{i}] (source: {src})\n{c['text']}")
    return "\n\n".join(parts)
