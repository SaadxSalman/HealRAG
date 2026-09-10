"""Query rewriter agent.

When retrieval proves low-relevance, the original query is rewritten into a
more precise form aimed at surfacing the intended information. Uses a fast
local SLM and returns a tight, search-friendly query. Rewrites are recorded so
the pipeline can explain *why* a query changed.
"""

from __future__ import annotations

from typing import Optional

from app.config import settings
from app.core.error_handling import logger
from app.core.llm import get_ollama_client

REWRITE_SYSTEM = """You are an expert search-query rewriter. Rewrite the user's question to make it
more likely to retrieve the precise information they want from a knowledge base.

Rules:
- Keep technical terms, names and numbers untouched.
- Remove filler words.
- Expand ambiguous acronyms only if context allows.
- Target a concise, noun-heavy search-style query (<= 25 words).
- Return STRICT JSON: {"rewritten": "<your query>"}
"""


class QueryRewriter:
    def __init__(self) -> None:
        self.client = get_ollama_client()

    def rewrite(self, query: str, feedback: Optional[str] = None) -> str:
        raw = self.client.chat_json(
            f"ORIGINAL: {query}\n" + (f"FEEDBACK: {feedback}\n" if feedback else "")
            + "Rewrite this query.",
            system=REWRITE_SYSTEM,
            model=settings.ollama_chat_model,
            temperature=settings.rewriter_temperature,
            max_tokens=120,
        )
        rewritten = str(raw.get("rewritten") or raw.get("query") or "").strip() or query
        # A rewrite identical to the original is useless in a corrective loop —
        # nudge the model once for a materially different variant.
        if rewritten.strip().lower() == query.strip().lower():
            try:
                raw2 = self.client.chat_json(
                    f"ORIGINAL: {query}\n"
                    "FEEDBACK: your previous rewrite was identical to the original. "
                    "Produce a clearly different, keyword-focused variant.\n"
                    "Rewrite this query.",
                    system=REWRITE_SYSTEM,
                    model=settings.ollama_chat_model,
                    temperature=max(0.5, settings.rewriter_temperature),
                    max_tokens=120,
                )
                alt = str(raw2.get("rewritten") or raw2.get("query") or "").strip()
                if alt and alt.lower() != query.strip().lower():
                    rewritten = alt
            except Exception as exc:  # nudging is best-effort
                logger.warning("Rewrite nudge failed: %s", exc)
        # Never return an empty rewrite; fall back to the original.
        return rewritten or query


def get_rewriter() -> QueryRewriter:
    global _rewriter
    if _rewriter is None:
        _rewriter = QueryRewriter()
    return _rewriter


_rewriter: Optional[QueryRewriter] = None