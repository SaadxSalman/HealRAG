"""Answer generator agent.

Builds a grounded answer strictly from the accepted context chunks. A "trust
frame" instructs the model to (a) answer only from the provided context,
(b) cite chunk numbers, and (c) explicitly say 'I don't know' when the context
is insufficient. The output is later validated by the hallucination grader.
"""

from __future__ import annotations

from typing import List, Optional

from app.config import settings
from app.core.llm import get_ollama_client
from app.agents.grader import build_context_block

GENERATION_SYSTEM = """You are a precise, grounded assistant. You MUST follow these rules:
1. Answer ONLY using the provided context passages. Do not use outside knowledge.
2. If the context does not contain the answer, say: "I don't have enough
   information to answer this." Do not guess.
3. Cite supporting passages inline using their bracket numbers, e.g. [1], [2].
4. Be concise but complete. Use clear paragraphs or bullets.
5. Never fabricate numbers, dates, names, or statistics.
"""

TRUE_SYSTEM = """You are a careful, honest assistant. Wherever the retrieved context is thin,
explicitly flag uncertainty. Reconcile contradictions between passages and tell
the user which sources conflict. Never paraphrase what you cannot find."""


class Generator:
    def __init__(self) -> None:
        self.client = get_ollama_client()
        self.model = settings.ollama_generation_model

    def generate(self, query: str, chunks: List[dict], refinement: Optional[str] = None) -> str:
        """Produce an answer from accepted chunks (dicts with 'text')."""
        context = build_context_block(chunks)
        system = GENERATION_SYSTEM
        user = (
            f"QUESTION: {query}\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"Answer the question using the context and cite sources with [n]."
        )
        if refinement:
            system = TRUE_SYSTEM
            user = (
                f"QUESTION: {query}\n\n"
                f"CONTEXT:\n{context}\n\n"
                "A previous answer had these unsupported claims:\n"
                f"{refinement}\n\n"
                "Produce a corrected, fully-grounded answer. Cite sources with [n]. "
                "If a claim cannot be grounded, remove it or state the uncertainty."
            )
        return self.client.chat(
            user,
            system=system,
            model=self.model,
            temperature=settings.generation_temperature,
            max_tokens=1200,
        )


def get_generator() -> Generator:
    global _generator
    if _generator is None:
        _generator = Generator()
    return _generator


_generator: Optional[Generator] = None