"""HealRAG — Agentic Self-Correction CRAG (Corrective RAG) Pipeline.

A full-stack reference implementation combining vector search with an
evaluator–optimizer agent loop. Retrieved chunks are graded for relevance by a
local SLM (via Ollama), and low-relevance retrieval triggers automated query
rewriting plus a BM25 keyword-index fallback. A hallucination-detection loop
rejects unsatisfactory generations, forces a re-retrieval pass, and refines the
final output before it is shown to the user.
"""

__version__ = "1.0.0"
