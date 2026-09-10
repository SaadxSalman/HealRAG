"""LangGraph state machine for the Corrective RAG (CRAG) pipeline.

The graph encodes two evaluator–optimizer loops:

1. **Relevance / corrective loop** — dense retrieval -> LLM relevance grading ->
   if low relevance, rewrite the query + fall back to the BM25 keyword index ->
   re-grade -> loop up to ``MAX_QUERY_REWRITES`` times.

2. **Hallucination loop** — generation -> hallucination check -> if the answer
   is not fully grounded, force an expanded re-retrieval pass and re-generate
   with refinement feedback, up to ``MAX_HALLUCINATION_RETRIES`` times.

Using LangGraph gives us a stateful, inspectable, rerunnable graph where every
node round-trips a typed ``GraphState`` and routing is deterministic.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, TypedDict

from langgraph.graph import END, START, StateGraph

from app.config import settings
from app.core.error_handling import EmptyCorpusError, logger
from app.agents.grader import HallucinationGrader, RelevanceGrader
from app.agents.generator import get_generator
from app.agents.retriever import get_retriever
from app.agents.rewriter import get_rewriter
from app.agents.types import Chunk, PipelineResult

# Enough accepted chunks to attempt grounded generation.
MIN_ACCEPTED = 1


class GraphState(TypedDict, total=False):
    query: str
    original_query: str
    active_query: str                       # may be a rewritten query
    rewrite_count: int
    rewritten_queries: List[str]
    chunks: List[Dict[str, Any]]            # all graded chunks
    accepted_chunks: List[Dict[str, Any]]
    answer: str
    answer_attempts: List[str]
    hallucination_result: Dict[str, Any]
    retries: int
    corrections: List[str]
    steps: Annotated[List[str], operator.add]
    result_error: str


# ---------------------------------------------------------------------- #
# Routing helpers
# ---------------------------------------------------------------------- #
def _accepted(state: GraphState) -> List[Dict[str, Any]]:
    return [c for c in state.get("chunks", []) if c.get("accepted")]


def _has_enough_accepted(state: GraphState) -> bool:
    return len(_accepted(state)) >= MIN_ACCEPTED or (
        state.get("rewrite_count", 0) >= settings.max_query_rewrites
    )


# ---------------------------------------------------------------------- #
# Nodes
# ---------------------------------------------------------------------- #
def _retrieve_node(state: GraphState) -> dict:
    retriever = get_retriever()
    retriever.sync_bm25()  # keep keyword index aligned with vector store
    query = state.get("active_query") or state["query"]
    chunks: List[Chunk]
    try:
        chunks = retriever.retrieve_vector(query, k=settings.retrieval_k)
    except EmptyCorpusError as exc:
        return {"result_error": str(exc)}
    if not chunks:
        # No vector hits at all -> try BM25 as an early fallback.
        chunks = retriever.retrieve_bm25(query, k=settings.bm25_k)
    return {"chunks": [c.to_dict() for c in chunks], "steps": ["retrieve"]}


def _grade_node(state: GraphState) -> dict:
    if not state.get("chunks"):
        return {"chunks": [], "accepted_chunks": [], "steps": ["grade"]}
    grader = RelevanceGrader()
    query = state.get("active_query") or state["query"]
    graded = grader.grade_batch(query, state["chunks"])
    return {
        "chunks": graded,
        "accepted_chunks": [c for c in graded if c.get("accepted")],
        "steps": ["grade"],
    }


def _rewrite_node(state: GraphState) -> dict:
    """Rewrite the failed query and run a BM25 fallback retrieval, then grade."""
    rewriter = get_rewriter()
    original = state["query"]
    last_query = state.get("active_query") or original
    try:
        new_query = rewriter.rewrite(last_query)
    except Exception as exc:
        logger.warning("Rewrite failed: %s", exc)
        new_query = last_query
    count = state.get("rewrite_count", 0) + 1

    retriever = get_retriever()
    bm25_chunks = retriever.retrieve_bm25(new_query, k=settings.bm25_k)
    # If BM25 turns up nothing useful, broaden with the vector store too.
    if not bm25_chunks:
        bm25_chunks = retriever.retrieve_vector(new_query, k=settings.retrieval_k)

    grader = RelevanceGrader()
    graded = grader.grade_batch(new_query, [c.to_dict() for c in bm25_chunks])
    rewritten_queries = state.get("rewritten_queries", []) + [new_query]
    return {
        "active_query": new_query,
        "rewrite_count": count,
        "rewritten_queries": rewritten_queries,
        "chunks": graded,
        "accepted_chunks": [c for c in graded if c.get("accepted")],
        "corrections": state.get("corrections", [])
        + [f"query rewritten ({count}): '{last_query}' -> '{new_query}' (BM25 fallback)"],
        "steps": ["rewrite"],
    }


def _generate_node(state: GraphState) -> dict:
    if not state.get("accepted_chunks"):
        return {
            "answer": "I don't have enough information in the knowledge base to answer this question accurately.",
            "answer_attempts": state.get("answer_attempts", [])
            + ["I don't have enough information in the knowledge base."],
            "steps": ["generate"],
        }
    generator = get_generator()
    query = state.get("active_query") or state["query"]
    refinement = None
    if state.get("hallucination_result"):
        claims = state["hallucination_result"].get("unsupported_claims", [])
        if claims:
            refinement = "- " + "\n- ".join(str(x) for x in claims)
    try:
        answer = generator.generate(query, state["accepted_chunks"], refinement=refinement)
    except Exception as exc:
        logger.error("Generation failed: %s", exc)
        answer = f"Generation failed: {exc}"
    return {
        "answer": answer,
        "answer_attempts": state.get("answer_attempts", []) + [answer],
        "steps": ["generate"],
    }


def _hallucination_node(state: GraphState) -> dict:
    """Grade the generated answer for grounding (hallucination detection)."""
    if not state.get("answer"):
        return {
            "hallucination_result": {
                "verdict": "pass",
                "score": 1.0,
                "unsupported_claims": [],
                "reason": "no answer to verify",
            },
            "steps": ["hallucination_check"],
        }
    grader = HallucinationGrader()
    result = grader.verify(
        query=state["query"],
        answer=state["answer"],
        context_passages=[c["text"] for c in state.get("accepted_chunks", [])],
    )
    return {"hallucination_result": result, "steps": ["hallucination_check"]}


def _corrective_rerefetch_node(state: GraphState) -> dict:
    """Expanded re-retrieval pass triggered when the answer is not grounded."""
    retriever = get_retriever()
    count = state.get("retries", 0) + 1
    query = state.get("active_query") or state["query"]
    # Broaden: more chunks, blend in BM25 lexical hits.
    chunks = retriever.retrieve_hybrid(
        query, vector_k=settings.retrieval_k + 2, bm25_k=settings.bm25_k + 2
    )
    existing_texts = {c.get("text") for c in state.get("accepted_chunks", [])}
    fresh = [c.to_dict() for c in chunks if c.text not in existing_texts]
    return {
        "retries": count,
        "chunks": state.get("chunks", []) + fresh,
        "corrections": state.get("corrections", [])
        + [f"hallucination re-retrieval pass #{count}: broadened to {len(fresh)} new chunk(s)"],
        "steps": ["corrective_rerefetch"],
    }


# ---------------------------------------------------------------------- #
# Conditional edges
# ---------------------------------------------------------------------- #
def route_after_grade(state: GraphState) -> str:
    if state.get("result_error"):
        return "end_with_error"
    if _has_enough_accepted(state):
        return "generate"
    return "rewrite"


def route_after_rewrite(state: GraphState) -> str:
    count = state.get("rewrite_count", 0)
    if len(state.get("accepted_chunks", [])) >= MIN_ACCEPTED or count >= settings.max_query_rewrites:
        return "generate"
    return "rewrite"


def route_after_hallucination(state: GraphState) -> str:
    result = state.get("hallucination_result", {})
    retries = state.get("retries", 0)
    if result.get("verdict") == "fail" and retries < settings.max_hallucination_retries:
        return "corrective_rerefetch"     # force another retrieval + regeneration
    return "end"


# ---------------------------------------------------------------------- #
# Graph construction
# ---------------------------------------------------------------------- #
def build_graph():
    builder = StateGraph(GraphState)

    builder.add_node("retrieve", _retrieve_node)
    builder.add_node("grade", _grade_node)
    builder.add_node("rewrite", _rewrite_node)
    builder.add_node("generate", _generate_node)
    builder.add_node("hallucination", _hallucination_node)
    builder.add_node("corrective_rerefetch", _corrective_rerefetch_node)
    builder.add_node("end_with_error", lambda s: s)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges(
        "grade",
        route_after_grade,
        {"generate": "generate", "rewrite": "rewrite", "end_with_error": "end_with_error"},
    )
    builder.add_conditional_edges(
        "rewrite",
        route_after_rewrite,
        {"generate": "generate", "rewrite": "rewrite"},
    )
    builder.add_edge("generate", "hallucination")
    builder.add_conditional_edges(
        "hallucination",
        route_after_hallucination,
        {"corrective_rerefetch": "corrective_rerefetch", "end": END},
    )
    builder.add_edge("corrective_rerefetch", "generate")
    builder.add_edge("end_with_error", END)

    return builder.compile()


_compiled = None


def get_graph():
    global _compiled
    if _compiled is None:
        _compiled = build_graph()
    return _compiled


# ---------------------------------------------------------------------- #
# Public entry point
# ---------------------------------------------------------------------- #
def run_pipeline(query: str, stream: bool = False) -> PipelineResult:
    """Run the full agentic CRAG pipeline for a query and return a result."""
    result = PipelineResult(query=query)
    initial: GraphState = {
        "query": query,
        "original_query": query,
        "active_query": query,
        "rewrite_count": 0,
        "rewritten_queries": [],
        "chunks": [],
        "accepted_chunks": [],
        "answer_attempts": [],
        "retries": 0,
        "corrections": [],
        "steps": [],
    }

    graph = get_graph()
    events: List[Dict[str, Any]] = []
    try:
        if stream:
            for snapshot in graph.stream(initial, stream_mode="values"):
                events.append(snapshot)
            final = events[-1] if events else initial
        else:
            final = graph.invoke(initial)
    except Exception as exc:
        logger.exception("Pipeline failed")
        result.error = str(exc)
        result.add_step("error")
        return result

    for s in final.get("steps", []):
        result.add_step(s)

    result.rewritten_queries = final.get("rewritten_queries", [])
    result.chunks = final.get("chunks", [])
    result.accepted_chunks = final.get("accepted_chunks", [])
    result.answer = final.get("answer", "")
    result.corrections = final.get("corrections", [])
    result.retries = final.get("retries", 0)

    hres = final.get("hallucination_result", {})
    result.hallucination_grade = hres.get("score")
    # Confidence = blend of hallucination pass certainty and accepted relevance.
    accepted = result.accepted_chunks
    if accepted:
        avg_rel = sum(float(c.get("relevance") or 0) for c in accepted) / len(accepted)
        if hres.get("score") is not None:
            result.confidence = round(
                min(1.0, 0.5 * avg_rel + 0.5 * float(hres["score"])), 3
            )
        else:
            result.confidence = round(avg_rel, 3)

    result.trace["answer_attempts"] = final.get("answer_attempts", [])
    result.trace["final_state"] = {
        k: v for k, v in final.items() if k not in ("chunks", "accepted_chunks")
    }
    result.trace["event_snapshots"] = events
    return result