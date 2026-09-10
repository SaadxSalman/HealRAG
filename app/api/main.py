"""HealRAG FastAPI application entry point.

Run with:  uvicorn app.api.main:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.deps import cors_middleware, register_exception_handlers
from app.api.routes import router
from app.config import settings
from app.core.error_handling import setup_logging

logger = setup_logging(settings.log_level)

description = """
**HealRAG — Agentic Self-Correction CRAG Pipeline**

A local-first RAG system that grades every retrieved chunk with a small language
model, auto-rewrites low-relevance queries, falls back to a BM25 keyword index,
and detects + corrects hallucinations before returning answers.

* Vector store: ChromaDB (local)
* Grader / generator / rewriter: Ollama (qwen3:8b | phi-4-mini)
* Orchestration: LangGraph evaluator–optimizer loops
"""

app = FastAPI(
    title=settings.app_name,
    description=description,
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

cors_middleware(app)
register_exception_handlers(app)
app.include_router(router)


@app.on_event("startup")
async def startup() -> None:
    """Validate prerequisites and keep the BM25 index in sync on boot."""
    from app.agents.retriever import get_retriever
    from app.core.llm import get_ollama_client

    retriever = get_retriever()
    retriever.sync_bm25()
    logger.info("BM25 index synced with %d chunk(s).", retriever.bm25.count)

    client = get_ollama_client()
    client.ensure_model(settings.ollama_chat_model)
    client.ensure_model(settings.ollama_generation_model)
    client.ensure_model(settings.ollama_embed_model)
    logger.info(
        "Startup complete. Ollama reachable: %s | chunks indexed: %d",
        client.is_healthy(),
        get_vector_store().count,
    )


@app.get("/")
async def root():
    return {
        "name": settings.app_name,
        "version": __version__,
        "docs": "/docs",
        "health": "/health",
        "message": "CRAG pipeline is running. Ask a question via POST /ask.",
    }