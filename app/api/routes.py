"""API routes: health, ask, ingest, corpus, traces."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile

from app import __version__
from app.agents.graph import run_pipeline
from app.api.deps import AuthDep
from app.api.schemas import (
    AskRequest,
    AskResponse,
    CorpusInfo,
    HealthResponse,
    IngestResponse,
)
from app.config import settings
from app.core.error_handling import EmptyCorpusError, HealRAGError, logger
from app.core.ingestion import ingest_file
from app.core.llm import get_ollama_client
from app.core.tracing import save_trace
from app.core.vector_store import get_vector_store

router = APIRouter()
ALLOWED_EXTENSIONS = {".txt", ".md", ".markdown", ".html", ".htm", ".pdf", ".docx"}


@router.get("/health", response_model=HealthResponse, dependencies=[AuthDep])
async def health() -> HealthResponse:
    client = get_ollama_client()
    try:
        models = client.list_models()
        connected = True
    except Exception:
        models, connected = [], False
    store = get_vector_store()
    return HealthResponse(
        ollama_connected=connected,
        ollama_models=models,
        documents_indexed=store.count,
    )


@router.post("/ask", response_model=AskResponse, dependencies=[AuthDep])
async def ask(req: AskRequest) -> AskResponse:
    """Run the full agentic CRAG pipeline on a question."""
    try:
        result = run_pipeline(req.query, stream=req.stream)
    except EmptyCorpusError as exc:
        logger.warning("Ask failed (empty corpus): %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HealRAGError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    trace_id = save_trace(result.to_dict(), tag="ask")
    return AskResponse(
        query=result.query,
        answer=result.answer,
        steps=result.steps,
        corrections=result.corrections,
        rewritten_queries=result.rewritten_queries,
        confidence=result.confidence,
        hallucination_grade=result.hallucination_grade,
        retries=result.retries,
        accepted_chunks=result.accepted_chunks,
        trace_id=trace_id or None,
        error=result.error,
    )


@router.post("/ingest", response_model=IngestResponse, dependencies=[AuthDep])
async def ingest_upload(file: UploadFile = File(...)) -> IngestResponse:
    """Ingest a single uploaded document file."""
    ext = Path(file.filename or "upload.bin").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )
    tmp = Path(tempfile.gettempdir()) / f"healrag_upload_{file.filename}"
    with open(tmp, "wb") as fh:
        shutil.copyfileobj(file.file, fh)
    try:
        added = ingest_file(tmp)
    finally:
        tmp.unlink(missing_ok=True)
    store = get_vector_store()
    return IngestResponse(added=added, total_documents=store.count)


@router.get("/corpus", response_model=CorpusInfo, dependencies=[AuthDep])
async def corpus_info() -> CorpusInfo:
    store = get_vector_store()
    try:
        data = store._collection.get(include=["metadatas"], limit=100000)
    except Exception:
        data = {"metadatas": []}
    files = sorted(
        {m.get("source_file", "unknown") for m in data.get("metadatas", []) if m}
    )
    return CorpusInfo(total_chunks=store.count, source_files=files)