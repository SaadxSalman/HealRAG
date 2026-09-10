"""Trace persistence: dump every pipeline run to the trace directory as JSON."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict

from app.config import settings
from app.core.error_handling import logger


def save_trace(payload: Dict[str, Any], tag: str = "query") -> str:
    """Persist a trace document; returns the trace id."""
    try:
        trace_dir = Path(settings.trace_path)
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_id = f"{tag}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        target = trace_dir / f"{trace_id}.json"
        with open(target, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
        logger.info("Trace written: %s", target)
        return trace_id
    except Exception as exc:
        logger.warning("Could not write trace: %s", exc)
        return ""


def list_traces(limit: int = 20) -> list:
    trace_dir = Path(settings.trace_path)
    if not trace_dir.exists():
        return []
    files = sorted(trace_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.stem for p in files[:limit]]