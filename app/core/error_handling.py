"""Centralized error handling, logging setup, and custom exceptions."""

from __future__ import annotations

import logging
import sys
import traceback
from typing import Any, Dict, Optional


class HealRAGError(Exception):
    """Base application exception."""


class ConfigError(HealRAGError):
    """Raised when the application is misconfigured."""


class OllamaUnavailableError(HealRAGError):
    """Raised when the Ollama server cannot be reached / model missing."""


class RetrievalError(HealRAGError):
    """Raised when a retrieval operation fails."""


class EmptyCorpusError(HealRAGError):
    """Raised when the vector store / index holds no documents."""


class GenerationError(HealRAGError):
    """Raised when answer generation fails."""


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure root + app logger with a consistent formatter."""
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(level.upper())
    # Avoid duplicate handlers on re-import.
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        root.addHandler(handler)

    return logging.getLogger("healrag")


def format_exception(exc: BaseException, include_trace: bool = True) -> str:
    """Friendly one-line description of an exception."""
    if include_trace:
        return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return f"{type(exc).__name__}: {exc}"


def to_dict(error: HealRAGError, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Serialize an error into a JSON-safe structure for API responses."""
    payload: Dict[str, Any] = {
        "error": type(error).__name__,
        "message": str(error),
    }
    if extra:
        payload.update(extra)
    return payload


logger = setup_logging()
