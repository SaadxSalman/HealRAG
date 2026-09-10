"""Ollama client wrapper.

Centralizes all calls to a local Ollama server: chat completion, JSON-mode
grading, embedding generation, and model/server health checks. All secrets and
URLs come from configuration; nothing sensitive is hard-coded.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import httpx

from app.config import settings
from app.core.error_handling import OllamaUnavailableError, logger


class OllamaClient:
    """Thin, dependency-light client for the Ollama HTTP API."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.timeout = timeout or settings.ollama_timeout
        self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout)

    # ------------------------------------------------------------------ #
    # Health / introspection
    # ------------------------------------------------------------------ #
    def is_healthy(self) -> bool:
        try:
            resp = self._client.get("/api/tags", timeout=10)
            return resp.status_code == 200
        except Exception as exc:  # pragma: no cover - network layer
            logger.warning("Ollama health check failed: %s", exc)
            return False

    def list_models(self) -> List[str]:
        try:
            resp = self._client.get("/api/tags", timeout=10)
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception as exc:
            raise OllamaUnavailableError(f"Cannot list Ollama models: {exc}") from exc

    def ensure_model(self, model: Optional[str] = None) -> None:
        """Verify the requested model is pullable (warn, do not crash)."""
        model = model or settings.ollama_chat_model
        try:
            available = self.list_models()
        except OllamaUnavailableError:
            return  # server down; the caller surfaces the error later
        base = model.split(":")[0]
        if base not in {m.split(":")[0] for m in available}:
            logger.warning(
                "Model '%s' not found locally. Run: ollama pull %s", model, model
            )

    # ------------------------------------------------------------------ #
    # Chat / generation
    # ------------------------------------------------------------------ #
    def chat(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 2048,
        format_json: bool = False,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Run a chat completion and return the text payload."""
        if messages is None:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": model or settings.ollama_chat_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature if temperature is not None else 0.2},
            "num_predict": max_tokens,
        }
        if format_json:
            payload["format"] = "json"

        try:
            resp = self._client.post("/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "").strip()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:400] if exc.response else str(exc)
            raise OllamaUnavailableError(
                f"Ollama returned HTTP {exc.response.status_code}: {detail}"
            ) from exc
        except httpx.HTTPError as exc:
            raise OllamaUnavailableError(
                f"Could not reach Ollama at {self.base_url}. Is it running? ({exc})"
            ) from exc

    def chat_json(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        """Run a chat completion and parse the JSON payload."""
        raw = self.chat(prompt, format_json=True, **kwargs)
        # Strip markdown fences that some models wrap around JSON.
        return _coerce_json(raw)

    # ------------------------------------------------------------------ #
    # Embeddings
    # ------------------------------------------------------------------ #
    def embed(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        """Embed one or more texts via the Ollama embeddings API."""
        if not texts:
            return []
        model = model or settings.ollama_embed_model
        vectors: List[List[float]] = []
        try:
            for text in texts:
                resp = self._client.post(
                    "/api/embed",
                    json={"model": model, "input": text},
                    timeout=max(60, self.timeout),
                )
                resp.raise_for_status()
                data = resp.json()
                emb = data.get("embedding")
                if emb is None:
                    # Some versions nest it under "embeddings"[0].
                    emb = data["embeddings"][0]
                vectors.append(emb)
        except Exception as exc:
            raise OllamaUnavailableError(f"Embedding failed via Ollama: {exc}") from exc
        return vectors

    def close(self) -> None:
        self._client.close()


def get_ollama_client() -> OllamaClient:
    """Return a process-wide OllamaClient singleton."""
    global _singleton
    if _singleton is None:
        _singleton = OllamaClient()
    return _singleton

# ---------------------------------------------------------------------- #
# JSON coercion helper
# ---------------------------------------------------------------------- #
def _coerce_json(raw: str) -> Dict[str, Any]:
    """Best-effort conversion from an LLM string to a Python dict."""
    if not raw:
        raise ValueError("Empty model output; expected JSON.")
    cleaned = raw.strip()
    # Strip ```json ... ``` fences
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
    if fenced:
        cleaned = fenced.group(1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Last resort: locate first { ... } balanced block.
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Model output is not valid JSON: {raw[:300]}...")

_singleton: Optional[OllamaClient] = None