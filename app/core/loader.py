"""Document text extraction and chunking.

Supports plain text, markdown, HTML, PDF and .docx. Returns a flat list of
``(text, metadata)`` chunks ready for embedding and storage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from app.config import settings
from app.core.error_handling import ConfigError, logger


def _chunk_text(text: str, size: int, overlap: int) -> List[str]:
    """Simple character-based chunking with an overlap window."""
    text = text.strip()
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        chunks.append(text[start:end].strip())
        if end >= n:
            break
        start = end - overlap
    return [c for c in chunks if c]


def extract_text(path: Path) -> str:
    """Extract raw text from a supported document file."""
    suffix = path.suffix.lower()
    data = path.read_bytes()

    if suffix in (".txt", ".md", ".markdown"):
        return data.decode("utf-8", errors="ignore")
    if suffix in (".html", ".htm"):
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(data.decode("utf-8", errors="ignore"), "html.parser")
        return soup.get_text(separator="\n")
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(path)
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    if suffix == ".docx":
        import docx2txt

        return docx2txt.process(str(path))
    raise ConfigError(f"Unsupported file type: {suffix}")


def load_document(path: Path) -> List[Tuple[str, Dict[str, str]]]:
    """Load a single document and return its text chunks with metadata."""
    if not path.exists():
        raise FileNotFoundError(path)
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > settings.max_file_size_mb:
        raise ConfigError(f"{path.name} exceeds max size {settings.max_file_size_mb} MB")

    text = extract_text(path)
    chunks = _chunk_text(text, settings.chunk_size, settings.chunk_overlap)
    logger.info("Loaded %s -> %d chunk(s)", path.name, len(chunks))
    return [
        (
            chunk,
            {
                "source_file": path.name,
                "chunk_index": str(i),
                "path": str(path),
            },
        )
        for i, chunk in enumerate(chunks)
    ]


def load_directory(directory: Path) -> List[Tuple[str, Dict[str, str]]]:
    """Load every supported file in a directory (non-recursive by default)."""
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
    supported = {".txt", ".md", ".markdown", ".html", ".htm", ".pdf", ".docx"}
    results: List[Tuple[str, Dict[str, str]]] = []
    for f in sorted(directory.iterdir()):
        if f.is_file() and f.suffix.lower() in supported:
            try:
                results.extend(load_document(f))
            except Exception as exc:
                logger.warning("Skipping %s: %s", f.name, exc)
    return results