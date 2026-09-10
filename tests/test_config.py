"""Unit tests for configuration loading."""

from __future__ import annotations

from app.config import Settings


def test_defaults():
    s = Settings(_env_file=None)
    assert s.app_name == "HealRAG"
    assert s.relevance_threshold == 0.5
    assert s.max_query_rewrites == 2
    assert s.max_hallucination_retries == 2
    assert s.retrieval_k == 6
    assert s.bm25_k == 4


def test_cors_parser_accepts_two_shapes():
    s = Settings(_env_file=None, cors_origins='["http://a", "http://b"]')
    assert s.cors_origins == ["http://a", "http://b"]

    s2 = Settings(_env_file=None, cors_origins="http://a, http://b")
    assert s2.cors_origins == ["http://a", "http://b"]


def test_chroma_path_resolves_to_project():
    from app.config import PROJECT_ROOT, settings

    assert settings.chroma_path != PROJECT_ROOT  # tests point it at tmp


def test_secrets_templated_not_committed():
    """Real secrets must live in .env only (never committed to git)."""
    import subprocess

    from app.config import PROJECT_ROOT

    ignores = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in ignores
    assert (PROJECT_ROOT / ".env.example").exists()  # template present

    # Git must both ignore .env and never track it, even if it exists locally.
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        ).stdout

    assert git("check-ignore", ".env").strip() == ".env"      # ignored
    assert git("ls-files").splitlines().count(".env") == 0    # never tracked