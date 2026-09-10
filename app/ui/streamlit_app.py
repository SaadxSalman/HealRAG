"""HealRAG — Streamlit chat UI.

Connects to the FastAPI backend, streams the CRAG pipeline answer, and renders
a fully transparent trace: retrieval grades, rewrite corrections, hallucination
checks, accepted chunks and confidence.
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

# --------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------- #
st.set_page_config(
    page_title="HealRAG — Corrective RAG",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = os.getenv("STREAMLIT_API_URL", "http://localhost:8000")
API_URL = os.getenv("API_URL", API_URL)
BEARER = os.getenv("API_BEARER_TOKEN", "").strip()


def _headers():
    h = {"Accept": "application/json"}
    if BEARER:
        h["Authorization"] = f"Bearer {BEARER}"
    return h


def _api(verb: str, path: str, **kwargs):
    kwargs.setdefault("headers", _headers())
    kwargs.setdefault("timeout", 300)
    return getattr(httpx, verb)(f"{API_URL}{path}", **kwargs)


# --------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------- #
with st.sidebar:
    st.title("🩺 HealRAG")
    st.caption("Agentic Self-Correction CRAG Pipeline")
    st.divider()

    health = None
    try:
        health = _api("get", "/health")
    except Exception:
        pass

    if health and health.status_code == 200:
        data = health.json()
        st.success(f"API: online · {data.get('documents_indexed', 0)} chunks indexed")
        if data.get("ollama_connected"):
            st.info("Ollama: connected")
            st.write("Models: " + ", ".join(data.get("ollama_models", [])[:4]) or "n/a")
        else:
            st.warning("Ollama: unreachable — grading will fail closed.")
    else:
        st.error("API unreachable. Start it with `python scripts/launch.py`.")

    st.divider()
    with st.expander("⚙️ Settings"):
        st.write("Retrieval K:", os.getenv("RETRIEVAL_K", "6 (env)"))
        st.write("Threshold: relevance grade decides accept/reject per chunk.")
        st.write("BM25 fallback active after first low-grade pass.")

    uploaded = st.file_uploader(
        "📄 Ingest a document",
        type=["txt", "md", "pdf", "docx", "html"],
        accept_multiple_files=False,
    )
    if uploaded is not None:
        with st.spinner("Ingesting..."):
            try:
                resp = _api(
                    "post",
                    "/ingest",
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                )
                if resp.status_code == 200:
                    st.success(f"Ingested: +{resp.json()['added']} chunks")
                else:
                    st.error(resp.text[:300])
            except Exception as exc:
                st.error(str(exc))


# --------------------------------------------------------------------- #
# Main chat area
# --------------------------------------------------------------------- #
st.title("Ask the Knowledge Base")
st.caption("Every answer is grounded, graded, and self-corrected before display.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a health/knowledge question…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("_Running the CRAG pipeline…_")
        try:
            resp = _api("post", "/ask", json={"query": prompt, "stream": False})
        except Exception as exc:
            placeholder.error(f"Backend error: {exc}")
            st.session_state.messages.append({"role": "assistant", "content": f"Error: {exc}"})
            st.stop()

        if resp.status_code != 200:
            placeholder.error(resp.text[:400])
            st.stop()

        data = resp.json()
        if data.get("error"):
            placeholder.error(f"Pipeline error: {data['error']}")
        else:
            placeholder.markdown(data["answer"])

            # ---- Transparency panel ----
            with st.expander("🔍 View pipeline trace", expanded=False):
                t1, t2, t3, t4 = st.columns(4)
                t1.metric("Confidence", f"{data.get('confidence') or 0:.0%}")
                t2.metric("Hallucination grade", f"{data.get('hallucination_grade') or 0:.0%}")
                t3.metric("Regeneration retries", data.get("retries", 0))
                t4.metric("Rewrite corrections", len(data.get("rewritten_queries", [])))

                st.subheader("Steps executed")
                for step in data.get("steps", []):
                    st.markdown(f"- `{step}`")

                if data.get("rewritten_queries"):
                    st.subheader("✏️ Query rewrites")
                    for rw in data["rewritten_queries"]:
                        st.markdown(f"- **{rw}**")

                if data.get("corrections"):
                    st.subheader("🛠 Corrections applied")
                    for c in data["corrections"]:
                        st.markdown(f"- {c}")

                st.subheader("📚 Accepted context chunks")
                chunks = data.get("accepted_chunks", [])
                if not chunks:
                    st.info("No chunks passed relevance grading.")
                for i, c in enumerate(chunks, 1):
                    with st.container(border=True):
                        cols = st.columns([4, 1, 1])
                        cols[0].markdown(c.get("text", ""))
                        cols[1].markdown(f"**relevance**\n\n{c.get('relevance'):.0%}")
                        src = c.get("metadata", {}).get("source_file", c.get("source", ""))
                        cols[2].markdown(f"**source**\n\n`{src}`")

            st.session_state.messages.append({"role": "assistant", "content": data["answer"]})

st.divider()
st.caption(
    "💡 HealRAG uses a local Ollama SLM to grade retrieval, rewrite weak queries, "
    "fall back to BM25, and reject hallucinated answers with an expanded re-retrieval loop."
)