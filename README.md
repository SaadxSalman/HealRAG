<p align="center">
  <img src="https://img.shields.io/badge/LangGraph-Evaluator--Optimizer-blueviolet" alt="LangGraph"/>
  <img src="https://img.shields.io/badge/Ollama-qwen3%3A8b%20%7C%20phi--4--mini-orange" alt="Ollama"/>
  <img src="https://img.shields.io/badge/Vector%20Store-ChromaDB-green" alt="ChromaDB"/>
  <img src="https://img.shields.io/badge/API-FastAPI-teal" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/UI-Streamlit-red" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/License-MIT-blue" alt="License"/>
</p>

# 🩺 HealRAG — Agentic Self-Correction CRAG (Corrective RAG) Pipeline

A **complete, local-first, full-stack Retrieval-Augmented Generation (RAG) system**
with an agentic *evaluator–optimizer* architecture. Instead of blindly trusting a
vector search, HealRAG **grades every retrieved chunk** with a fast local SLM, and
when relevance fails, it **automatically rewrites the query** and **falls back to a
BM25 keyword index**. On the way out, a dedicated **hallucination-detection loop**
rejects ungrounded answers, forces an *expanded re-retrieval pass*, and dynamically
**refines the final output** before it is ever shown to a user.

```
 User query ──▶ retrieve ──▶ grade ──▶ accepted? ──▶ generate ──▶ ground? ──▶ answer
                                   │                   │ (hallucination loop)
                                   ▼                   ▼
                            rewrite + BM25          re-retrieval + refine
                                   │                   │
                                   └──────────▶ (loop up to N times)
```

Everything runs **on your machine**: no cloud LLM API keys, no data leaves your
computer. The entire stack is orchestrated with **LangGraph** (stateful, cyclic,
inspectable routing), embeddings and grading via **Ollama**, persistence in
**ChromaDB**, an API served by **FastAPI**, and an interactive **Streamlit** UI.

---

## Table of Contents

1. [Why corrective RAG?](#-why-corrective-rag)
2. [Feature highlights](#-feature-highlights)
3. [Tech stack](#-tech-stack)
4. [How the pipeline works (architecture deep-dive)](#-how-the-pipeline-works)
5. [Project structure](#-project-structure)
6. [Prerequisites](#-prerequisites)
7. [Installation](#-installation)
8. [Configuration (.env)](#-configuration-env)
9. [Seed the knowledge base](#-seed-the-knowledge-base)
10. [Run the full stack](#-run-the-full-stack)
11. [FastAPI reference](#-fastapi-reference)
12. [Streamlit UI guide](#-streamlit-ui-guide)
13. [Inside the LangGraph state machine](#-inside-the-langgraph-state-machine)
14. [Step-by-step: a corrective run](#-step-by-step-a-corrective-run)
15. [Hallucination detection & refinement](#-hallucination-detection--refinement)
16. [Confidence scoring](#-confidence-scoring)
17. [Testing](#-testing)
18. [Docker (optional)](#-docker-optional)
19. [Extending the project](#-extending-the-project)
20. [Security & secrets](#-security--secrets)
21. [Troubleshooting](#-troubleshooting)
22. [Roadmap](#-roadmap)
23. [License](#-license)

---

## 🎯 Why corrective RAG?

Classic RAG has a fundamental blind spot: **it trusts retrieval scores blindly.**
A dense vector store can return highly-similar-looking chunks that are actually
irrelevant to the *intent* of the question, or the query words can be missing from
the embedding space entirely (rare terms, typos, domain jargon). When retrieval is
bad, the generator has only two options: **hallucinate** or **apologize**.

The production-funded answer is **Corrective RAG (CRAG)** — an agentic pattern in
which:

> *"Retrieval is not a one-shot step. It is a hypothesis that gets evaluated, and
> if the hypothesis is weak, the system corrects its own retrieval."*

HealRAG implements this with an **evaluator–optimizer** loop (the same pattern
Anthropic highlights for self-correcting agents):

| Phase | Classic RAG | HealRAG (CRAG) |
|-------|-------------|----------------|
| Retrieve | top-k by cosine similarity once | dense retrieval, then *SLM grading* |
| Weak retrieval | (silently) generate garbage | **rewrite query + BM25 fallback**, re-grade |
| Generate | one shot, unverified | grounded generation with source citations |
| Verify | nothing | **hallucination grader** gates the answer |
---

## ✨ Feature highlights

- **Agentic relevance grading** — every retrieved chunk is scored `0..10` by a
  local SLM in **strict JSON mode**; only chunks that pass the configured
  threshold are allowed into the generator's context.
- **Self-correcting retrieval** — when too many chunks are rejected, the pipeline:
  1. **rewrites** the query into a sharper, search-friendly form, and
  2. **falls back to a local BM25 keyword index** (`rank-bm25`) so lexical signals
     that the embedding model missed are still surfaced.
- **Hallucination detection loop** — a second grader verifies each generated
  answer against the accepted context, lists *unsupported claims verbatim*, and
  rejects failures. Rejected answers trigger an **expanded hybrid re-retrieval**
  and a **refinement-guided regeneration** (with the unsupported claims fed back
  into the generator). Bounded by `MAX_HALLUCINATION_RETRIES`.
- **Bounded loops** — every correction loop is capped (`MAX_QUERY_REWRITES`,
  `MAX_HALLUCINATION_RETRIES`) so the pipeline degrades gracefully instead of
  spinning forever.
- **Full transparency** — every run records: steps taken, rewritten queries,
  corrections applied, per-chunk relevance grades, hallucination verdicts,
  confidence, and attempt history. The UI shows it all; traces are persisted to
  `./traces/` as JSON for offline analysis.
- **Local-first & private** — all inference through **Ollama**, all vectors in
  **ChromaDB** on disk, no cloud calls, no API keys required.
- **Ready-to-run demo corpus** — bundled health-domain Markdown docs
  (hypertension, type 2 diabetes, sleep) so you can be productive in minutes.
- **Extensible ingestion** — `.txt`, `.md`, `.html`, `.pdf`, `.docx` via a
  unified loader, with chunk-size/overlap tuning from `.env`.

---

## 🧱 Tech stack

| Layer | Technology | Role |
|-------|-----------|------|
| Orchestration | **LangGraph** | Stateful, cyclic graph; routing decisions between retrieve → grade → rewrite → generate → verify |
| Local LLM / SLM | **Ollama** (`qwen3:8b`, `phi-4-mini`) | Relevance grading, query rewriting, generation, hallucination checking |
| Embeddings | **Ollama** `nomic-embed-text` (or `sentence-transformers`) | Dense vectorization of queries + chunks |
| Vector store | **ChromaDB** (persistent) | Local dense retrieval with metadata filtering |
| Lexical fallback | **rank-bm25** | Keyword index used in corrective steps |
| Backend API | **FastAPI** + Uvicorn | REST endpoints for ask / ingest / health / corpus |
| Frontend UI | **Streamlit** | Chat interface with trace transparency panel |
| Config/secrets | `pydantic-settings` + `.env` | All tunable knobs and bearer-token auth |
| Tests | `pytest` | Unit coverage for chunking, BM25, routing logic, config |

A note on the SLMs: `qwen3:8b` is the default grader/generator because it is
fast, small enough to run on a laptop, and produces reliable structured JSON.
`phi-4-mini` is an excellent drop-in alternative for *faster* grading if your
hardware prefers a lighter model — switch it in via `.env`.

---

## 🔬 How the pipeline works

```
                    ┌──────────────────────────  CORRECTIVE LOOP  ──────────────────────────┐
                    │                                                                      │
  START ──▶ RETRIEVE ──▶ GRADE ──▶ enough accepted? ── no ──▶ REWRITE ──▶ BM25/BLEND ──┐    │
  (dense top-k)       (SLM scores   │                      query            ──▶ GRADE ──┘    │
                       each chunk)  │ yes                                                │
                                    ▼                                                   │
                                 GENERATE ◀──────────────────────────────────────────────┘
                                    │  answer + citations [1][2]…                     (rewrite budget
                                    ▼                                                   exhausted → force)
                              HALLUCINATION CHECK
                                    │
                    ┌───── pass? ───┐
                    │ no            │ yes
                    ▼               ▼
         CORRECTIVE RE-RETRIEVAL   END
         (broader hybrid, +2 K)
                    │
                    └────▶ GENERATE (refinement feedback) ── retries < MAX?
```

### Layer 1 — Retrieval (`app/agents/retriever.py`, `app/core/vector_store.py`)
Queries are embedded with **nomic-embed-text** and matched against the persistent
**ChromaDB** collection (cosine distance → similarity). The BM25 keyword index is
rebuilt from the same corpus each boot/ingest so the corrective fallback always
sees identical content.

### Layer 2 — Relevance evaluation (`app/agents/grader.py`)
A **`RelevanceGrader`** asks the SLM to grade *each chunk independently*:

```json
{"score": 8, "verdict": "accept", "reason": "Directly answers dosage question."}
```

The grader runs at **temperature 0** with Ollama's `format=json`, so routing
becomes deterministic and cheap. `accepted = score ≥ threshold`.

### Layer 3 — Corrective steering (`app/agents/rewriter.py`)
If the accepted set is empty (or below `MIN_ACCEPTED`), LangGraph routes to
**`rewrite`**: the `QueryRewriter` produces a sharp, noun-heavy search query and
the retriever hits the **BM25 keyword index** (blended with vectors when BM25 is
thin). The new chunks are graded again. This cycle is bounded by
### Layer 4 — Grounded generation (`app/agents/generator.py`)
The generator receives **only accepted chunks** formatted as a numbered context
block with source labels. The system prompt forbids outside knowledge, demands
inline citations `[n]`, and requires an explicit *"I don't have enough
information"* when the context is insufficient.

### Layer 5 — Hallucination detection (`app/agents/grader.py`)
A **`HallucinationGrader`** re-reads the generated answer against the accepted
context and returns:

```json
{"score": 4, "verdict": "fail",
 "unsupported_claims": ["Anemia is treated with aspirin daily."],
 "reason": "Claim not present in any context passage."}
```

### Layer 6 — Correction on failure (`app/agents/graph.py`)
On **fail**, `corrective_rerefetch` runs an **expanded hybrid retrieval**
(larger `K` from both dense + BM25), feeds the fresh chunks into the state, and
**regenerates** with the unsupported claims passed back as refinement feedback.
Bounded by `MAX_HALLUCINATION_RETRIES`; the best attempt is returned with a
flagging note in corrections when it still fails.

---

## 📁 Project structure

```
HealRAG/
├── .env.example               # Template for all configuration (copy → .env)
├── .gitignore                 # Ignores .env, chroma_db, venvs, traces, caches
├── requirements.txt           # Python dependencies
├── Makefile                   # install / seed / ingest / run-api / run-ui / test
├── README.md                  # You are here
│
├── app/
│   ├── __init__.py
│   ├── config.py              # Pydantic-settings; loads .env into Settings
│   ├── core/
│   │   ├── bm25_index.py      # BM25Okapi keyword fallback index
│   │   ├── embeddings.py      # Ollama / sentence-transformers backends
│   │   ├── error_handling.py  # Logging + typed exceptions
│   │   ├── ingestion.py       # Orchestrates file → chunks → store
│   │   ├── llm.py             # Ollama HTTP client (chat, JSON, embeddings)
│   │   ├── loader.py          # txt/md/pdf/docx/html extraction + chunking
│   │   ├── tracing.py         # Trace persistence to ./traces
│   │   └── vector_store.py    # ChromaDB persistent store + query
│   ├── agents/
│   │   ├── grader.py          # RelevanceGrader + HallucinationGrader
│   │   ├── generator.py       # Grounded, citation-aware answer generator
│   │   ├── graph.py           # ★ LangGraph state machine (the orchestration)
│   │   ├── retriever.py       # Vector + BM25 + hybrid retrieval
│   │   ├── rewriter.py        # Query rewriting agent
│   │   └── types.py           # Chunk / PipelineResult data classes
│   ├── api/
│   │   ├── main.py            # FastAPI app (startup sync, CORS, docs)
│   │   ├── routes.py          # /health /ask /ingest /corpus
│   │   ├── schemas.py         # Pydantic request/response models
│   │   └── deps.py            # Bearer-token auth + exception handlers
│   ├── scripts/
│   │   ├── seed_data.py       # Loads ./data/documents demo corpus
│   │   └── ingest.py          # CLI ingest of your own files
│   └── ui/
│       └── streamlit_app.py   # Chat UI with trace transparency panel
│
├── scripts/
│   └── launch.py              # One-command API + UI launcher
├── data/
│   └── documents/             # Bundled demo docs (hypertension, diabetes, sleep)
├── tests/                     # pytest suite (BM25, loader, routing, config)
---

## ✅ Prerequisites

Before anything else, make sure you have:

1. **Python 3.11+** (3.13 recommended; the project is tested on Windows + POSIX).
2. **Ollama** installed and running:
   - Install from <https://ollama.com/download>
   - Start the service (on Windows it runs in the tray; on macOS/Linux `ollama serve`)
3. The required **Ollama models** pulled (one command each):

```bash
# Grader / generator (7-8B, fast & reliable JSON output)
ollama pull qwen3:8b

# Alternative, even faster grading model (optional)
ollama pull phi4-mini

# Embedding model for the vector store
ollama pull nomic-embed-text
```

> The app tries to be helpful here too: on `/health` it reports exactly which
> models Ollama has, and startup logs a warning for missing models.

---

## ⚙️ Installation

```bash
# 1) Clone the repo
git clone https://github.com/SaadxSalman/HealRAG.git
cd HealRAG

# 2) Create a virtual environment (Windows / POSIX)
python -m venv .venv
.venv\Scripts\activate        # Windows PowerShell
source .venv/bin/activate     # macOS / Linux

# 3) Install dependencies
pip install -r requirements.txt

# 4) Create your environment file from the template
cp .env.example .env          # Windows
copy .env.example .env        # (PowerShell alternative)

# 5) (Recommended) verify everything is wired up
python -m pytest tests/ -v
```

Optionally, if you prefer the pure-Python `sentence-transformers` embedding
backend instead of Ollama embeddings (e.g. for higher quality `all-MiniLM-L6-v2`),
install the extra and switch `EMBEDDING_BACKEND=sentence_transformers` in `.env`:

```bash
```

---

## 🔑 Configuration (.env)

All knobs — including **every secret** — live in environment variables loaded
from `.env` (which is **gitignored**; never commit it). Copy `.env.example` to
`.env` and adjust:

| Variable | Default | Meaning |
|----------|---------|---------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server endpoint |
| `OLLAMA_CHAT_MODEL` | `qwen3:8b` | SLM used for relevance & hallucination grading |
| `OLLAMA_GENERATION_MODEL` | `qwen3:8b` | Model for final answer generation |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model (dense vectors) |
| `EMBEDDING_BACKEND` | `ollama` | `ollama` or `sentence_transformers` |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Where ChromaDB persists on disk |
| `CHROMA_COLLECTION_NAME` | `healrag_documents` | Collection name |
| `RETRIEVAL_K` | `6` | Dense top-k per query |
| `BM25_K` | `4` | BM25 fallback top-k |
| `RELEVANCE_THRESHOLD` | `0.5` | Relevance score gate (0..1) |
| `MAX_QUERY_REWRITES` | `2` | Max query-rewrite correction cycles |
| `MAX_HALLUCINATION_RETRIES` | `2` | Max hallucination-loop cycles |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `120` | Ingestion chunking |
| `GENERATION_TEMPERATURE` | `0.4` | Creativity at generation time |
| `GRADER_TEMPERATURE` | `0.0` | Deterministic grading |
| `API_HOST` / `API_PORT` | `0.0.0.0` / `8000` | FastAPI bind address |
| `API_BEARER_TOKEN` | _(empty)_ | If set, every API call needs `Authorization: Bearer <token>` |
| `CORS_ORIGINS` | `["http://localhost:8501"]` | Allowed UI origins |
| `TRACE_DIR` | `./traces` | Per-query trace JSON output |

**Secrets in the environment** (this is the only place credentials belong):

```
API_BEARER_TOKEN=change-me-to-a-long-random-string
```

> Using `API_BEARER_TOKEN` is recommended even for local stacks if you bind the
> API to a non-loopback interface. The Streamlit UI forwards it automatically
> when the `API_BEARER_TOKEN` env var is exported to it as well.

---

## 🌱 Seed the knowledge base

The repo ships with a small, sensible demo corpus in `data/documents/`
(hypertension, type-2 diabetes, and healthy sleep guides) so you can click
through the whole experience immediately:

```bash
python -m app.scripts.seed_data
```

You should see something like:

```
INFO  Seeding corpus from …\data\documents
INFO  Loaded hypertension.md → 7 chunk(s)
INFO  Loaded diabetes.md → 9 chunk(s)
INFO  Loaded sleep.md → 8 chunk(s)
INFO  BM25 index rebuilt with 24 document(s)
INFO  Done. Added 24 chunk(s); vector store now holds 24 document(s).
```

To ingest **your own** documents instead, drop files (`.txt`, `.md`, `.pdf`,
`.docx`, `.html`) into `data/documents/` and run:

```bash
python -m app.scripts.ingest --dir ./data/documents
# or a single file, or wipe first:
python -m app.scripts.ingest --file ./my_doc.pdf --reset
```

The BM25 keyword index is automatically kept in sync with the vector store
after every ingestion, so the corrective fallback is always up to date.

---

## 🚀 Run the full stack

The launcher starts **both** the FastAPI backend and the Streamlit UI, and it
auto-seeds the demo corpus on first boot if the store is empty:

```bash
python scripts/launch.py
```

Or start them separately (e.g. for development with auto-reload):

```terminal
# Terminal 1 — API
uvicorn app.api.main:app --reload --port 8000

# Terminal 2 — UI
streamlit run app/ui/streamlit_app.py
```

What you get:

| Service | URL |
|---------|-----|
| Streamlit UI | <http://localhost:8501> |
| FastAPI docs (Swagger) | <http://localhost:8000/docs> |
| Health check | <http://localhost:8000/health> |

A quick smoke test via curl/powershell after startup:

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the target blood pressure for most adults?"}'
```

---

## 🔌 FastAPI reference

Interactive docs are served at `/docs` (Swagger UI) and `/redoc`. All endpoints
except `/` are protected by optional bearer auth.

### `GET /health`
Server + Ollama + corpus status.

```json
{
  "status": "ok",
  "app": "HealRAG",
  "version": "1.0.0",
  "ollama_connected": true,
  "ollama_models": ["nomic-embed-text:latest", "qwen3:8b"],
  "documents_indexed": 24
}
```

### `POST /ask`

The hero endpoint — runs the entire agentic CRAG pipeline.

```json
// Request
{ "query": "What is stage 1 hypertension?", "stream": false }

// Response (abridged)
{
  "query": "What is stage 1 hypertension?",
  "answer": "Stage 1 hypertension is systolic 130-139 or diastolic 80-89 [1]. ...",
  "steps": ["retrieve", "grade", "generate", "hallucination_check"],
  "corrections": [],
  "rewritten_queries": [],
  "confidence": 0.87,
  "hallucination_grade": 0.9,
  "retries": 0,
  "accepted_chunks": [
    {
      "id": "…",
      "text": "Stage 1 Hypertension: systolic 130-139 or diastolic 80-89…",
      "metadata": {"source_file": "hypertension.md", "chunk_index": "1"},
      "score": 0.78,
      "source": "vector",
      "relevance": 0.9,
      "verdict": "accept",
      "reason": "Directly defines stage 1 hypertension.",
      "accepted": true
    }
  ],
  "trace_id": "ask-20260225-…",
  "error": null
}
```

When corrections fire, `steps` grows and `corrections` documents each fix:

```json
{
  "steps": ["retrieve", "grade", "rewrite", "grade", "generate", "hallucination_check"],
  "rewritten_queries": ["sodium restriction hypertension diet"],
  "corrections": [
    "query rewritten (1): 'how do i cut salt for bp' -> 'sodium restriction hypertension diet' (BM25 fallback)"
  ]
}
```

### `POST /ingest`
Multipart file upload (`file` field). Supports `.txt/.md/.pdf/.docx/.html`.

```bash
curl -X POST http://localhost:8000/ingest -F "file=@./my_guide.pdf"
# → {"added": 41, "total_documents": 65}
```

### `GET /corpus`
Summary of what is indexed, grouped by source file.

---

## 🖥 Streamlit UI guide

The UI needs to reach the backend at `http://localhost:8000` (configurable via
`STREAMLIT_API_URL` or `API_URL`). Features:

- **Chat** — ask anything; answers render with the pipeline's transparency panel.
- **Pipeline trace** (`🔍 View pipeline trace`) — expandable panel showing:
  - `Confidence`, `Hallucination grade`, `Regeneration retries`, `Rewrite corrections`
  - every graph `steps` executed (`retrieve`, `grade`, `rewrite`, …)
  - the rewritten query strings
  - each correction message (query rewrite + BM25 fallback, re-retrieval passes)
  - every **accepted chunk** with its relevance %, source file, and text
- **Sidebar health card** — live API/Ollama status and chunk count.
- **One-click ingestion** — upload a `.pdf/.md/.txt/.docx/.html` straight from
  the sidebar; it is chunked, embedded, indexed, and the BM25 index synced.

### Example session transcript

> **You:** What are the stages of high blood pressure?
>
> **HealRAG:** Blood pressure is classified as normal (<120/<80), elevated
> (120–129/<80), Stage 1 (130–139 or 80–89), Stage 2 (≥140 or ≥90), and
> hypertensive crisis (≥180 and/or ≥120) [1]. The first line of management is
> lifestyle change: sodium under 2,300 mg/day, the DASH diet, and regular
> aerobic exercise [2].
>
> **Trace:** steps `[retrieve, grade, generate, hallucination_check]` ·
> confidence `92%` · hallucination grade `95%` · 3 accepted chunks (relevance
---

## 🧠 Inside the LangGraph state machine

`app/agents/graph.py` defines a `GraphState` (a `TypedDict`) carrying the query,
retrieval, grades, corrections, attempts, and trace history. LangGraph makes the
*cycles* first-class: conditional edges with **pure routing functions** mean every
decision is unit-testable.

**Nodes**

| Node | Job |
|------|-----|
| `retrieve` | Dense vector top-k (starts on the active/original query); BM25 if the store is empty |
| `grade` | `RelevanceGrader` scores every chunk; `accepted` (verdict accept) chunks are recorded |
| `rewrite` | `QueryRewriter` produces a new query, then BM25 retrieval (+ vector blend), then re-grades |
| `generate` | Grounded generation from accepted context, with optional refinement feedback |
| `hallucination` | `HallucinationGrader.verify()` against accepted context |
| `corrective_rerefetch` | Expanded hybrid retrieval (`K+2` both sides); new chunks merged into state |

**Routing functions**

| Function | Decision |
|----------|----------|
| `route_after_grade` | accepted enough → `generate`; else → `rewrite`; on error → `end_with_error` |
| `route_after_rewrite` | accepted now, or rewrite budget spent → `generate`; else loop → `rewrite` |
| `route_after_hallucination` | `fail` & retries left → `corrective_rerefetch`; otherwise → `END` |

Because routing is pure and no node knows about the others, you can test each
leg in isolation (see `tests/test_agents.py`) and even render the graph.

**Key design choices**

- **Caps everywhere.** Loops are bounded in *both* directions: rewriting can
  only happen `MAX_QUERY_REWRITES` times; hallucination repairs at most
  `MAX_HALLUCINATION_RETRIES`. Costs stay predictable and worst-case latency is
  finite.
- **Fail closed.** Grading or generation errors never silently pass: failed
  grades become `reject`, an unverifiable hallucination check becomes `fail`.
- **State-first.** The full dialogue of a run — original query → rewrites →
  chunks → grades → attempts — is stored in state and dumped to `./traces/`
  ---

## 🧪 Step-by-step: a corrective run

Imagine a user asks: **"how do I lower my salt for bp issues"**

1. **`retrieve`** — embeds the query and pulls 6 dense chunks. Many hits come
   back with low similarity and mixed topical relevance.
2. **`grade`** — the SLM grades all 6. Only 0 pass the strict relevance bar
   (chunks talk about *monitoring home BP* rather than *dietary sodium*).
3. **route_after_grade → `rewrite`** — the `QueryRewriter` outputs
   `"sodium restriction hypertension diet"`. The retriever now queries the
   **BM25 keyword index** (plus a vector blend), which excellently matches
   "sodium → 2,300 mg per day, DASH diet, reduce salt".
4. **`grade`** — the new chunks pass, `accepted_chunks` populated, corrections
   log: *"query rewritten (1): '…' → '…' (BM25 fallback)"*.
5. **`generate`** — the generator produces a grounded answer with citations,
   using only the accepted diet chunks.
6. **`hallucination`** — the fact-checker verifies every claim against the
   context. Claims like *"sodium under 2,300 mg/day is recommended"* are
   confirmed; anything speculative (e.g. an invented supplement) gets flagged.
7. Verdict `pass` → `END`, confidence computed, trace persisted, answer shown.

If step 6 had **failed**, the graph would have:

- run `corrective_rerefetch` → retrieved a *broader* hybrid set (K+2 dense and
  BM25), merged fresh chunks into state;
- regenerated with the refinement prompt carrying the unsupported claims;
- re-checked; repeated up to `MAX_HALLUCINATION_RETRIES` times; and
- returned the last attempt with `corrections` noting exactly what happened.

---

## 🚫 Hallucination detection & refinement

This loop is the difference between a *demo* RAG and a *usable* RAG. Details:

**Detection prompt** asks the SLM to act as a strict fact-checker: given the
context passages and the answer, list **unsupported claims verbatim** and decide
`pass`/`fail` with a 0–10 grounding score.

**Refinement prompt** then feeds the generator:

```
A previous answer had these unsupported claims:
- Anemia is treated with aspirin daily.
Produce a corrected, fully-grounded answer… remove or flag the claim.
```

Why this works: passing the *actual unsupported text* back — rather than just a
`"you failed"` — gives the model concrete targets to remove or source, which
measurably reduces hallucination persistence across re-generation loops.

**Design choice:** after the retry budget is exhausted the *last* attempt is
returned, not dropped — but the UI surfaces `retries > 0` and the corrections
---

## 📊 Confidence scoring

HealRAG exposes a single `confidence` (0..1) per answer, computed as a blend of
*how well the context supports the answer* and *how grounded the text is*:

```
confidence = 0.5 × mean(relevance of accepted chunks) + 0.5 × hallucination_grade
```

It is deliberately transparent: you can reconstruct either half from the trace.
When the hallucination loop had to repair the answer, `retries` and
`corrections` tell the story; a low `hallucination_grade` with high retries is
a strong signal to expand the corpus.

---

## 🧪 Testing

The suite runs on pure Python with **no Ollama required** (LLM calls are mocked
by design — grading logic degrades gracefully offline):

```bash
python -m pytest tests/ -v
```

Coverage includes:

- **BM25 index** — tokenization, stopword removal, ranking, empty-state errors.
- **Document loading** — chunk overlap, metadata propagation, unsupported types.
- **Graph routing** — every conditional edge and its budget caps.
- **Config/secrets hygiene** — defaults, CORS parsing, and the guarantee that
  no real `.env` is committed.

---

## 🐳 Docker (optional)

A lean deployment profile can run the API (UI optional) in a container talking
to Ollama on the host:

```dockerfile
# Dockerfile (add yours)
FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t healrag .
docker run -p 8000:8000 -v ./chroma_db:/app/chroma_db \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 healrag
```

---

## 🧩 Extending the project

The architecture is intentionally swappable at every seam:

| Swap | Where |
|------|-------|
| Grading/generation model | `.env` → `OLLAMA_CHAT_MODEL`, `OLLAMA_GENERATION_MODEL` |
| Embedding source | `EMBEDDING_BACKEND` (`ollama` ↔ `sentence_transformers`) |
| Vector store | `app/core/vector_store.py` (swap ChromaDB for e.g. Qdrant/LanceDB) |
| Chunking policy | `app/core/loader.py::_chunk_text` (or plug a recursive splitter) |
| Grading prompt/style | `app/agents/grader.py` system prompts |
| Routing behavior | `app/agents/graph.py` — add nodes + conditional edges |
| Guardrails / PII filters | insert new LangGraph nodes between `generate` and `hallucination` |
| Streaming answers | `run_pipeline(..., stream=True)` + SSE endpoint in `routes.py` |

Example — add a **citation verifier** node in 10 lines:

```python
def _citation_node(state):
    # parse [n] references from answer; confirm n in accepted_chunks
    return {"citations_ok": ...}

---

## 🔒 Security & secrets

- **Everything sensitive lives in `.env`**, which is `.gitignore`d and never
  committed. `.env.example` is the committed *template*.
- Optional **bearer-token authentication**: set `API_BEARER_TOKEN` and every
  API route (except `/`) requires `Authorization: Bearer <token>`.
  The Streamlit UI forwards it via the `API_BEARER_TOKEN` env var.
- **`chroma_db/`, `traces/`, `.venv/`, `__pycache__/`, and DB files are
  gitignored** — you never accidentally push indexes, fills, or Python caches.
- **Local by design.** No external API keys, no telemetry, no data egress;
  Ollama + ChromaDB run entirely on your machine. Pair with a firewall rule
  (`API_HOST=127.0.0.1`) when exposing the API.
- Inputs are size-capped (`MAX_FILE_SIZE_MB`) and restricted to whitelisted
  file extensions at both the CLI and API boundaries.

---

## 🛠 Troubleshooting

| Symptom | Cause / Fix |
|---------|-------------|
| `Could not reach Ollama at http://localhost:11434` | Ollama isn't running. Start it (Windows tray icon / `ollama serve`). |
| `Model 'qwen3:8b' not found` | Run `ollama pull qwen3:8b` (and `nomic-embed-text`). |
| `/ask` returns *"vector store is empty"* | Seed first: `python -m app.scripts.seed_data`. |
| Grades are always `reject` | Lower `RELEVANCE_THRESHOLD`, or check the corpus actually covers the question. |
| Embedding model changed → collection rebuilt | Expected — ChromaDB metadata detects the mismatch and rebuilds automatically. |
| Slow runtime on CPU | Use a smaller grader (`phi4-mini`), raise `RETRIEVAL_K` less aggressively, or reduce `MAX_*` budgets. |
| ChromaDB version conflicts | Keep `chromadb>=0.5` and `rank-bm25>=0.2.2`; avoid mixing pre-built artifacts. |
| Port 8000 already in use | `API_PORT=8001` in `.env` and update `STREAMLIT_API_URL`. |

---

## 🗺 Roadmap

- [x] Dense + BM25 hybrid retrieval with LLM-graded relevance
- [x] Query rewriting with bounded corrective loop (LangGraph)
- [x] Hallucination detection + refinement re-generation loop
- [x] Full-stack API + chat UI with transparency
- [x] Trace persistence + confidence scoring
- [ ] Streaming / SSE answers
- [ ] Citation-only validation node bound to chunk `[n]` references
- [ ] Multi-collection, per-tenant vectors with metadata filters
- [ ] Evaluation harness (RAGAS / LLM-as-judge) over the `/traces` backlog
- [ ] Ingestion watcher (auto-index new files dropped into `data/documents`)

---

## 📄 License

HealRAG is released under the **MIT License**. You are free to use, modify, and
ship it — attribution appreciated. The bundled demo documents are illustrative
health-domain content and are **not medical advice**; HealRAG is a software
reference project, not a clinical device.

---

<p align="center">
  Built with LangGraph · Ollama · ChromaDB · FastAPI · Streamlit
  <br/>
  <sub>Agentic Self-Correction CRAG — retrieval that knows when it is wrong.</sub>
</p>