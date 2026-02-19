# Architecture Overview

## Goal
A production-grade Agentic RAG Chatbot that autonomously selects tools (document search, weather analysis, code execution, memory writing) via a LangGraph StateGraph with native function calling, powered by Llama 3.3 70B via OpenRouter.

---

## High-Level Flow

```
User Query → LangGraph Agent → [Tool Selection via Function Calling] → Grounded Answer + Citations
                  ↑                           ↓
            Chat History          search_documents | get_weather | execute_code | write_memory
                                              ↓
                                        Tool Observation
                                              ↓
                                      LLM Synthesizes Answer
```

### 1) Ingestion (Upload → Parse → Chunk → Embed → Index)

- **Supported inputs**: PDF, TXT, Markdown, HTML
- **Parsing**: `pypdf` for PDFs, built-in HTML stripper, raw text for .txt/.md
- **Chunking**: `RecursiveCharacterTextSplitter` (1000 chars, 200 overlap) with semantic separators (`\n\n`, `\n`, `. `, ` `)
- **Embedding**: `qwen/qwen3-embedding-8b` (4096-dim) via OpenRouter, with `nvidia/llama-3.2-nv-embedqa-1b-v2` (2048-dim) fallback via NVIDIA NIM
  - Documents embedded with `input_type="passage"`
  - Queries embedded with `input_type="query"`
  - Batched (50/batch) with rate throttling for NVIDIA fallback (40 RPM)
- **Metadata per chunk**: filename, page_num, chunk_index, source path, file_type

### 2) Indexing / Storage

- **Vector store**: FAISS `IndexFlatIP` (inner product on L2-normalized vectors = cosine similarity)
  - Exact nearest neighbor — 100% recall, optimal for hackathon-scale datasets (<100K chunks)
  - Dimension validation on load — auto-recreates index on mismatch
- **Metadata sidecar**: SQLite database for chunk text, filename, page numbers
- **Keyword index**: BM25Okapi via `rank-bm25`, pickled to disk
- **Persistence**: FAISS index binary + SQLite DB + BM25 pickle, all in `data/`

### 3) Retrieval + Reranking + Grounded Answering

- **Hybrid retrieval**:
  1. **FAISS semantic search** — top-20 by cosine similarity
  2. **BM25 keyword search** — top-20 by term frequency
  3. **Reciprocal Rank Fusion (RRF)** — merges both ranked lists with `score = Σ 1/(k + rank)`, returns top candidates
- **Neural reranking**: NVIDIA NIM `llama-3.2-nv-rerankqa-1b-v2` reranks the fused candidates → top-5
- **Citation format**: `[Source: filename, Page: X]` — each citation includes source, locator (page/chunk), and snippet
- **Prompt injection defense**: System prompt instructs LLM to treat ALL retrieved content as DATA, never as instructions. Injection attempts in documents are reported as content.
- **Failure behavior**: When retrieval returns empty results, the agent explicitly states "I don't have enough information from the uploaded documents" — never fabricates.

### 4) Agentic Orchestrator (LangGraph StateGraph)

- **Architecture**: LangGraph `StateGraph` with native OpenAI-compatible function calling — no manual JSON parsing
- **Graph structure**: `router` node → tool execution → `synthesize` node → `END`
- **Tool binding**: Tools registered as `langchain_core.tools.BaseTool` subclasses, bound to the LLM via `ChatOpenAI.bind_tools()`
- **Tools available**:
  - `search_documents` — hybrid RAG search with reranking
  - `get_weather` — Open-Meteo with NLP location parsing
  - `execute_code` — safe Python sandbox
  - `write_memory` — persist high-signal facts
- **Max steps**: 4 reasoning steps before forcing a final answer
- **Streaming**: Token-by-token streaming via LangGraph `.stream()` for real-time UX
- **LLM providers**:
  - **Primary**: Llama 3.3 70B via OpenRouter (`ChatOpenAI` with OpenRouter base URL)
  - **Fallback**: Llama 3.3 70B via Groq (`ChatGroq`) — auto-fallback on provider failure

### 5) Memory System (Selective)

- **Decision engine**: LLM-based analysis at low temperature (0.3)
- **High-signal criteria**: User roles, preferences, workflow patterns, organizational structure
- **Explicitly not stored**: Casual chat, one-time questions, PII, secrets, raw transcripts
- **Confidence threshold**: 0.7 minimum — below this, memory is not written
- **Duplicate detection**: Substring matching prevents redundant entries
- **Format**: Markdown files at project root:
  - `USER_MEMORY.md` — personal facts (role, preferences, habits)
  - `COMPANY_MEMORY.md` — organizational knowledge (processes, tools, team structure)

### 6) Safe Tooling (Open-Meteo + Sandbox)

- **Weather**: Open-Meteo free API (no key needed)
  - NLP query parsing: extracts location, metric, time period from natural language
  - Geocoding via geopy/Nominatim
  - Statistical analysis: mean, median, stdev, volatility, anomaly detection (Z-score)
  - Both forecast (hourly) and historical (daily) data supported
- **Sandbox security** (defense in depth):
  1. AST-based static analysis — blocks dangerous constructs before execution
  2. Restricted builtins — `__import__` replaced with whitelisted version, `compile`/`open`/`eval`/`exec` removed
  3. Module whitelist — only math, statistics, numpy, pandas, collections, itertools, etc.
  4. Thread-based timeout — configurable (default 30s)
  5. Output length limits — prevents memory exhaustion

### 7) Frontend (Next.js)

- **Framework**: Next.js 16 with React 19 and Tailwind CSS 4
- **State management**: Zustand store for chat, files, memory, and inspector state
- **Streaming**: Server-Sent Events (SSE) for real-time token streaming from backend
- **UI features**:
  - Orange-themed dark mode with glow effects
  - Collapsible thinking block (shows agent reasoning, collapses on answer)
  - Response time badges on each message
  - PDF viewer with citation highlighting
  - Inspector panel: Trace logs, Source viewer, Memory display, Tools access
  - Toast notifications for file upload lifecycle
- **Backend API**: FastAPI with CORS, file serving, SSE streaming

---

## Tradeoffs & Design Decisions

| Decision | Why | Alternative Considered |
|----------|-----|----------------------|
| **LangGraph StateGraph** | Native function calling, structured graph, streaming support, no manual JSON parsing | Custom ReAct loop (simpler but fragile) |
| **Llama 3.3 70B via OpenRouter** | Strong instruction following, credit-based (no RPM limits), OpenAI-compatible | Kimi K2.5 (thinking mode but slower, NIM rate limits) |
| **Qwen3 Embedding 8B** (4096-dim) | High quality, large context, available on OpenRouter | NVIDIA llama-3.2 (2048-dim, rate limited) |
| **FAISS IndexFlatIP** (exact) | 100% recall, no index build time, fast for <100K vectors | HNSW (faster at scale but approximate) |
| **RRF + NVIDIA Reranker** | RRF is parameter-free fusion; neural reranker improves precision on top-k | Learned fusion (needs training data) |
| **AST + restricted builtins** | Defense in depth — AST catches before exec, builtins catch at runtime | Docker isolation (overkill for hackathon) |
| **Next.js + SSE** | Modern React with streaming UX, static export capability | Streamlit (simpler but limited UX control) |
| **Groq fallback** | 0.4s response time — 22x faster than primary when rate limited | Single provider (fragile) |

## What We Would Improve With More Time

- Knowledge graph augmented retrieval for multi-hop reasoning
- Semantic chunking (section-aware, heading-based splitting)
- Embedding cache to avoid re-embedding duplicate document chunks
- Multi-user support with isolated memory namespaces
- Evaluation harness with test questions and expected citations
- Cross-encoder reranking as second-pass (on top of NVIDIA reranker)
