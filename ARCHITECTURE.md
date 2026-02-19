# Architecture Overview

## Goal
A production-grade Agentic RAG Chatbot that autonomously selects tools (document search, weather analysis, code execution, memory writing) via a ReAct-style agent loop powered by Kimi K2.5 on NVIDIA NIM.

---

## High-Level Flow

```
User Query → Agent Orchestrator → [Tool Selection Loop] → Grounded Answer + Citations
                  ↑                      ↓
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
- **Embedding**: `nvidia/llama-3.2-nv-embedqa-1b-v2` (2048-dim, QA-optimized, 8K context)
  - Documents embedded with `input_type="passage"`
  - Queries embedded with `input_type="query"`
  - Batched (50/batch) with rate throttling (40 RPM)
- **Metadata per chunk**: filename, page_num, chunk_index, source path, file_type

### 2) Indexing / Storage

- **Vector store**: FAISS `IndexFlatIP` (inner product on L2-normalized vectors = cosine similarity)
  - Exact nearest neighbor — 100% recall, optimal for hackathon-scale datasets (<100K chunks)
  - Dimension validation on load — auto-recreates index on mismatch
- **Metadata sidecar**: SQLite database for chunk text, filename, page numbers
- **Keyword index**: BM25Okapi via `rank-bm25`, pickled to disk
- **Persistence**: FAISS index binary + SQLite DB + BM25 pickle, all in `data/`

### 3) Retrieval + Grounded Answering

- **Hybrid retrieval**:
  1. **FAISS semantic search** — top-20 by cosine similarity
  2. **BM25 keyword search** — top-20 by term frequency
  3. **Reciprocal Rank Fusion (RRF)** — merges both ranked lists with `score = Σ 1/(k + rank)`, returns top-5
- **Citation format**: `[filename, page X]` — each citation includes source, locator (page/chunk), and snippet
- **Prompt injection defense**: System prompt instructs LLM to treat ALL retrieved content as DATA, never as instructions. Injection attempts in documents are reported as content.
- **Failure behavior**: When retrieval returns empty results, the agent explicitly states "I don't have enough information from the uploaded documents" — never fabricates.

### 4) Agentic Orchestrator (ReAct Loop)

- **Architecture**: ReAct-style tool-use loop — the LLM autonomously decides which tool to invoke, interprets results, and synthesizes a grounded answer
- **No keyword routing** — the LLM chooses tools based on its understanding of the query
- **Tools available**:
  - `search_documents` — hybrid RAG search
  - `get_weather` — Open-Meteo with NLP location parsing
  - `execute_code` — safe Python sandbox
  - `write_memory` — persist high-signal facts
- **Max steps**: 4 reasoning steps before forcing a final answer
- **Streaming**: Token-by-token streaming for real-time UX
- **LLM**: Kimi K2.5 (1T param MoE, 32B active) via NVIDIA NIM, with auto-fallback to Groq Llama 3.1 70B on failure

### 5) Memory System (Selective)

- **Decision engine**: LLM-based analysis at low temperature (0.3) with thinking mode OFF
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

---

## Tradeoffs & Design Decisions

| Decision | Why | Alternative Considered |
|----------|-----|----------------------|
| **FAISS IndexFlatIP** (exact) | 100% recall, no index build time, fast for <100K vectors | HNSW (faster at scale but approximate) |
| **RRF over learned fusion** | No training data needed, robust across query types | Cross-encoder reranking (better but slower) |
| **Custom agent over LangGraph** | Zero dependency conflicts, simpler installation, full control | LangGraph (heavier, complex setup) |
| **2048-dim embeddings** | Maximum quality from llama-3.2-nv-embedqa-1b-v2 | 768-dim (faster but lower quality) |
| **AST + restricted builtins** | Defense in depth — AST catches before exec, builtins catch at runtime | Docker isolation (overkill for hackathon) |
| **Kimi K2.5 + Groq fallback** | K2.5 has thinking mode for complex reasoning; Groq provides instant fallback | Single provider (fragile) |
| **Separate embedding API key** | Dedicated quota prevents chat API usage from blocking embeddings | Shared key (rate limit conflicts) |

## What We Would Improve With More Time

- Cross-encoder reranking (e.g., NVIDIA NIM reranker) for the top-k results
- Knowledge graph augmented retrieval for multi-hop reasoning
- Semantic chunking (section-aware, heading-based splitting)
- Embedding cache to avoid re-embedding duplicate document chunks
- Multi-user support with isolated memory namespaces
- Evaluation harness with test questions and expected citations
