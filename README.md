[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/P5MLsfQv)

# Agentic RAG Chatbot

A production-grade AI agent that autonomously searches uploaded documents, cites sources, fetches real-time weather data, executes Python safely, and remembers high-signal facts across sessions — all orchestrated by a LangGraph StateGraph with native function calling.

**Built for the Community Dreams Foundation Hackathon.**

---

## Participant Info

| Field | Details |
|-------|---------|
| **Full Name** | Sai Sudheer Naraharisetty |
| **Email** | [sudheernaraharisetti7@gmail.com](mailto:sudheernaraharisetti7@gmail.com) |
| **GitHub** | [SudheerNaraharisetty](https://github.com/SudheerNaraharisetty) |

---

## Video Walkthrough

[Video Demo — Google Drive](https://drive.google.com/drive/folders/1rWtbf4SAqg9VXoWphG8skF1JJC9NEdct?usp=sharing)

---

## Features Implemented

### Feature A — File Upload + RAG with Citations (Core)
- Upload PDF, TXT, Markdown, or HTML files via drag-and-drop or the paperclip button
- Documents are parsed, chunked (1000 chars, 200 overlap), embedded (Qwen3 8B, 4096-dim), and indexed into FAISS + BM25
- **Hybrid retrieval**: FAISS semantic search + BM25 keyword search, fused via Reciprocal Rank Fusion (RRF)
- **Neural reranking**: NVIDIA NIM `llama-3.2-nv-rerankqa-1b-v2` reranks candidates to top-5
- **Citations**: Every answer includes `[Source: filename, Page: X]` with clickable references in the UI
- **Prompt injection defense**: System prompt treats all retrieved content as DATA, never as instructions
- **Graceful failure**: When retrieval returns nothing, the agent says so — never fabricates

### Feature B — Persistent Memory (Core)
- LLM-based selective memory with 0.7 confidence threshold
- Writes to `USER_MEMORY.md` (personal facts: role, preferences) and `COMPANY_MEMORY.md` (org knowledge: processes, tools)
- Duplicate detection prevents redundant entries
- Memory is visible in the Inspector panel's Memory tab in real-time

### Feature C — Safe Python Sandbox + Open-Meteo Weather (Optional)
- **Weather**: Natural language queries → geocoding → Open-Meteo API → statistical analysis (mean, stdev, anomaly detection via Z-score)
- **Sandbox security** (defense in depth):
  1. AST-based static analysis blocks dangerous constructs before execution
  2. Restricted builtins — `__import__` replaced with whitelisted version
  3. Module whitelist — only math, statistics, numpy, pandas, etc.
  4. Thread-based timeout (30s default)
  5. Output length limits prevent memory exhaustion

### Bonus Features
- **Streaming responses**: Token-by-token SSE streaming with real-time thinking indicators
- **Conversation history**: Full chat with response time badges
- **Inspector panel**: Trace logs, Source viewer (PDF with citation highlighting), Memory display, Tools access
- **Dual LLM providers**: OpenRouter (primary) + Groq (fallback, 22x faster for routing)
- **One-command startup**: `node start.js` runs both FastAPI backend and Next.js frontend

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **LLM** | Llama 3.3 70B via OpenRouter (primary) + Groq (fallback) |
| **Embeddings** | Qwen3 Embedding 8B (4096-dim) via OpenRouter |
| **Reranker** | NVIDIA NIM `llama-3.2-nv-rerankqa-1b-v2` |
| **Vector DB** | FAISS IndexFlatIP (exact cosine similarity) |
| **Keyword Search** | BM25Okapi via rank-bm25 |
| **Agent Framework** | LangGraph StateGraph with native function calling |
| **Backend** | FastAPI with SSE streaming |
| **Frontend** | Next.js 16, React 19, Tailwind CSS 4, Zustand |
| **Weather API** | Open-Meteo (free, no key needed) |

---

## Quick Start

```bash
# 1. Clone and enter the repository
git clone <your-repo-url>
cd agentic-rag-chatbot-SudheerNaraharisetty

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows

# 3. Install dependencies
pip install -r requirements.txt
cd frontend && npm install && cd ..

# 4. Configure API keys
cp .env.example .env
# Edit .env and add your API keys:
#   OPENROUTER_API_KEY=sk-or-...       (required - LLM + embeddings)
#   GROQ_API_KEY=gsk-...               (required - fast fallback LLM)
#   NVIDIA_RERANK_API_KEY=nvapi-...    (required - reranking)
#   NVIDIA_EMBEDDING_API_KEY=nvapi-... (optional - fallback embeddings)

# 5. Run the full stack (FastAPI backend + Next.js frontend)
node start.js
# Or run separately:
#   Terminal 1: python -m uvicorn src.api.server:app --reload --port 8000
#   Terminal 2: cd frontend && npm run dev

# 6. Open the app
#   Frontend: http://localhost:3000
#   API Docs: http://localhost:8000/docs

# 7. Run sanity check (for judges)
make sanity
python scripts/verify_output.py artifacts/sanity_output.json
```

**Requirements**: Python 3.10+, Node.js 18+, pip, npm, internet connection for API access.

---

## Project Structure

```
agentic-rag-chatbot-SudheerNaraharisetty/
├── src/
│   ├── api/server.py              # FastAPI backend with SSE streaming
│   ├── core/
│   │   ├── agent.py               # LangGraph StateGraph orchestrator
│   │   ├── config.py              # Settings and environment config
│   │   ├── rag_pipeline.py        # Main pipeline: query → tools → answer
│   │   ├── llm/client.py          # LLM + embedding client (OpenRouter/NVIDIA/Groq)
│   │   ├── retrieval/
│   │   │   ├── hybrid_retriever.py # FAISS + BM25 + RRF fusion
│   │   │   └── reranker.py        # NVIDIA NIM neural reranker
│   │   ├── memory/manager.py      # Selective memory with confidence scoring
│   │   └── document/processor.py  # PDF/TXT/MD/HTML parsing + chunking
│   └── tools/
│       ├── weather.py             # Open-Meteo with NLP location parsing
│       └── sandbox.py             # AST-validated safe Python execution
├── frontend/                      # Next.js 16 + React 19 app
│   ├── app/page.tsx               # Main layout with resizable panels
│   ├── components/
│   │   ├── chat-interface.tsx      # Chat with streaming + thinking indicators
│   │   └── inspector-panel.tsx     # Source viewer, memory, trace, tools
│   └── lib/
│       ├── api.ts                 # SSE streaming client
│       └── store.ts               # Zustand state management
├── start.js                       # Cross-platform startup script
├── Makefile                       # make install / make run / make sanity
├── ARCHITECTURE.md                # Detailed architecture and tradeoffs
├── EVAL_QUESTIONS.md              # Test questions for demo
├── USER_MEMORY.md                 # Written by the agent (user facts)
├── COMPANY_MEMORY.md              # Written by the agent (org knowledge)
└── artifacts/sanity_output.json   # Generated by make sanity
```

---

## Architecture & Tradeoffs

See **[ARCHITECTURE.md](ARCHITECTURE.md)** for the full architecture overview, including:
- Ingestion pipeline (parse → chunk → embed → index)
- Hybrid retrieval with RRF fusion and neural reranking
- LangGraph agent orchestration with native function calling
- Memory decision engine with confidence scoring
- Sandbox security (defense in depth)
- Tradeoff table: why each technology was chosen and what alternatives were considered

---

## Evaluation Prompts

See **[EVAL_QUESTIONS.md](EVAL_QUESTIONS.md)** for test questions covering:
- RAG + Citations (grounded answers with source references)
- Retrieval failure behavior (no hallucinations)
- Memory selectivity (high-signal facts only)
- Prompt injection awareness (treats malicious content as data)

---

## Sanity Check

```bash
# Run the automated sanity check
make sanity

# Verify output format
python scripts/verify_output.py artifacts/sanity_output.json
```

This runs a minimal end-to-end flow (upload → ingest → query → citations → memory write) and produces `artifacts/sanity_output.json` with the required format.
