# Video Walkthrough Script (~7-8 minutes)

> **Style**: Casual, first person, confident. You're selling a product, not presenting homework.
> **Rule**: ZERO code walkthrough. Judges don't care about functions — they care about what it DOES.

---

## INTRO (30 seconds)

"Hey, I'm Sudheer. So I built this Agentic RAG Chatbot for the CDF Hackathon, and I went all in — all three features: file-grounded Q&A with citations, persistent memory, AND the safe sandbox with weather analysis. Not a partial implementation, the full thing.

Quick context on the stack — I'm running Llama 3.3 70B through OpenRouter, Qwen3 embeddings at 4096 dimensions, NVIDIA NIM for neural reranking, FAISS plus BM25 for hybrid retrieval, LangGraph for the agent orchestration, and the frontend is Next.js 16 with React 19. No Ollama, no local models — this is production-grade cloud infrastructure."

---

## DEMO: STARTUP (30 seconds)

*Show terminal, run `node start.js`*

"One command — `node start.js` — spins up both the FastAPI backend and the Next.js frontend. Backend loads the FAISS index, BM25 index, connects to all three AI providers. Frontend comes up on localhost:3000.

And if judges want to verify the submission, it's just `make sanity` — produces the sanity output JSON, pass it through the verifier, done."

---

## FEATURE A: RAG + CITATIONS (2.5 minutes)

*Show the app in browser*

"Alright, let's upload a document. I'm going with this arXiv paper — just click the paperclip, drop the PDF."

*Upload a PDF, wait for toast notification*

"See that? Parsed, chunked, embedded, indexed — you get a toast telling you exactly how many chunks were created. Now let's actually use it."

*Type: "Summarize the main contribution in 3 bullets"*

"Watch the thinking indicator — you can see the agent's reasoning in real-time. It decided to call `search_documents`, you can see the tool badge pop up. And here comes the answer, streaming token by token — not waiting for the whole thing to generate, it's Server-Sent Events streaming directly from the backend."

*Point at citations*

"Citations. Every claim is backed by a source reference — filename, page number. Click on one and it opens in the Source tab on the right with the actual document. This isn't cosmetic — the retrieval pipeline is doing hybrid search: FAISS for semantic similarity, BM25 for exact keyword matching, then Reciprocal Rank Fusion merges both result sets, and NVIDIA's neural reranker picks the top 5 most relevant passages. That's a four-stage retrieval pipeline."

*Type: "What is the CEO's phone number?"*

"Now here's the important one — retrieval failure. I'm asking something that's obviously not in the paper. Watch — it doesn't hallucinate, doesn't make up a phone number. It explicitly says 'I couldn't find this in the uploaded documents.' That's the grounding rule — if the tools don't return it, the agent doesn't invent it.

And honestly, this is where a lot of competitors fall apart. If you're running Ollama with some 7B model locally, you get hallucinations the moment retrieval returns nothing. These smaller models don't have the instruction-following discipline to say 'I don't know.' Llama 3.3 70B does."

---

## FEATURE B: MEMORY (1.5 minutes)

*Type: "I'm a Project Finance Analyst and I prefer weekly summaries on Mondays"*

"Alright, I just told it some personal facts. Watch the Memory tab on the right..."

*Click Memory tab in inspector*

"Boom — USER_MEMORY.md just updated. It wrote 'User is a Project Finance Analyst' and 'Prefers weekly summaries on Mondays' with timestamps and confidence scores.

Here's what's important — this is SELECTIVE memory. The agent runs every message through an LLM decision at low temperature — 0.3 — and only writes if confidence is above 0.7. It asks itself: 'Is this a high-signal fact worth remembering? Or just casual chat?' If I said 'hey what's up,' it wouldn't write anything. No transcript dumping.

And there's duplicate detection too — if I say 'I'm a Project Finance Analyst' again, it won't create a second entry. It checks for substring matches before writing.

COMPANY_MEMORY.md captures organizational knowledge — stuff that would be useful to other colleagues, like 'Asset Management interfaces with Project Finance' or 'Recurring workflow bottleneck is X.' Two separate markdown files, both visible in the Memory tab."

---

## FEATURE C: WEATHER + SANDBOX (1.5 minutes)

*Type: "What's the weather in New York for this week?"*

"Feature C — the optional one that most people skipped. I didn't. Watch — the agent calls `get_weather`, it does NLP parsing on the query to extract the location and time period, geocodes 'New York' to coordinates, hits the Open-Meteo API, and runs statistical analysis: mean, median, standard deviation, anomaly detection using Z-scores. All from a natural language question."

*Type: "Calculate the standard deviation of [23, 45, 12, 67, 34, 89, 56]"*

"Now the sandbox. I'm asking it to run actual Python code. The agent generates the code, runs it in a sandboxed environment. And this sandbox has five layers of security:

One — AST analysis checks the code BEFORE execution, blocking things like os.system or subprocess.
Two — restricted builtins, so __import__ is replaced with a whitelisted version.
Three — only approved modules: math, statistics, numpy, pandas.
Four — thread-based timeout, 30 seconds max.
Five — output length limits so you can't memory-bomb the server.

This is defense in depth. Not just 'we blocked eval()' — it's a real security model. And I'd love to see someone running code execution through Ollama's local setup try to match this."

---

## THE COMPETITIVE EDGE (1 minute)

"Let me be real about why this submission is different.

Most competitors are running Ollama with like a 7B or 13B model locally. That works for your first three questions. But the moment you need complex multi-step reasoning — search documents, analyze results, decide whether to write memory, generate an answer with citations — those small models lose coherence. They hallucinate citations, they dump entire transcripts into memory, and they definitely can't handle function calling properly.

I'm using Llama 3.3 70B — a 70 billion parameter model with proper instruction following and native function calling through the OpenAI API format. No regex parsing tool calls out of raw text. LangGraph handles the orchestration with a proper state graph — it's not a hacky while-loop.

And the dual-provider setup means if OpenRouter is rate-limited, Groq picks up automatically — 0.4 second response time on Groq vs 9 seconds on some providers. That's 22x faster failover.

The embedding model is Qwen3 at 4096 dimensions — most people are using whatever default 384 or 768 dimension model ships with their setup. Higher dimensions means better semantic separation means better retrieval. Combined with the neural reranker, the retrieval quality here is genuinely good."

---

## TRADEOFFS & WHAT I'D IMPROVE (1 minute)

"Tradeoffs — I chose FAISS with exact search over approximate nearest neighbors. At hackathon scale with less than 100K vectors, exact search gives 100% recall with negligible speed difference. If this needed to scale to millions of documents, I'd switch to HNSW.

I chose cloud APIs over local models. Yes, it needs internet. Yes, it costs money per API call. But the quality difference between a 70B model with proper function calling and a quantized 7B running on a laptop is night and day. For a production product, you want the best model you can get.

I chose RRF fusion over learned fusion. RRF is parameter-free — no training data needed, works out of the box. Learned fusion would be better at scale but needs labeled relevance judgments I don't have.

What I'd improve with more time:
- Knowledge graph augmented retrieval for multi-hop reasoning
- Semantic chunking that respects section boundaries and headings
- Embedding cache to avoid re-embedding duplicate chunks
- Multi-user support with isolated memory namespaces
- A proper evaluation harness with ground truth answers and citation verification"

---

## CLOSING (15 seconds)

"That's it — all three features, production-grade infrastructure, real streaming UX, and a security model that actually holds up. Thanks for watching."

---

## RECORDING TIPS

- **Screen resolution**: 1920x1080, zoom browser to 110% so text is readable
- **Have a PDF ready**: Use an arXiv paper from sample_docs/ or download one
- **Clear the chat first**: Fresh start looks cleaner
- **Show the Inspector panel**: Toggle between Source, Memory, Trace tabs during demo
- **Don't rush**: Let streaming complete before talking about the result
- **Kill and restart clean**: `node start.js` from a fresh terminal, no leftover processes
- **Clear USER_MEMORY.md and COMPANY_MEMORY.md** before recording so memory writes are visible in real-time
