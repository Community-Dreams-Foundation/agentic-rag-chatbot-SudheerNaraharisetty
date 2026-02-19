from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import shutil
import logging
from pathlib import Path
from sse_starlette.sse import EventSourceResponse, ServerSentEvent
import json
import asyncio

# Import core modules
from src.core.config import get_settings
from src.core.llm.client import LLMClient
from src.core.rag_pipeline import RAGPipeline
from src.core.memory.manager import MemoryManager
from src.tools.weather import get_weather_for_agent, OpenMeteoClient, WeatherAnalyzer, WeatherQueryParser
from src.tools.sandbox import SafeSandbox

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Agentic RAG API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State
class SystemState:
    def __init__(self):
        self.pipeline: Optional[RAGPipeline] = None
        self.memory_manager: Optional[MemoryManager] = None
        self.initialized = False
        self.ingested_docs = []
        self.uploaded_files: Dict[str, bytes] = {}  # filename -> bytes for serving

state = SystemState()

@app.on_event("startup")
async def startup_event():
    logger.info("Initializing Agentic RAG System...")
    try:
        llm_client = LLMClient()
        memory_manager = MemoryManager(llm_client)
        state.pipeline = RAGPipeline(
            llm_client=llm_client,
            memory_manager=memory_manager,
        )
        state.memory_manager = memory_manager
        state.initialized = True
        logger.info("Initialization Complete.")
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")

# --- Schemas ---
class ChatRequest(BaseModel):
    message: str
    model: str = "openrouter" # or "groq"
    history: List[Dict[str, str]] = []

class WeatherRequest(BaseModel):
    query: str

class SandboxRequest(BaseModel):
    code: str

# --- Endpoints ---

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "initialized": state.initialized,
        "docs_count": len(state.ingested_docs)
    }

@app.get("/api/memory")
async def get_memory():
    if not state.initialized:
        return {"user": "", "company": ""}
    return {
        "user": state.memory_manager.read_memory("USER"),
        "company": state.memory_manager.read_memory("COMPANY")
    }

@app.get("/api/files/{filename}")
async def get_file(filename: str):
    """Serve an uploaded file for the PDF viewer."""
    if filename not in state.uploaded_files:
        raise HTTPException(status_code=404, detail="File not found")

    content = state.uploaded_files[filename]
    content_type = "application/pdf" if filename.lower().endswith(".pdf") else "application/octet-stream"
    if filename.lower().endswith(".txt") or filename.lower().endswith(".md"):
        content_type = "text/plain"

    return Response(
        content=content,
        media_type=content_type,
        headers={
            "Content-Disposition": f'inline; filename="{filename}"',
            "Access-Control-Allow-Origin": "*",
        }
    )

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    if not state.initialized:
        raise HTTPException(status_code=503, detail="System not initialized")

    async def event_generator():
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def _enqueue(item):
            """Thread-safe enqueue from the worker thread."""
            loop.call_soon_threadsafe(queue.put_nowait, item)

        def _run_sync_pipeline():
            """Run the blocking sync generator in a thread."""
            try:
                for event_type, data in state.pipeline.query_stream(
                    question=req.message,
                    chat_history=req.history,
                    model=req.model
                ):
                    _enqueue((event_type, data))
            except Exception as e:
                _enqueue(("_error", str(e)))
            finally:
                _enqueue(None)  # sentinel: pipeline done

        # Start the sync pipeline in a thread pool
        loop.run_in_executor(None, _run_sync_pipeline)

        # Consume events from the queue — each await yields to the event loop,
        # allowing uvicorn to flush SSE events to the client in real-time
        token_count = 0
        while True:
            item = await queue.get()
            if item is None:
                break

            event_type, data = item

            if event_type == "_error":
                logger.error(f"Pipeline error: {data}")
                yield ServerSentEvent(data=json.dumps({"type": "error", "content": data}))
                break

            payload = None
            if event_type == "token":
                token_count += 1
                payload = {"type": "token", "content": data}
            elif event_type == "tool":
                logger.info(f"SSE tool: {data.get('tool')}")
                payload = {"type": "tool", "tool": data["tool"], "args": data["args"]}
            elif event_type == "citations":
                logger.info(f"SSE citations: {len(data)} sources")
                payload = {"type": "citations", "citations": data}
            elif event_type == "status":
                payload = {"type": "status", "content": data}

            if payload:
                yield ServerSentEvent(data=json.dumps(payload))
            else:
                logger.warning(f"Unknown event type: {event_type} Data: {str(data)[:50]}")

        logger.info(f"SSE stream complete: {token_count} token events yielded")
        yield ServerSentEvent(data=json.dumps({"type": "done"}))

    return EventSourceResponse(event_generator())

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    if not state.initialized:
        raise HTTPException(status_code=503, detail="System not initialized")

    try:
        if file.filename in state.ingested_docs:
            return {"status": "skipped", "message": "Already ingested"}

        content = await file.read()

        # Store file bytes for serving in PDF viewer
        state.uploaded_files[file.filename] = content

        # Run blocking ingest in thread pool to avoid blocking the event loop
        result = await asyncio.to_thread(state.pipeline.ingest_bytes, content, file.filename)

        if result["success"]:
            state.ingested_docs.append(file.filename)
            return {"status": "success", "filename": file.filename, "chunks": result["chunks_added"]}
        else:
            raise HTTPException(status_code=500, detail=result.get("error"))

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/tools/weather")
async def tool_weather(req: WeatherRequest):
    # Direct tool access for Tools tab
    parser = WeatherQueryParser()
    try:
        parsed = parser.parse_query(req.query)
        if parsed["location"]:
            result = get_weather_for_agent(
                location=parsed["location"],
                metric=parsed["metric"],
                period=parsed["time_period"]
            )
            return result
        return {"error": "Could not parse location"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/tools/sandbox")
async def tool_sandbox(req: SandboxRequest):
    sandbox = SafeSandbox()
    return sandbox.execute(req.code)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
