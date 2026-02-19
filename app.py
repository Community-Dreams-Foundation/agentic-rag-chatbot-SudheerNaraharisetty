#!/usr/bin/env python3
"""
Agentic RAG Chatbot — Streamlit Application
Features: File Upload + RAG with Citations, Persistent Memory, Safe Sandbox + Weather Tools
Powered by Kimi K2.5 on NVIDIA NIM with Groq auto-fallback.

Author: Sai Sudheer Naraharisetty
Hackathon: Community Dreams Foundation - Agentic RAG Chatbot Challenge
"""

# Fix Streamlit + PyTorch file watcher conflict
# Must be set BEFORE any torch imports
import os

os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

import sys
import json
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from dotenv import load_dotenv
import numpy as np

# Import core modules
from src.core.config import get_settings
from src.core.llm.client import LLMClient
from src.core.rag_pipeline import RAGPipeline
from src.core.memory.manager import MemoryManager
from src.tools.weather import (
    OpenMeteoClient,
    WeatherAnalyzer,
    WeatherQueryParser,
    get_weather_for_agent,
)
from src.tools.sandbox import SafeSandbox

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Page Configuration ──────────────────────────────────────────────

st.set_page_config(
    page_title="Agentic RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session State Initialization ────────────────────────────────────


def init_session():
    """Initialize all session state components."""
    if "initialized" in st.session_state:
        return

    try:
        llm_client = LLMClient()
        memory_manager = MemoryManager(llm_client)
        pipeline = RAGPipeline(
            llm_client=llm_client,
            memory_manager=memory_manager,
        )

        st.session_state.llm_client = llm_client
        st.session_state.pipeline = pipeline
        st.session_state.memory_manager = memory_manager
        st.session_state.weather_client = OpenMeteoClient()
        st.session_state.weather_analyzer = WeatherAnalyzer()
        st.session_state.weather_parser = WeatherQueryParser()
        st.session_state.sandbox = SafeSandbox()
        st.session_state.messages = []
        st.session_state.documents_ingested = []
        st.session_state.initialized = True

    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        st.session_state.initialized = False
        st.session_state.init_error = str(e)


init_session()


# ── Document Upload Handler ─────────────────────────────────────────


def process_uploaded_files(uploaded_files):
    """Process uploaded files and index into RAG system."""
    pipeline = st.session_state.pipeline

    for uploaded_file in uploaded_files:
        if uploaded_file.name in st.session_state.documents_ingested:
            continue

        try:
            result = pipeline.ingest_bytes(
                content=uploaded_file.getvalue(),
                filename=uploaded_file.name,
            )

            if result["success"]:
                st.session_state.documents_ingested.append(uploaded_file.name)
                st.toast(
                    f"Indexed {uploaded_file.name} ({result['chunks_added']} chunks)"
                )
            else:
                st.error(
                    f"Failed to process {uploaded_file.name}: {result.get('error')}"
                )

        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")


# ── Main Application ────────────────────────────────────────────────


def main():
    """Main Streamlit application."""

    # Header
    st.title("🤖 Agentic RAG Chatbot")
    st.caption(
        "Powered by Kimi K2.5 | Hybrid Search (FAISS + BM25) | "
        "Persistent Memory | Safe Sandbox + Weather Tools"
    )

    # Check initialization
    if not st.session_state.get("initialized", False):
        st.error("System failed to initialize. Check your API keys in `.env` file.")
        st.code(
            "# Required in .env:\n"
            "NVIDIA_API_KEY=nvapi-your-key-here\n"
            "NVIDIA_EMBEDDING_API_KEY=nvapi-your-embedding-key-here\n"
            "GROQ_API_KEY=gsk-your-key-here  # optional fallback",
            language="bash",
        )
        if "init_error" in st.session_state:
            with st.expander("Error details"):
                st.code(st.session_state.init_error)
        return

    # ── Sidebar ─────────────────────────────────────────────────

    with st.sidebar:
        st.header("Configuration")

        # LLM Provider Selection
        llm_provider = st.selectbox(
            "LLM Provider",
            ["NVIDIA NIM (Kimi K2.5)", "Groq (Llama 3.1)"],
            index=0,
        )
        model = "nvidia" if "NVIDIA" in llm_provider else "groq"

        st.divider()

        # Feature Toggles
        st.subheader("Features")
        enable_memory = st.checkbox("Memory System", value=True)
        enable_streaming = st.checkbox("Streaming Responses", value=True)

        st.divider()

        # File Upload
        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload PDFs, TXT, or Markdown files",
            type=["pdf", "txt", "md", "html"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            with st.spinner("Processing documents..."):
                process_uploaded_files(uploaded_files)

            if st.session_state.documents_ingested:
                st.success(
                    f"{len(st.session_state.documents_ingested)} document(s) indexed"
                )

        st.divider()

        # System Stats
        st.subheader("System Status")
        pipeline = st.session_state.pipeline
        stats = pipeline.retriever.get_stats()
        st.metric("Indexed Chunks", stats["faiss_documents"])
        st.metric("BM25 Corpus", stats["bm25_documents"])

        # API Key Status
        nvidia_key = os.getenv("NVIDIA_API_KEY", "")
        groq_key = os.getenv("GROQ_API_KEY", "")
        embed_key = os.getenv("NVIDIA_EMBEDDING_API_KEY", "")

        col1, col2 = st.columns(2)
        with col1:
            if nvidia_key and "your-key" not in nvidia_key:
                st.success("NVIDIA NIM", icon="✅")
            else:
                st.error("NVIDIA NIM", icon="❌")
        with col2:
            if groq_key and "your-key" not in groq_key:
                st.success("Groq", icon="✅")
            else:
                st.warning("Groq", icon="⚠️")

        if embed_key and "your-key" not in embed_key:
            st.success("Embedding API", icon="✅")

    # ── Main Content Tabs ───────────────────────────────────────

    tab1, tab2, tab3, tab4 = st.tabs(
        ["💬 Chat", "📚 Documents", "🧠 Memory", "🛠️ Tools"]
    )

    # ────────────────────────────────────────────────────────────
    # TAB 1: Chat Interface (Agent-Driven)
    # ────────────────────────────────────────────────────────────

    with tab1:
        st.header("Chat")
        st.caption(
            "Ask questions about uploaded documents, request weather data, "
            "run calculations, or have a general conversation. "
            "The agent decides which tools to use automatically."
        )

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                # Show citations if available
                if message.get("citations"):
                    with st.expander("📚 Citations", expanded=False):
                        for citation in message["citations"]:
                            st.markdown(
                                f"**{citation.get('source', '?')}, "
                                f"{citation.get('locator', '?')}**"
                            )
                            snippet = citation.get("snippet", "")[:200]
                            st.caption(f"_{snippet}..._")

                # Show tool calls if available
                if message.get("tool_calls"):
                    with st.expander("🔧 Tool Calls", expanded=False):
                        for tc in message["tool_calls"]:
                            st.code(
                                f"{tc['tool']}({json.dumps(tc.get('args', {}), indent=2)})",
                                language="json",
                            )

        # Chat input
        user_input = st.chat_input("Ask about your documents, weather, or anything...")

        if user_input:
            # Add user message to history
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": user_input,
                }
            )

            with st.chat_message("user"):
                st.markdown(user_input)

            # Process with agent
            with st.chat_message("assistant"):
                # Build chat history for agent context
                chat_history = []
                for msg in st.session_state.messages[-6:]:
                    chat_history.append(
                        {
                            "role": msg["role"],
                            "content": msg["content"],
                        }
                    )

                if enable_streaming:
                    _handle_streaming_response(
                        user_input, chat_history, model, enable_memory
                    )
                else:
                    _handle_sync_response(
                        user_input, chat_history, model, enable_memory
                    )

    # ────────────────────────────────────────────────────────────
    # TAB 2: Document Library
    # ────────────────────────────────────────────────────────────

    with tab2:
        st.header("Document Library")

        if st.session_state.documents_ingested:
            for doc_name in st.session_state.documents_ingested:
                with st.expander(f"📄 {doc_name}", expanded=False):
                    st.write("**Status:** Indexed and ready for queries")

            st.divider()
            st.metric(
                "Total Indexed Chunks",
                stats["faiss_documents"],
            )

            # File management
            if st.button("🗑️ Reset All Indexes"):
                pipeline.retriever.faiss_engine.reset()
                st.session_state.documents_ingested = []
                st.success("All indexes cleared.")
                st.rerun()
        else:
            st.info("Upload documents in the sidebar to get started.")

    # ────────────────────────────────────────────────────────────
    # TAB 3: Memory Viewer
    # ────────────────────────────────────────────────────────────

    with tab3:
        st.header("Memory Viewer")
        st.caption(
            "The system automatically identifies and stores high-signal facts "
            "from conversations. Memories personalize future interactions."
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("👤 User Memory")
            user_memories = st.session_state.memory_manager.read_memory("USER")
            if user_memories and len(user_memories.strip()) > 50:
                st.markdown(user_memories)
            else:
                st.info("No user memories yet. Chat with the bot to build memory.")

        with col2:
            st.subheader("🏢 Company Memory")
            company_memories = st.session_state.memory_manager.read_memory("COMPANY")
            if company_memories and len(company_memories.strip()) > 50:
                st.markdown(company_memories)
            else:
                st.info(
                    "No company memories yet. Discuss organizational topics to build memory."
                )

        # Memory management
        st.divider()
        if st.button("🔄 Refresh Memories"):
            st.rerun()

    # ────────────────────────────────────────────────────────────
    # TAB 4: Tools (Weather NLP + Code Sandbox)
    # ────────────────────────────────────────────────────────────

    with tab4:
        st.header("Tools & Sandbox")

        tool_col1, tool_col2 = st.columns(2)

        # ── Weather Tool (NLP Primary) ──
        with tool_col1:
            st.subheader("🌤️ Weather Analysis")
            st.caption("Enter a natural language query or use advanced mode.")

            # NLP Input (Primary)
            weather_query = st.text_input(
                "Weather query",
                placeholder="e.g., What's the temperature in Tokyo last week?",
            )

            if st.button("🔍 Get Weather", key="weather_nlp"):
                if weather_query:
                    with st.spinner("Fetching weather data..."):
                        try:
                            parser = st.session_state.weather_parser
                            parsed = parser.parse_query(weather_query)

                            if parsed["location"]:
                                result = get_weather_for_agent(
                                    location=parsed["location"],
                                    metric=parsed["metric"],
                                    period=parsed["time_period"],
                                )

                                if "error" in result:
                                    st.error(result["error"])
                                else:
                                    st.success(
                                        f"Weather for **{result['location']}** "
                                        f"({result['latitude']}, {result['longitude']})"
                                    )

                                    if "analysis" in result:
                                        st.write("**Statistical Analysis:**")
                                        st.json(result["analysis"])

                                    if "anomalies" in result:
                                        anomaly_data = result["anomalies"]
                                        if anomaly_data.get("anomaly_count", 0) > 0:
                                            st.write("**Anomalies Detected:**")
                                            st.json(anomaly_data)
                                        else:
                                            st.info("No anomalies detected.")

                                    if "date_range" in result:
                                        dr = result["date_range"]
                                        st.caption(
                                            f"Period: {dr['start']} to {dr['end']}"
                                        )
                            else:
                                st.warning(
                                    "Could not identify a location. "
                                    "Try: 'temperature in London last week'"
                                )
                        except Exception as e:
                            st.error(f"Weather error: {e}")
                else:
                    st.warning("Enter a weather query first.")

            # Advanced Mode (Lat/Lon fallback)
            with st.expander("🔧 Advanced: Manual Coordinates"):
                adv_lat = st.number_input("Latitude", value=37.7749, format="%.4f")
                adv_lon = st.number_input("Longitude", value=-122.4194, format="%.4f")
                adv_metric = st.selectbox(
                    "Metric",
                    [
                        "temperature_2m",
                        "relative_humidity_2m",
                        "precipitation",
                        "wind_speed_10m",
                    ],
                )

                if st.button("Fetch by Coordinates", key="weather_adv"):
                    with st.spinner("Fetching..."):
                        try:
                            client = st.session_state.weather_client
                            analyzer = st.session_state.weather_analyzer

                            data = client.get_weather(
                                latitude=adv_lat,
                                longitude=adv_lon,
                                hourly=[adv_metric],
                            )
                            analysis = analyzer.analyze_time_series(
                                data, variable=adv_metric
                            )
                            st.json(analysis)

                        except Exception as e:
                            st.error(f"Error: {e}")

        # ── Code Sandbox ──
        with tool_col2:
            st.subheader("💻 Safe Code Sandbox")
            st.caption(
                "Execute Python safely. Available: math, statistics, numpy, pandas, "
                "collections, itertools, json, re, random, datetime."
            )

            code = st.text_area(
                "Python code",
                value=(
                    "import statistics\n\n"
                    "data = [23.1, 24.5, 22.8, 25.0, 23.7, 26.1]\n"
                    "result = {\n"
                    "    'mean': statistics.mean(data),\n"
                    "    'stdev': statistics.stdev(data),\n"
                    "    'median': statistics.median(data),\n"
                    "}\n"
                    "print(f'Analysis: {result}')"
                ),
                height=180,
            )

            if st.button("▶️ Execute", key="sandbox_exec"):
                with st.spinner("Running in sandbox..."):
                    try:
                        result = st.session_state.sandbox.execute(code)

                        if result["success"]:
                            st.success("Execution successful")
                            if result["output"]:
                                st.code(result["output"], language="text")
                            if result["result"] is not None:
                                st.write(f"**Result:** `{result['result']}`")
                        else:
                            st.error(f"Execution failed: {result['error']}")

                    except Exception as e:
                        st.error(f"Sandbox error: {e}")


# ── Response Handlers ───────────────────────────────────────────────


def _handle_streaming_response(
    user_input: str,
    chat_history: list,
    model: str,
    enable_memory: bool,
):
    """Handle streaming response from the agent."""
    pipeline = st.session_state.pipeline
    all_citations = []
    tool_calls = []
    full_response = ""

    try:
        # Show tool usage and stream final answer
        tool_status = st.empty()
        response_container = st.empty()

        for event_type, data in pipeline.query_stream(
            question=user_input,
            chat_history=chat_history,
            model=model,
        ):
            if event_type == "tool":
                tool_calls.append(data)
                tool_status.info(
                    f"🔧 Using tool: **{data['tool']}** "
                    f"({json.dumps(data.get('args', {}))[:100]}...)"
                )

            elif event_type == "token":
                full_response += data
                response_container.markdown(full_response + "▌")

            elif event_type == "citations":
                all_citations = data

        # Final render
        response_container.markdown(full_response)
        tool_status.empty()

        # Show citations
        if all_citations:
            with st.expander("📚 Citations", expanded=False):
                for citation in all_citations:
                    st.markdown(
                        f"**{citation.get('source', '?')}, "
                        f"{citation.get('locator', '?')}**"
                    )
                    snippet = citation.get("snippet", "")[:200]
                    st.caption(f"_{snippet}..._")

        # Show tool calls
        if tool_calls:
            with st.expander("🔧 Tool Calls", expanded=False):
                for tc in tool_calls:
                    st.code(
                        f"{tc['tool']}({json.dumps(tc.get('args', {}), indent=2)})",
                        language="json",
                    )

        # Save to history
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_response,
                "citations": all_citations,
                "tool_calls": tool_calls,
            }
        )

        # Memory update
        if enable_memory and full_response:
            conversation = f"User: {user_input}\nAssistant: {full_response}"
            st.session_state.memory_manager.update_memory_from_conversation(
                conversation
            )

    except Exception as e:
        st.error(f"Error: {e}")
        logger.error(f"Streaming response failed: {e}", exc_info=True)


def _handle_sync_response(
    user_input: str,
    chat_history: list,
    model: str,
    enable_memory: bool,
):
    """Handle synchronous (non-streaming) response from the agent."""
    pipeline = st.session_state.pipeline

    try:
        with st.spinner("Thinking..."):
            result = pipeline.query(
                question=user_input,
                chat_history=chat_history,
                model=model,
            )

        response = result.get("answer", "I was unable to generate a response.")
        citations = result.get("citations", [])
        tool_calls = result.get("tool_calls", [])

        st.markdown(response)

        if citations:
            with st.expander("📚 Citations", expanded=False):
                for citation in citations:
                    st.markdown(
                        f"**{citation.get('source', '?')}, "
                        f"{citation.get('locator', '?')}**"
                    )
                    snippet = citation.get("snippet", "")[:200]
                    st.caption(f"_{snippet}..._")

        if tool_calls:
            with st.expander("🔧 Tool Calls", expanded=False):
                for tc in tool_calls:
                    st.code(
                        f"{tc['tool']}({json.dumps(tc.get('args', {}), indent=2)})",
                        language="json",
                    )

        # Save to history
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response,
                "citations": citations,
                "tool_calls": tool_calls,
            }
        )

        # Memory update
        if enable_memory and response:
            conversation = f"User: {user_input}\nAssistant: {response}"
            st.session_state.memory_manager.update_memory_from_conversation(
                conversation
            )

    except Exception as e:
        st.error(f"Error: {e}")
        logger.error(f"Sync response failed: {e}", exc_info=True)


# ── Entry Point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
