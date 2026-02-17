#!/usr/bin/env python3
"""
Complete Agentic RAG Chatbot Application
Integrates all components: RAG, Memory, Tools
"""

import os
import sys
import json
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from dotenv import load_dotenv
import numpy as np

# Import our modules
from src.core.config import get_settings
from src.core.llm.client import LLMClient
from src.core.rag_pipeline import RAGPipeline
from src.core.memory.manager import MemoryManager
from src.tools.weather import OpenMeteoClient, WeatherAnalyzer
from src.tools.sandbox import SafeSandbox

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Agentic RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "rag_pipeline" not in st.session_state:
    try:
        st.session_state.rag_pipeline = RAGPipeline()
        st.session_state.memory_manager = MemoryManager()
        st.session_state.weather_client = OpenMeteoClient()
        st.session_state.weather_analyzer = WeatherAnalyzer()
        st.session_state.sandbox = SafeSandbox()
        st.session_state.messages = []
        st.session_state.documents_ingested = []
        st.session_state.initialized = True
    except Exception as e:
        st.error(f"❌ Error initializing system: {e}")
        st.session_state.initialized = False


def process_uploaded_files(uploaded_files):
    """Process uploaded files and add to RAG system."""
    pipeline = st.session_state.rag_pipeline

    for uploaded_file in uploaded_files:
        if uploaded_file.name in st.session_state.documents_ingested:
            continue

        try:
            # Process file
            chunks = pipeline.doc_processor.process_bytes(
                content=uploaded_file.getvalue(), filename=uploaded_file.name
            )

            if chunks:
                # Get embeddings
                texts = [chunk.text for chunk in chunks]
                embeddings = pipeline.llm_client.get_embeddings(texts)

                # Add to retriever
                metadata_list = [chunk.metadata for chunk in chunks]
                pipeline.retriever.add_documents(
                    embeddings=np.array(embeddings),
                    texts=texts,
                    metadata_list=metadata_list,
                )

                st.session_state.documents_ingested.append(uploaded_file.name)

        except Exception as e:
            st.error(f"❌ Error processing {uploaded_file.name}: {e}")


def main():
    """Main application."""

    # Header
    st.title("🤖 Agentic RAG Chatbot")
    st.markdown("""
    **High-Performance RAG System with:**
    - ✅ File Upload & Document Processing  
    - ✅ Hybrid Search (Semantic + Keyword)
    - ✅ Citations & Grounded Answers
    - ✅ Persistent Memory
    - ✅ Weather Tools & Code Sandbox
    """)

    st.divider()

    # Check initialization
    if not st.session_state.get("initialized", False):
        st.error("⚠️ System failed to initialize. Check your API keys in .env file.")
        st.code("""
# Make sure you have:
NVIDIA_API_KEY=nvapi-your-key-here
GROQ_API_KEY=gsk-your-key-here
        """)
        return

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # LLM Selection
        llm_provider = st.selectbox(
            "LLM Provider", ["NVIDIA NIM (Kimi K2.5)", "Groq (Llama 3.1)"], index=0
        )
        model = "nvidia" if "NVIDIA" in llm_provider else "groq"

        st.divider()

        # Features
        st.subheader("🔧 Features")
        enable_hybrid = st.checkbox("Hybrid Search", value=True)
        enable_citations = st.checkbox("Citations", value=True)
        enable_memory = st.checkbox("Memory System", value=True)
        enable_sandbox = st.checkbox("Code Sandbox", value=True)

        st.divider()

        # File Upload
        st.subheader("📄 Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload PDFs, TXT, or Markdown files",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            with st.spinner("Processing documents..."):
                process_uploaded_files(uploaded_files)

            if st.session_state.documents_ingested:
                st.success(
                    f"✅ {len(st.session_state.documents_ingested)} document(s) processed"
                )

        st.divider()

        # System Stats
        st.subheader("📊 System Status")
        pipeline = st.session_state.rag_pipeline
        stats = pipeline.retriever.get_stats()
        st.write(f"📚 Documents: {stats['faiss_documents']}")

        # Check API keys
        nvidia_key = os.getenv("NVIDIA_API_KEY")
        groq_key = os.getenv("GROQ_API_KEY")

        if nvidia_key and nvidia_key != "nvapi-your-key-here":
            st.success("✅ NVIDIA NIM")
        else:
            st.error("❌ NVIDIA NIM Key")

        if groq_key and groq_key != "gsk-your-key-here":
            st.success("✅ Groq")
        else:
            st.warning("⚠️ Groq Key (Optional)")

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["💬 Chat", "📚 Documents", "🧠 Memory", "🛠️ Tools"]
    )

    with tab1:
        st.header("Chat Interface")

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if message.get("citations"):
                    with st.expander("📚 Citations"):
                        for citation in message["citations"]:
                            st.write(f"**{citation['source']}, {citation['locator']}**")
                            st.write(f"_{citation['snippet'][:200]}..._")

        # Chat input
        user_input = st.chat_input(
            "Ask a question about your documents or use tools..."
        )

        if user_input:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_input})

            with st.chat_message("user"):
                st.write(user_input)

            # Process query
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Check if it's a tool query
                        if any(
                            keyword in user_input.lower()
                            for keyword in ["weather", "temperature", "forecast"]
                        ):
                            # Handle weather query
                            st.write(
                                "🌤️ I'll help you with weather information. Please use the Tools tab for detailed weather analysis."
                            )
                            response = "Use the Tools tab for weather queries with location coordinates."
                            citations = []

                        elif any(
                            keyword in user_input.lower()
                            for keyword in ["code", "analyze", "calculate", "python"]
                        ):
                            st.write(
                                "💻 For code execution, please use the Tools tab with the Safe Sandbox."
                            )
                            response = "Use the Tools tab for code sandbox execution."
                            citations = []

                        else:
                            # RAG query
                            pipeline = st.session_state.rag_pipeline
                            result = pipeline.query(user_input, k=5)

                            response = result["answer"]
                            citations = result.get("citations", [])

                            # Update memory if enabled
                            if enable_memory:
                                conversation = (
                                    f"User: {user_input}\nAssistant: {response}"
                                )
                                st.session_state.memory_manager.update_memory_from_conversation(
                                    conversation
                                )

                        st.write(response)

                        if citations:
                            with st.expander("📚 Citations"):
                                for citation in citations:
                                    st.write(
                                        f"**{citation['source']}, {citation['locator']}**"
                                    )
                                    st.write(f"_{citation['snippet'][:200]}..._")

                        # Add to chat history
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": response,
                                "citations": citations,
                            }
                        )

                    except Exception as e:
                        st.error(f"❌ Error: {e}")

    with tab2:
        st.header("Document Library")

        if st.session_state.documents_ingested:
            for doc in st.session_state.documents_ingested:
                with st.expander(f"📄 {doc}"):
                    st.write("**Status:** Indexed and ready for queries")
                    st.write(
                        f"**Total documents in system:** {stats['faiss_documents']}"
                    )
        else:
            st.info("📤 Upload documents in the sidebar to get started")

    with tab3:
        st.header("Memory Viewer")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("👤 User Memory")
            user_memories = st.session_state.memory_manager.read_memory("USER")
            if user_memories:
                st.markdown(user_memories)
            else:
                st.info(
                    "No user memories yet. Memories will be created during conversation."
                )

        with col2:
            st.subheader("🏢 Company Memory")
            company_memories = st.session_state.memory_manager.read_memory("COMPANY")
            if company_memories:
                st.markdown(company_memories)
            else:
                st.info(
                    "No company memories yet. Memories will be created during conversation."
                )

    with tab4:
        st.header("Tools & Sandbox")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🌤️ Weather Analysis")

            lat = st.number_input("Latitude", value=37.7749, format="%.4f")
            lon = st.number_input("Longitude", value=-122.4194, format="%.4f")

            if st.button("Get Weather Analysis"):
                with st.spinner("Fetching weather data..."):
                    try:
                        # Get weather data
                        weather_data = st.session_state.weather_client.get_weather(
                            latitude=lat,
                            longitude=lon,
                            hourly=["temperature_2m", "relative_humidity_2m"],
                        )

                        # Analyze
                        analysis = (
                            st.session_state.weather_analyzer.analyze_time_series(
                                weather_data, variable="temperature_2m"
                            )
                        )

                        st.write("**Temperature Analysis:**")
                        st.json(analysis)

                    except Exception as e:
                        st.error(f"Error: {e}")

        with col2:
            st.subheader("💻 Safe Code Sandbox")

            code = st.text_area(
                "Enter Python code:",
                value="# Example: data analysis\nresult = sum([1, 2, 3, 4, 5])\nprint(f'Sum: {result}')",
                height=150,
            )

            if st.button("Execute Code"):
                with st.spinner("Executing in sandbox..."):
                    try:
                        result = st.session_state.sandbox.execute(code)

                        if result["success"]:
                            st.success("✅ Execution successful")
                            if result["output"]:
                                st.code(result["output"], language="python")
                            if result["result"] is not None:
                                st.write(f"**Result:** {result['result']}")
                        else:
                            st.error(f"❌ Execution failed: {result['error']}")

                    except Exception as e:
                        st.error(f"Sandbox error: {e}")


if __name__ == "__main__":
    main()
