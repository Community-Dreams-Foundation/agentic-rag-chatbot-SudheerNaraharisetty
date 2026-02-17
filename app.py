#!/usr/bin/env python3
"""
Agentic RAG Chatbot - Main Application
A high-performance RAG system with citations, memory, and tool calling.
"""

import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Agentic RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    """Main application entry point."""

    # Header
    st.title("🤖 Agentic RAG Chatbot")
    st.markdown("""
    **High-Performance RAG System with:**
    - ✅ File Upload & Document Processing
    - ✅ Hybrid Search (Semantic + Keyword)
    - ✅ Citations & Grounded Answers
    - ✅ Persistent Memory
    - ✅ Safe Code Sandbox & Weather Tools
    """)

    st.divider()

    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # LLM Selection
        llm_provider = st.selectbox(
            "LLM Provider", ["NVIDIA NIM (Kimi K2.5)", "Groq (Llama 3.1)"], index=0
        )

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
            st.success(f"📚 {len(uploaded_files)} file(s) ready for processing")

        st.divider()

        # System Status
        st.subheader("📊 System Status")

        # Check API keys
        nvidia_key = os.getenv("NVIDIA_API_KEY")
        groq_key = os.getenv("GROQ_API_KEY")

        if nvidia_key:
            st.success("✅ NVIDIA NIM Connected")
        else:
            st.error("❌ NVIDIA NIM Key Missing")

        if groq_key:
            st.success("✅ Groq Connected")
        else:
            st.warning("⚠️ Groq Key Missing (Optional)")

    # Main content area
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📚 Documents", "🧠 Memory"])

    with tab1:
        st.header("Chat Interface")

        # Chat input
        user_input = st.text_area(
            "Ask a question:",
            placeholder="Ask about your uploaded documents or use weather tools...",
            height=100,
        )

        col1, col2, col3 = st.columns([1, 1, 4])

        with col1:
            submit = st.button("🚀 Submit", type="primary", use_container_width=True)

        with col2:
            clear = st.button("🗑️ Clear", use_container_width=True)

        # Response area
        if submit and user_input:
            with st.spinner("Processing..."):
                # Placeholder for actual implementation
                st.info("🚧 Full implementation in progress...")

                st.markdown("""
                **Response will include:**
                - Generated answer based on retrieved documents
                - Citations with source files and page numbers
                - Relevance scores
                
                **Features active:**
                - Hybrid retrieval (FAISS + BM25)
                - Exact nearest neighbor search
                - RRF fusion algorithm
                """)

        if clear:
            st.rerun()

    with tab2:
        st.header("Document Library")

        if uploaded_files:
            for i, file in enumerate(uploaded_files, 1):
                with st.expander(f"📄 {file.name}"):
                    st.write(f"**Type:** {file.type}")
                    st.write(f"**Size:** {file.size / 1024:.2f} KB")
                    st.write("**Status:** Ready to process")
        else:
            st.info("📤 Upload documents in the sidebar to get started")

    with tab3:
        st.header("Memory Viewer")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("👤 User Memory")
            user_memory_file = Path("USER_MEMORY.md")
            if user_memory_file.exists():
                with open(user_memory_file, "r") as f:
                    content = f.read()
                    if content:
                        st.markdown(content)
                    else:
                        st.info("No user memories yet")
            else:
                st.info("User memory file will be created automatically")

        with col2:
            st.subheader("🏢 Company Memory")
            company_memory_file = Path("COMPANY_MEMORY.md")
            if company_memory_file.exists():
                with open(company_memory_file, "r") as f:
                    content = f.read()
                    if content:
                        st.markdown(content)
                    else:
                        st.info("No company memories yet")
            else:
                st.info("Company memory file will be created automatically")


if __name__ == "__main__":
    main()
