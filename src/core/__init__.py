# src/core/__init__.py
"""Core modules for the Agentic RAG Chatbot."""

from src.core.config import get_settings, Settings
from src.core.llm.client import LLMClient
from src.core.retrieval.vector_engine import FaissEngine, DocumentMetadata
from src.core.retrieval.hybrid_retriever import HybridRetriever
from src.core.retrieval.citation_manager import CitationManager, Citation
from src.core.memory.manager import MemoryManager, MemoryEntry

__all__ = [
    "get_settings",
    "Settings",
    "LLMClient",
    "FaissEngine",
    "DocumentMetadata",
    "HybridRetriever",
    "CitationManager",
    "Citation",
    "MemoryManager",
    "MemoryEntry",
]
