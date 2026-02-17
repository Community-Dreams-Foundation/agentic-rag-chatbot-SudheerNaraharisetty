# src/core/retrieval/__init__.py
"""Retrieval modules."""

from src.core.retrieval.vector_engine import FaissEngine, DocumentMetadata
from src.core.retrieval.hybrid_retriever import HybridRetriever
from src.core.retrieval.citation_manager import CitationManager, Citation

__all__ = [
    "FaissEngine",
    "DocumentMetadata",
    "HybridRetriever",
    "CitationManager",
    "Citation",
]
