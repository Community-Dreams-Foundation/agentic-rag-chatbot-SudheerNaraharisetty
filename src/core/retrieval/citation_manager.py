"""
Citation Manager: Handles generation and formatting of citations.
Ensures every claim is backed by source documentation.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import json


@dataclass
class Citation:
    """Represents a single citation."""

    source: str
    locator: str
    snippet: str
    page: int = 1
    chunk_index: int = 0
    relevance_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "locator": self.locator,
            "snippet": self.snippet[:200] + "..."
            if len(self.snippet) > 200
            else self.snippet,
            "page": self.page,
            "chunk_index": self.chunk_index,
            "relevance_score": round(self.relevance_score, 4),
        }

    def format(self) -> str:
        """Format citation for display."""
        return f'[{self.source}, {self.locator}]: "{self.snippet[:150]}..."'


class CitationManager:
    """
    Manages citations for RAG responses.
    Ensures all claims are grounded in retrieved documents.
    """

    def __init__(self, max_citations: int = 5):
        self.max_citations = max_citations

    def create_citations(
        self, retrieved_docs: List[tuple], min_relevance_threshold: float = 0.0
    ) -> List[Citation]:
        """
        Create citations from retrieved documents.

        Args:
            retrieved_docs: List of (metadata, score, source_type) tuples
            min_relevance_threshold: Minimum relevance score to include

        Returns:
            List of Citation objects
        """
        citations = []

        for metadata, score, source_type in retrieved_docs:
            if score < min_relevance_threshold:
                continue

            # Create locator — show page number only (no chunk noise)
            if metadata.page_num > 0:
                locator = f"Page {metadata.page_num}"
            else:
                locator = "Page 1"

            citation = Citation(
                source=metadata.filename,
                locator=locator,
                snippet=metadata.text,
                page=metadata.page_num,
                chunk_index=metadata.chunk_index,
                relevance_score=score,
            )

            citations.append(citation)

        # Sort by relevance and limit
        citations.sort(key=lambda x: x.relevance_score, reverse=True)
        return citations[: self.max_citations]

    def format_citations_for_prompt(self, citations: List[Citation]) -> str:
        """
        Format citations for inclusion in LLM prompt.

        Args:
            citations: List of citations

        Returns:
            Formatted string for prompt
        """
        if not citations:
            return ""

        formatted = "\n\nCitations:\n"
        for i, citation in enumerate(citations, 1):
            formatted += f"[{i}] {citation.source}, {citation.locator}\n"
            formatted += f'    "{citation.snippet[:200]}..."\n\n'

        return formatted

    def format_citations_for_response(
        self, citations: List[Citation]
    ) -> List[Dict[str, Any]]:
        """
        Format citations for JSON response.

        Args:
            citations: List of citations

        Returns:
            List of citation dicts
        """
        return [citation.to_dict() for citation in citations]

    def verify_claim(
        self, claim: str, citations: List[Citation], llm_client=None
    ) -> Dict[str, Any]:
        """
        Verify if a claim is supported by citations.

        Args:
            claim: The claim to verify
            citations: Supporting citations
            llm_client: Optional LLM client for verification

        Returns:
            Verification result dict
        """
        # Simple keyword matching for now
        # Could be enhanced with LLM-based verification

        claim_lower = claim.lower()
        supported = False
        supporting_citations = []

        for citation in citations:
            # Check if claim keywords appear in citation
            citation_text = citation.snippet.lower()

            # Simple overlap check
            claim_words = set(claim_lower.split())
            citation_words = set(citation_text.split())
            overlap = len(claim_words & citation_words)

            if overlap > 0:
                supported = True
                supporting_citations.append(citation.to_dict())

        return {
            "claim": claim,
            "verified": supported,
            "confidence": len(supporting_citations) / max(len(citations), 1),
            "supporting_citations": supporting_citations,
        }

    def extract_claims(self, text: str) -> List[str]:
        """
        Extract factual claims from text.
        Simple implementation - could be enhanced with NLP.

        Args:
            text: Text to analyze

        Returns:
            List of claims
        """
        # Split by sentences
        sentences = text.replace("!", ".").replace("?", ".").split(".")

        # Filter for factual statements (heuristic)
        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  # Minimum length
                # Check for factual indicators
                factual_indicators = [
                    "is",
                    "are",
                    "was",
                    "were",
                    "has",
                    "have",
                    "shows",
                    "demonstrates",
                    "indicates",
                    "reveals",
                    "found",
                    "discovered",
                    "achieved",
                    "reached",
                    "decreased",
                    "increased",
                    "improved",
                    "reduced",
                ]

                if any(
                    indicator in sentence.lower() for indicator in factual_indicators
                ):
                    claims.append(sentence)

        return claims


if __name__ == "__main__":
    # Test citation manager
    manager = CitationManager()

    # Mock retrieved docs
    from src.core.retrieval.vector_engine import DocumentMetadata

    mock_docs = [
        (
            DocumentMetadata(
                id=1,
                filename="test.pdf",
                page_num=5,
                chunk_index=2,
                text="This is a test citation snippet from the document.",
                source="test.pdf",
                file_type="pdf",
            ),
            0.95,
            "semantic",
        ),
        (
            DocumentMetadata(
                id=2,
                filename="test.pdf",
                page_num=3,
                chunk_index=1,
                text="Another relevant piece of information.",
                source="test.pdf",
                file_type="pdf",
            ),
            0.87,
            "keyword",
        ),
    ]

    citations = manager.create_citations(mock_docs)
    print("Citations:")
    for citation in citations:
        print(f"  {citation.format()}")
