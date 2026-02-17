"""
RAG Pipeline: Integrates LangChain with our custom components.
"""

from typing import List, Dict, Any, Optional, AsyncGenerator
from pathlib import Path
import numpy as np

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

from src.core.config import get_settings
from src.core.llm.client import LLMClient
from src.core.retrieval.hybrid_retriever import HybridRetriever
from src.core.retrieval.citation_manager import CitationManager
from src.core.document_processor import DocumentProcessor, DocumentChunk


class RAGPipeline:
    """
    End-to-end RAG pipeline integrating all components.
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        retriever: Optional[HybridRetriever] = None,
        citation_manager: Optional[CitationManager] = None,
    ):
        self.settings = get_settings()
        self.llm_client = llm_client or LLMClient()
        self.retriever = retriever or HybridRetriever()
        self.citation_manager = citation_manager or CitationManager()

        # Initialize document processor
        self.doc_processor = DocumentProcessor()

    def ingest_documents(self, file_path: Path) -> Dict[str, Any]:
        """
        Ingest documents into the RAG system.

        Args:
            file_path: Path to document file

        Returns:
            Ingestion stats
        """
        # Process document
        chunks = self.doc_processor.process_file(file_path)

        if not chunks:
            return {"success": False, "error": "No chunks created", "chunks_added": 0}

        # Get embeddings
        texts = [chunk.text for chunk in chunks]
        embeddings = self.llm_client.get_embeddings(texts)

        # Prepare metadata
        metadata_list = [chunk.metadata for chunk in chunks]

        # Add to hybrid retriever
        ids = self.retriever.add_documents(
            embeddings=np.array(embeddings), texts=texts, metadata_list=metadata_list
        )

        return {"success": True, "chunks_added": len(ids), "file": file_path.name}

    def query(self, question: str, k: int = 5) -> Dict[str, Any]:
        """
        Query the RAG system.

        Args:
            question: User question
            k: Number of chunks to retrieve

        Returns:
            Response with answer and citations
        """
        # Get query embedding
        query_embedding = self.llm_client.get_embeddings([question])[0]

        # Hybrid retrieval
        retrieved_docs = self.retriever.search(
            query=question, query_embedding=np.array(query_embedding), k=k
        )

        if not retrieved_docs:
            return {
                "answer": "I don't have enough information to answer this question based on the uploaded documents.",
                "citations": [],
                "sources": [],
            }

        # Create citations
        citations = self.citation_manager.create_citations(retrieved_docs)

        # Format context for LLM
        context = self._format_context(retrieved_docs)
        citation_text = self.citation_manager.format_citations_for_prompt(citations)

        # Generate answer
        prompt = self._create_rag_prompt(question, context, citation_text)

        answer = self.llm_client.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Answer based on the provided context and cite your sources.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=2048,
        )

        # Format response
        return {
            "answer": answer,
            "citations": [c.to_dict() for c in citations],
            "sources": list(set(c.source for c in citations)),
        }

    def _format_context(self, retrieved_docs: List[tuple]) -> str:
        """Format retrieved documents into context string."""
        context_parts = []

        for i, (metadata, score, source_type) in enumerate(retrieved_docs, 1):
            context_parts.append(
                f"[Document {i}] From {metadata.filename} (page {metadata.page_num}):\n"
                f"{metadata.text}\n"
            )

        return "\n".join(context_parts)

    def _create_rag_prompt(self, question: str, context: str, citations: str) -> str:
        """Create RAG prompt."""
        return f"""Answer the following question based on the provided context.

Question: {question}

Context:
{context}

{citations}

Instructions:
1. Answer based ONLY on the provided context
2. Cite your sources using the format [Document X] or [filename, page Y]
3. If the context doesn't contain the answer, say "I don't have enough information to answer this question"
4. Be concise but thorough

Answer:"""

    def stream_query(self, question: str, k: int = 5) -> AsyncGenerator[str, None]:
        """
        Stream query response.

        Args:
            question: User question
            k: Number of chunks to retrieve

        Yields:
            Response chunks
        """
        # Get query embedding
        query_embedding = self.llm_client.get_embeddings([question])[0]

        # Hybrid retrieval
        retrieved_docs = self.retriever.search(
            query=question, query_embedding=np.array(query_embedding), k=k
        )

        if not retrieved_docs:
            yield "I don't have enough information to answer this question based on the uploaded documents."
            return

        # Create context
        context = self._format_context(retrieved_docs)
        citations = self.citation_manager.create_citations(retrieved_docs)
        citation_text = self.citation_manager.format_citations_for_prompt(citations)

        # Generate streaming answer
        prompt = self._create_rag_prompt(question, context, citation_text)

        for chunk in self.llm_client.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Answer based on the provided context and cite your sources.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=2048,
            stream=True,
        ):
            yield chunk


if __name__ == "__main__":
    # Test the pipeline
    pipeline = RAGPipeline()

    print("RAG Pipeline initialized successfully")
    print(f"Current stats: {pipeline.retriever.get_stats()}")
