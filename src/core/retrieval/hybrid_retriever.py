"""
Hybrid Retriever: Combines FAISS (semantic) + BM25 (keyword) for maximum accuracy.
Uses Reciprocal Rank Fusion (RRF) to merge results.
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from rank_bm25 import BM25Okapi

from src.core.retrieval.vector_engine import FaissEngine, DocumentMetadata
from src.core.config import get_settings


class HybridRetriever:
    """
    Hybrid search combining semantic (FAISS) and keyword (BM25) retrieval.
    Uses RRF (Reciprocal Rank Fusion) for result merging.
    """

    def __init__(
        self,
        faiss_engine: Optional[FaissEngine] = None,
        bm25_index_path: Optional[Path] = None,
    ):
        self.settings = get_settings()
        self.faiss_engine = faiss_engine or FaissEngine()
        self.bm25_index_path = bm25_index_path or self.settings.bm25_index_path

        # BM25 components
        self.bm25: Optional[BM25Okapi] = None
        self.corpus: List[str] = []
        self.corpus_metadata: List[Dict] = []

        # Load existing BM25 index if available
        self._load_bm25_index()

    def _load_bm25_index(self):
        """Load BM25 index from disk if it exists."""
        if self.bm25_index_path.exists():
            print(f"Loading existing BM25 index from {self.bm25_index_path}")
            with open(self.bm25_index_path, "rb") as f:
                data = pickle.load(f)
                self.bm25 = data["bm25"]
                self.corpus = data["corpus"]
                self.corpus_metadata = data["metadata"]

    def _save_bm25_index(self):
        """Save BM25 index to disk."""
        self.bm25_index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.bm25_index_path, "wb") as f:
            pickle.dump(
                {
                    "bm25": self.bm25,
                    "corpus": self.corpus,
                    "metadata": self.corpus_metadata,
                },
                f,
            )

    def add_documents(
        self, embeddings: np.ndarray, texts: List[str], metadata_list: List[Dict]
    ) -> List[int]:
        """
        Add documents to both FAISS and BM25 indexes.

        Args:
            embeddings: Document embeddings for FAISS
            texts: Raw texts for BM25
            metadata_list: Metadata for each document

        Returns:
            List of assigned IDs from FAISS
        """
        if not (len(embeddings) == len(texts) == len(metadata_list)):
            raise ValueError("All inputs must have the same length")

        # Add to FAISS
        faiss_ids = self.faiss_engine.add_documents(embeddings, metadata_list)

        # Add to BM25
        self.corpus.extend(texts)
        self.corpus_metadata.extend(metadata_list)

        # Tokenize corpus for BM25
        tokenized_corpus = [doc.lower().split() for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # Save BM25 index
        self._save_bm25_index()

        return faiss_ids

    def _reciprocal_rank_fusion(
        self,
        semantic_results: List[Tuple[int, float]],
        keyword_results: List[Tuple[int, float]],
        k: int = 60,
    ) -> List[Tuple[int, float]]:
        """
        Merge results using Reciprocal Rank Fusion.

        RRF formula: score = sum(1 / (k + rank)) for each list

        Args:
            semantic_results: List of (id, score) from semantic search
            keyword_results: List of (id, score) from keyword search
            k: RRF constant (typically 60)

        Returns:
            List of (id, rrf_score) sorted by score descending
        """
        scores: Dict[int, float] = {}

        # Add semantic results
        for rank, (doc_id, _) in enumerate(semantic_results):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)

        # Add keyword results
        for rank, (doc_id, _) in enumerate(keyword_results):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)

        # Sort by score descending
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results

    def search(
        self,
        query: str,
        query_embedding: np.ndarray,
        k: int = 5,
        semantic_k: int = 20,
        keyword_k: int = 20,
    ) -> List[Tuple[DocumentMetadata, float, str]]:
        """
        Hybrid search using both semantic and keyword retrieval.

        Args:
            query: Raw query text for BM25
            query_embedding: Query embedding for FAISS
            k: Number of final results
            semantic_k: Number of candidates from semantic search
            keyword_k: Number of candidates from keyword search

        Returns:
            List of (metadata, rrf_score, source_type) tuples
        """
        # Semantic search
        semantic_results_raw = self.faiss_engine.search(query_embedding, k=semantic_k)
        semantic_results = [
            (metadata.id, distance) for metadata, distance in semantic_results_raw
        ]

        # Keyword search
        keyword_results: List[Tuple[int, float]] = []
        if self.bm25 and query.strip():
            tokenized_query = query.lower().split()
            bm25_scores = self.bm25.get_scores(tokenized_query)

            # Get top-k indices
            top_indices = np.argsort(bm25_scores)[::-1][:keyword_k]
            keyword_results = [
                (self.corpus_metadata[idx].get("id", idx), bm25_scores[idx])
                for idx in top_indices
                if bm25_scores[idx] > 0
            ]

        # Merge with RRF
        fused_results = self._reciprocal_rank_fusion(semantic_results, keyword_results)[
            :k
        ]

        # Retrieve full metadata for fused results
        conn = self.faiss_engine._init_sqlite.__wrapped__
        # Actually, let me fix this - I need to retrieve from SQLite

        results = []
        for doc_id, rrf_score in fused_results:
            # Determine source type
            source_type = "hybrid"
            if doc_id in [r[0] for r in semantic_results] and doc_id in [
                r[0] for r in keyword_results
            ]:
                source_type = "hybrid"
            elif doc_id in [r[0] for r in semantic_results]:
                source_type = "semantic"
            else:
                source_type = "keyword"

            # Get metadata from FAISS engine
            # We need to query SQLite by ID
            import sqlite3

            conn = sqlite3.connect(self.faiss_engine.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, filename, page_num, chunk_index, text, source, file_type FROM chunks WHERE id=?",
                (doc_id,),
            )
            row = cursor.fetchone()
            conn.close()

            if row:
                metadata = DocumentMetadata(
                    id=row[0],
                    filename=row[1],
                    page_num=row[2],
                    chunk_index=row[3],
                    text=row[4],
                    source=row[5],
                    file_type=row[6],
                )
                results.append((metadata, rrf_score, source_type))

        return results

    def get_stats(self) -> Dict:
        """Get statistics about the indexes."""
        return {
            "faiss_documents": self.faiss_engine.get_document_count(),
            "bm25_documents": len(self.corpus),
            "faiss_index_path": str(self.faiss_engine.index_path),
            "bm25_index_path": str(self.bm25_index_path),
        }
