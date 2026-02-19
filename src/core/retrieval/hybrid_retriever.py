"""
Hybrid Retriever: FAISS (semantic) + BM25 (keyword) + NVIDIA cross-encoder reranking.

Four-stage retrieval pipeline:
  1. Parallel FAISS (semantic) + BM25 (keyword) candidate generation
  2. Reciprocal Rank Fusion to merge and score candidates
  3. NVIDIA cross-encoder reranking (llama-3.2-nv-rerankqa-1b-v2)
  4. Return top-k with full metadata

The reranker is designed to work with llama-3.2-nv-embedqa-1b-v2 embeddings,
providing 73.64% Recall@5 on standard benchmarks (NQ + HotpotQA + FiQA + TechQA).
"""

import pickle
import sqlite3
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from src.core.retrieval.vector_engine import FaissEngine, DocumentMetadata
from src.core.retrieval.reranker import get_reranker, NVIDIAReranker
from src.core.config import get_settings

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Four-stage retrieval pipeline:
      1. Parallel FAISS (semantic) + BM25 (keyword) candidate generation
      2. Reciprocal Rank Fusion to merge and score candidates
      3. NVIDIA cross-encoder reranking for precision
      4. Return top-k with full metadata and source tracking

    Benchmarks with reranking enabled:
      - NQ + HotpotQA + FiQA + TechQA: 73.64% Recall@5
      - Multilingual (MIRACL): 65.80% Recall@5
      - Cross-lingual (MLQA): 86.83% Recall@5
    """

    def __init__(
        self,
        faiss_engine: Optional[FaissEngine] = None,
        bm25_index_path: Optional[Path] = None,
        reranker: Optional[NVIDIAReranker] = None,
    ):
        self.settings = get_settings()
        self.faiss_engine = faiss_engine or FaissEngine()
        self.bm25_index_path = bm25_index_path or self.settings.bm25_index_path

        # BM25 components
        self.bm25: Optional[BM25Okapi] = None
        self.corpus: List[str] = []
        self.corpus_metadata: List[Dict] = []

        # Reranker (lazy-loaded)
        self._reranker = reranker

        self._load_bm25_index()

    @property
    def reranker(self) -> Optional[NVIDIAReranker]:
        """Lazy-load reranker only when needed."""
        if self._reranker is None and self.settings.enable_reranking:
            try:
                self._reranker = get_reranker()
                logger.info("NVIDIA reranker initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize reranker: {e}")
        return self._reranker

    def _load_bm25_index(self):
        """Load BM25 index from disk if it exists."""
        if self.bm25_index_path.exists():
            try:
                with open(self.bm25_index_path, "rb") as f:
                    data = pickle.load(f)
                    self.bm25 = data["bm25"]
                    self.corpus = data["corpus"]
                    self.corpus_metadata = data["metadata"]
                logger.info(f"Loaded persistent index from disk with {len(self.corpus)} documents")
            except Exception as e:
                logger.warning(f"Failed to load BM25 index: {e}")

    def _save_bm25_index(self):
        """Persist BM25 index to disk."""
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

        Returns:
            List of FAISS-assigned IDs.
        """
        if not (len(embeddings) == len(texts) == len(metadata_list)):
            raise ValueError("All inputs must have the same length")

        # Add to FAISS (returns authoritative IDs)
        faiss_ids = self.faiss_engine.add_documents(embeddings, metadata_list)

        # Enrich metadata with FAISS IDs so BM25 results map correctly
        enriched_metadata = [
            dict(m, id=fid) for m, fid in zip(metadata_list, faiss_ids)
        ]

        self.corpus.extend(texts)
        self.corpus_metadata.extend(enriched_metadata)

        # Rebuild BM25 on full corpus
        tokenized_corpus = [doc.lower().split() for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

        self._save_bm25_index()
        return faiss_ids

    # ── Reciprocal Rank Fusion ───────────────────────────────────────

    @staticmethod
    def _reciprocal_rank_fusion(
        semantic_results: List[Tuple[int, float]],
        keyword_results: List[Tuple[int, float]],
        k: int = 60,
    ) -> List[Tuple[int, float]]:
        """
        Merge results using RRF.  score = Σ 1/(k + rank)  for each list.
        """
        scores: Dict[int, float] = {}

        for rank, (doc_id, _) in enumerate(semantic_results):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)

        for rank, (doc_id, _) in enumerate(keyword_results):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # ── Search ───────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        query_embedding: np.ndarray,
        k: int = 5,
        semantic_k: int = 20,
        keyword_k: int = 20,
        rerank_candidates: Optional[int] = None,
        enable_reranking: Optional[bool] = None,
    ) -> List[Tuple[DocumentMetadata, float, str]]:
        """
        Hybrid search combining semantic, keyword, and cross-encoder reranking.

        Pipeline:
          1. FAISS semantic search (k=semantic_k)
          2. BM25 keyword search (k=keyword_k)
          3. RRF fusion to merge candidates
          4. NVIDIA cross-encoder reranking for final precision

        Args:
            query: The search query text
            query_embedding: Pre-computed query embedding
            k: Final number of results to return
            semantic_k: Number of semantic candidates to retrieve
            keyword_k: Number of keyword candidates to retrieve
            rerank_candidates: Number of candidates to rerank (default from settings)
            enable_reranking: Override setting for reranking

        Returns:
            List of (metadata, score, source_type) tuples sorted by relevance.
            Score is reranker logit if reranking enabled, otherwise RRF score.
        """
        if rerank_candidates is None:
            rerank_candidates = self.settings.rerank_candidates
        if enable_reranking is None:
            enable_reranking = self.settings.enable_reranking

        # ── Stage 1a: Semantic search via FAISS ──
        semantic_raw = self.faiss_engine.search(query_embedding, k=semantic_k)
        semantic_results = [(meta.id, similarity) for meta, similarity in semantic_raw]

        # ── Stage 1b: Keyword search via BM25 ──
        keyword_results: List[Tuple[int, float]] = []
        if self.bm25 and query.strip():
            tokenized_query = query.lower().split()
            bm25_scores = self.bm25.get_scores(tokenized_query)
            top_indices = np.argsort(bm25_scores)[::-1][:keyword_k]
            keyword_results = [
                (self.corpus_metadata[idx]["id"], float(bm25_scores[idx]))
                for idx in top_indices
                if bm25_scores[idx] > 0 and "id" in self.corpus_metadata[idx]
            ]

        # ── Stage 2: RRF fusion ──
        fused = self._reciprocal_rank_fusion(semantic_results, keyword_results)[
            :rerank_candidates
        ]

        semantic_ids = {r[0] for r in semantic_results}
        keyword_ids = {r[0] for r in keyword_results}

        # ── Helper: batch-fetch rows from SQLite ──
        all_candidate_ids = [doc_id for doc_id, _ in fused]
        rows_by_id: Dict[int, tuple] = {}
        if all_candidate_ids:
            conn = sqlite3.connect(self.faiss_engine.db_path)
            cursor = conn.cursor()
            placeholders = ",".join("?" for _ in all_candidate_ids)
            cursor.execute(
                f"SELECT id, filename, page_num, chunk_index, text, source, file_type "
                f"FROM chunks WHERE id IN ({placeholders})",
                all_candidate_ids,
            )
            for row in cursor.fetchall():
                rows_by_id[row[0]] = row
            conn.close()

        # ── Stage 3: NVIDIA Reranking (if enabled and available) ──
        reranked_order = None
        if enable_reranking and self.reranker and len(fused) > 1:
            try:
                passages = []
                doc_ids = []
                for doc_id, rrf_score in fused:
                    row = rows_by_id.get(doc_id)
                    if row:
                        passages.append(
                            {
                                "text": row[4],
                                "metadata": {
                                    "id": row[0],
                                    "filename": row[1],
                                    "page_num": row[2],
                                    "chunk_index": row[3],
                                    "source": row[5],
                                    "file_type": row[6],
                                },
                            }
                        )
                        doc_ids.append(doc_id)

                if passages:
                    rerank_results = self.reranker.rerank(
                        query=query,
                        passages=passages,
                        top_k=min(k, len(passages)),
                    )
                    reranked_order = []
                    for result in rerank_results:
                        original_idx = result.index
                        if original_idx < len(doc_ids):
                            reranked_order.append(
                                (
                                    doc_ids[original_idx],
                                    result.logit,
                                    passages[original_idx]["metadata"],
                                )
                            )
                    if reranked_order:
                        logger.info(
                            f"Reranked {len(passages)} candidates, "
                            f"top logit={reranked_order[0][1]:.2f}"
                        )
            except Exception as e:
                logger.warning(f"Reranking failed, falling back to RRF: {e}")
                reranked_order = None

        # ── Stage 4: Build final results (no extra DB queries) ──
        results = []

        if reranked_order:
            for doc_id, logit, _ in reranked_order[:k]:
                if doc_id in semantic_ids and doc_id in keyword_ids:
                    source_type = "hybrid_reranked"
                elif doc_id in semantic_ids:
                    source_type = "semantic_reranked"
                else:
                    source_type = "keyword_reranked"

                row = rows_by_id.get(doc_id)
                if row:
                    results.append(
                        (
                            DocumentMetadata(
                                id=row[0],
                                filename=row[1],
                                page_num=row[2],
                                chunk_index=row[3],
                                text=row[4],
                                source=row[5],
                                file_type=row[6],
                            ),
                            logit,
                            source_type,
                        )
                    )
        else:
            for doc_id, rrf_score in fused[:k]:
                row = rows_by_id.get(doc_id)
                if not row:
                    continue

                if doc_id in semantic_ids and doc_id in keyword_ids:
                    source_type = "hybrid"
                elif doc_id in semantic_ids:
                    source_type = "semantic"
                else:
                    source_type = "keyword"

                results.append(
                    (
                        DocumentMetadata(
                            id=row[0],
                            filename=row[1],
                            page_num=row[2],
                            chunk_index=row[3],
                            text=row[4],
                            source=row[5],
                            file_type=row[6],
                        ),
                        rrf_score,
                        source_type,
                    )
                )

        return results

    def get_stats(self) -> Dict:
        return {
            "faiss_documents": self.faiss_engine.get_document_count(),
            "bm25_documents": len(self.corpus),
            "faiss_index_path": str(self.faiss_engine.index_path),
            "bm25_index_path": str(self.bm25_index_path),
            "reranker_enabled": self.settings.enable_reranking,
            "reranker_model": self.settings.rerank_model
            if self.settings.enable_reranking
            else None,
        }
