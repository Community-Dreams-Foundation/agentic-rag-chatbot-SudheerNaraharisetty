"""
FAISS Vector Engine with SQLite metadata sidecar.
Uses IndexFlatIP (inner product) on L2-normalized vectors for cosine similarity.

Supports dynamic embedding dimensions:
  - OpenRouter (qwen3-embedding-8b): 4096 dimensions
  - NVIDIA NIM (llama-3.2-nv-embedqa-1b-v2): 2048 dimensions
"""

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

from src.core.config import get_settings


class DocumentMetadata:
    """Represents metadata for a document chunk."""

    __slots__ = (
        "id",
        "filename",
        "page_num",
        "chunk_index",
        "text",
        "source",
        "file_type",
    )

    def __init__(
        self,
        id: int,
        filename: str,
        page_num: int,
        chunk_index: int,
        text: str,
        source: str,
        file_type: str,
    ):
        self.id = id
        self.filename = filename
        self.page_num = page_num
        self.chunk_index = chunk_index
        self.text = text
        self.source = source
        self.file_type = file_type

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "filename": self.filename,
            "page_num": self.page_num,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "source": self.source,
            "file_type": self.file_type,
        }


class FaissEngine:
    """
    FAISS vector engine using IndexFlatIP (inner product) on L2-normalized
    vectors, which is mathematically equivalent to cosine similarity.
    Paired with SQLite sidecar for chunk metadata storage.

    Dimension is determined by the active embedding provider:
      - OpenRouter (qwen3-embedding-8b): 4096
      - NVIDIA NIM (llama-3.2-nv-embedqa): 2048
    """

    def __init__(
        self,
        index_path: Optional[Path] = None,
        db_path: Optional[Path] = None,
        dimension: Optional[int] = None,
    ):
        self.settings = get_settings()
        self.index_path = index_path or self.settings.faiss_index_path
        self.db_path = db_path or self.settings.sqlite_db_path
        # Use dynamic dimension from config (property based on active provider)
        self.dimension = dimension or self.settings.embedding_dimension

        self.index = self._init_faiss_index()
        self._init_sqlite()
        self.next_id = self.index.ntotal

    def _init_faiss_index(self) -> faiss.Index:
        """Initialize or load FAISS index, migrating L2→IP if needed."""
        if self.index_path.exists():
            loaded = faiss.read_index(str(self.index_path))
            # Validate dimension compatibility
            if loaded.d != self.dimension:
                print(
                    f"Index dimension mismatch ({loaded.d} vs {self.dimension}). "
                    "Creating fresh index."
                )
                return faiss.IndexFlatIP(self.dimension)
            return loaded
        else:
            print(f"Creating FAISS IndexFlatIP (cosine similarity, {self.dimension}d)")
            return faiss.IndexFlatIP(self.dimension)

    def _init_sqlite(self):
        """Initialize SQLite database for metadata."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                filename TEXT NOT NULL,
                page_num INTEGER DEFAULT 1,
                chunk_index INTEGER DEFAULT 0,
                text TEXT NOT NULL,
                source TEXT NOT NULL,
                file_type TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_filename ON chunks(filename)"
        )
        conn.commit()
        conn.close()

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        """L2-normalize vectors so inner product == cosine similarity."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return vectors / norms

    def add_documents(
        self, embeddings: np.ndarray, metadata_list: List[Dict[str, Any]]
    ) -> List[int]:
        """
        Add documents to FAISS index and metadata to SQLite.
        Embeddings are L2-normalized before indexing.

        Returns:
            List of assigned integer IDs.
        """
        if len(embeddings) != len(metadata_list):
            raise ValueError("Embeddings and metadata must have same length")

        vectors = np.array(embeddings).astype("float32")
        vectors = self._normalize(vectors)
        n_docs = len(vectors)

        start_id = self.next_id
        ids = list(range(start_id, start_id + n_docs))

        self.index.add(vectors)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        rows = [
            (
                start_id + i,
                meta.get("filename", "unknown"),
                meta.get("page_num", 1),
                meta.get("chunk_index", 0),
                meta.get("text", ""),
                meta.get("source", ""),
                meta.get("file_type", "unknown"),
            )
            for i, meta in enumerate(metadata_list)
        ]
        cursor.executemany(
            "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))", rows
        )
        conn.commit()
        conn.close()

        self.next_id += n_docs
        self._save_index()
        return ids

    def search(
        self, query_vector: np.ndarray, k: int = 5
    ) -> List[Tuple[DocumentMetadata, float]]:
        """
        Search for nearest neighbors using cosine similarity.

        Returns:
            List of (metadata, similarity_score) sorted by relevance descending.
        """
        query = np.array([query_vector]).astype("float32")
        query = self._normalize(query)

        actual_k = min(k, self.index.ntotal) if self.index.ntotal > 0 else 0
        if actual_k == 0:
            return []

        similarities, indices = self.index.search(query, actual_k)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        results = []
        for idx, sim in zip(indices[0], similarities[0]):
            if idx == -1:
                continue
            cursor.execute(
                "SELECT id, filename, page_num, chunk_index, text, source, file_type "
                "FROM chunks WHERE id=?",
                (int(idx),),
            )
            row = cursor.fetchone()
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
                results.append((metadata, float(sim)))

        conn.close()
        return results

    def get_document_count(self) -> int:
        return self.index.ntotal

    def get_indexed_files(self) -> List[Dict[str, Any]]:
        """Return summary of all indexed files."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT filename, COUNT(*) as chunks, MIN(created_at) as indexed_at "
            "FROM chunks GROUP BY filename ORDER BY indexed_at DESC"
        )
        files = [
            {"filename": r[0], "chunks": r[1], "indexed_at": r[2]}
            for r in cursor.fetchall()
        ]
        conn.close()
        return files

    def delete_file(self, filename: str) -> int:
        """Delete all chunks for a filename from SQLite (FAISS requires rebuild)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chunks WHERE filename=?", (filename,))
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        return deleted

    def _save_index(self):
        faiss.write_index(self.index, str(self.index_path))

    def reset(self):
        """Clear all data and recreate empty index."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self._save_index()

        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM chunks")
        conn.commit()
        conn.close()
        self.next_id = 0
