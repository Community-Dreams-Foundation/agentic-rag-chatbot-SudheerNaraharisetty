"""
FAISS Vector Engine with SQLite sidecar for metadata.
Implements exact nearest neighbor search (IndexFlatL2) for 100% accuracy.
"""

import pickle
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import faiss

from src.core.config import get_settings


class DocumentMetadata:
    """Represents metadata for a document chunk."""

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
    High-performance FAISS vector engine with SQLite metadata sidecar.
    Uses IndexFlatL2 for mathematically exact nearest neighbor search.
    """

    def __init__(
        self,
        index_path: Optional[Path] = None,
        db_path: Optional[Path] = None,
        dimension: int = 1536,  # OpenAI embedding dimension
    ):
        self.settings = get_settings()
        self.index_path = index_path or self.settings.faiss_index_path
        self.db_path = db_path or self.settings.sqlite_db_path
        self.dimension = dimension

        # Initialize FAISS index
        self.index = self._init_faiss_index()

        # Initialize SQLite
        self._init_sqlite()

        # Track next ID
        self.next_id = self.index.ntotal

    def _init_faiss_index(self) -> faiss.Index:
        """Initialize or load FAISS index."""
        if self.index_path.exists():
            print(f"Loading existing FAISS index from {self.index_path}")
            return faiss.read_index(str(self.index_path))
        else:
            print("Creating new FAISS IndexFlatL2 (100% exact search)")
            # IndexFlatL2 provides exact nearest neighbor search
            return faiss.IndexFlatL2(self.dimension)

    def _init_sqlite(self):
        """Initialize SQLite database for metadata."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
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
        """)

        # Create index for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_filename ON chunks(filename)
        """)

        conn.commit()
        conn.close()

    def add_documents(
        self, embeddings: np.ndarray, metadata_list: List[Dict[str, Any]]
    ) -> List[int]:
        """
        Add documents to FAISS index and metadata to SQLite.

        Args:
            embeddings: Array of embedding vectors (shape: n_docs x dimension)
            metadata_list: List of metadata dicts for each document

        Returns:
            List of assigned IDs
        """
        if len(embeddings) != len(metadata_list):
            raise ValueError("Embeddings and metadata must have same length")

        # Convert to float32 (FAISS requirement)
        vectors = np.array(embeddings).astype("float32")
        n_docs = len(vectors)

        # Generate IDs
        start_id = self.next_id
        ids = list(range(start_id, start_id + n_docs))

        # Add to FAISS
        self.index.add(vectors)

        # Add to SQLite
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        data_to_insert = []
        for i, meta in enumerate(metadata_list):
            faiss_id = start_id + i
            data_to_insert.append(
                (
                    faiss_id,
                    meta.get("filename", "unknown"),
                    meta.get("page_num", 1),
                    meta.get("chunk_index", 0),
                    meta.get("text", ""),
                    meta.get("source", ""),
                    meta.get("file_type", "unknown"),
                )
            )

        cursor.executemany(
            "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))",
            data_to_insert,
        )

        conn.commit()
        conn.close()

        # Update next_id
        self.next_id += n_docs

        # Save index to disk
        self._save_index()

        return ids

    def search(
        self, query_vector: np.ndarray, k: int = 5
    ) -> List[Tuple[DocumentMetadata, float]]:
        """
        Search for nearest neighbors.

        Args:
            query_vector: Query embedding vector
            k: Number of results to return

        Returns:
            List of (metadata, distance) tuples sorted by distance
        """
        # Ensure query is float32
        query = np.array([query_vector]).astype("float32")

        # Search FAISS
        distances, indices = self.index.search(query, k)

        # Retrieve metadata from SQLite
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:  # No match found
                continue

            cursor.execute(
                "SELECT id, filename, page_num, chunk_index, text, source, file_type FROM chunks WHERE id=?",
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
                results.append((metadata, float(dist)))

        conn.close()
        return results

    def get_document_count(self) -> int:
        """Get total number of documents in index."""
        return self.index.ntotal

    def _save_index(self):
        """Save FAISS index to disk."""
        faiss.write_index(self.index, str(self.index_path))

    def reset(self):
        """Clear all data (use with caution)."""
        # Reset FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)
        self._save_index()

        # Clear SQLite
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chunks")
        conn.commit()
        conn.close()

        self.next_id = 0


if __name__ == "__main__":
    # Test the engine
    engine = FaissEngine(dimension=1536)

    # Add test documents
    test_embeddings = np.random.randn(5, 1536).astype("float32")
    test_metadata = [
        {
            "filename": f"test_{i}.pdf",
            "page_num": i + 1,
            "chunk_index": 0,
            "text": f"This is test document {i}",
            "source": f"test_{i}.pdf",
            "file_type": "pdf",
        }
        for i in range(5)
    ]

    ids = engine.add_documents(test_embeddings, test_metadata)
    print(f"Added {len(ids)} documents with IDs: {ids}")

    # Search
    query = np.random.randn(1536).astype("float32")
    results = engine.search(query, k=3)
    print(f"\nSearch results:")
    for metadata, distance in results:
        print(
            f"  - {metadata.filename} (page {metadata.page_num}): distance={distance:.4f}"
        )
