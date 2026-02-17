"""
Document Processor: Handles PDF, text, and markdown document parsing.
Implements intelligent chunking with metadata extraction.
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.config import get_settings


@dataclass
class DocumentChunk:
    """Represents a document chunk with metadata."""

    text: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "metadata": self.metadata}


class DocumentProcessor:
    """
    Processes various document types and creates chunks for indexing.
    """

    def __init__(self):
        self.settings = get_settings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def process_file(self, file_path: Path) -> List[DocumentChunk]:
        """
        Process a file and return chunks.

        Args:
            file_path: Path to the file

        Returns:
            List of DocumentChunk objects
        """
        file_path = Path(file_path)
        file_type = file_path.suffix.lower()

        if file_type == ".pdf":
            return self._process_pdf(file_path)
        elif file_type in [".txt", ".md", ".markdown"]:
            return self._process_text(file_path)
        elif file_type in [".html", ".htm"]:
            return self._process_html(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    def _process_pdf(self, file_path: Path) -> List[DocumentChunk]:
        """Process PDF file with page tracking."""
        chunks = []

        with open(file_path, "rb") as file:
            pdf_reader = pypdf.PdfReader(file)

            for page_num, page in enumerate(pdf_reader.pages, start=1):
                # Extract text from page
                text = page.extract_text()

                if not text.strip():
                    continue

                # Split page text into chunks
                page_chunks = self.text_splitter.create_documents(
                    texts=[text],
                    metadatas=[
                        {
                            "filename": file_path.name,
                            "page_num": page_num,
                            "source": str(file_path),
                            "file_type": "pdf",
                        }
                    ],
                )

                # Convert to DocumentChunk
                for i, doc in enumerate(page_chunks):
                    chunks.append(
                        DocumentChunk(
                            text=doc.page_content,
                            metadata={
                                **doc.metadata,
                                "chunk_index": i,
                                "total_pages": len(pdf_reader.pages),
                            },
                        )
                    )

        return chunks

    def _process_text(self, file_path: Path) -> List[DocumentChunk]:
        """Process text/markdown file."""
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

        # Create chunks
        text_chunks = self.text_splitter.create_documents(
            texts=[text],
            metadatas=[
                {
                    "filename": file_path.name,
                    "page_num": 1,
                    "source": str(file_path),
                    "file_type": file_path.suffix[1:],
                }
            ],
        )

        return [
            DocumentChunk(
                text=doc.page_content, metadata={**doc.metadata, "chunk_index": i}
            )
            for i, doc in enumerate(text_chunks)
        ]

    def _process_html(self, file_path: Path) -> List[DocumentChunk]:
        """Process HTML file (basic text extraction)."""
        from html.parser import HTMLParser

        class MLStripper(HTMLParser):
            def __init__(self):
                super().__init__()
                self.reset()
                self.fed = []

            def handle_data(self, d):
                self.fed.append(d)

            def get_data(self):
                return "".join(self.fed)

        with open(file_path, "r", encoding="utf-8") as file:
            html = file.read()

        # Strip HTML tags
        stripper = MLStripper()
        stripper.feed(html)
        text = stripper.get_data()

        # Clean up whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Create chunks
        text_chunks = self.text_splitter.create_documents(
            texts=[text],
            metadatas=[
                {
                    "filename": file_path.name,
                    "page_num": 1,
                    "source": str(file_path),
                    "file_type": "html",
                }
            ],
        )

        return [
            DocumentChunk(
                text=doc.page_content, metadata={**doc.metadata, "chunk_index": i}
            )
            for i, doc in enumerate(text_chunks)
        ]

    def process_bytes(
        self, content: bytes, filename: str, content_type: Optional[str] = None
    ) -> List[DocumentChunk]:
        """
        Process file content from bytes (for uploaded files).

        Args:
            content: File content as bytes
            filename: Original filename
            content_type: MIME type (optional)

        Returns:
            List of DocumentChunk objects
        """
        file_path = Path(filename)
        file_type = file_path.suffix.lower()

        # Write to temp file for processing
        import tempfile
        import os

        temp_dir = tempfile.gettempdir()
        temp_path = Path(temp_dir) / filename

        try:
            with open(temp_path, "wb") as f:
                f.write(content)

            chunks = self.process_file(temp_path)

            # Update metadata to use original filename
            for chunk in chunks:
                chunk.metadata["filename"] = filename
                chunk.metadata["source"] = filename

            return chunks

        finally:
            # Clean up temp file
            if temp_path.exists():
                os.remove(temp_path)


if __name__ == "__main__":
    # Test the processor
    processor = DocumentProcessor()

    # Create a test text file
    test_content = """
    # Test Document
    
    This is the first paragraph. It contains some important information.
    
    This is the second paragraph. It has more details about the topic.
    
    ## Section 2
    
    Here is another section with additional content.
    """

    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(test_content)
        temp_file = f.name

    try:
        chunks = processor.process_file(Path(temp_file))
        print(f"Created {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks, 1):
            print(f"\nChunk {i}:")
            print(f"  Text: {chunk.text[:100]}...")
            print(f"  Metadata: {chunk.metadata}")
    finally:
        import os

        os.remove(temp_file)
