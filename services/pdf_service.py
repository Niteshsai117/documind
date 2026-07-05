"""
PDF parsing and chunking.

Extracts raw text from a PDF with PyPDF2, then splits it into overlapping
chunks with LangChain's RecursiveCharacterTextSplitter. Overlap preserves
context across chunk boundaries so a fact split across two chunks is still
retrievable from either one.
"""

from dataclasses import dataclass
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError

from config import settings


class PDFProcessingError(Exception):
    """Raised when a PDF cannot be read or contains no extractable text."""


@dataclass
class Chunk:
    text: str
    chunk_index: int
    page_number: int | None


class PDFService:
    def __init__(self) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def extract_text_by_page(self, pdf_path: str | Path) -> list[str]:
        """Return a list of extracted text, one entry per PDF page."""
        try:
            reader = PdfReader(str(pdf_path))
        except (PdfReadError, OSError) as exc:
            raise PDFProcessingError(f"Could not open PDF: {exc}") from exc

        if reader.is_encrypted:
            raise PDFProcessingError("PDF is password-protected and cannot be read.")

        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception as exc:  # PyPDF2 can raise various parsing errors per-page
                raise PDFProcessingError(f"Failed to extract text from page: {exc}") from exc

        if not any(page_text.strip() for page_text in pages):
            raise PDFProcessingError(
                "No extractable text found in PDF (it may be a scanned/image-only document)."
            )

        return pages

    def chunk_document(self, pdf_path: str | Path) -> list[Chunk]:
        """Extract text from a PDF and split it into overlapping chunks."""
        pages = self.extract_text_by_page(pdf_path)

        chunks: list[Chunk] = []
        chunk_index = 0
        for page_number, page_text in enumerate(pages, start=1):
            if not page_text.strip():
                continue
            for piece in self._splitter.split_text(page_text):
                if not piece.strip():
                    continue
                chunks.append(
                    Chunk(text=piece, chunk_index=chunk_index, page_number=page_number)
                )
                chunk_index += 1

        if not chunks:
            raise PDFProcessingError("PDF produced no usable text chunks.")

        return chunks


pdf_service = PDFService()
