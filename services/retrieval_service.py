"""
ChromaDB vector store operations.

Wraps a single persistent Chroma collection. Every chunk is stored with
a `document_id` in its metadata so multiple uploaded PDFs can share one
collection while queries can still be scoped to a specific document.
"""

import uuid

import chromadb
from chromadb.config import Settings as ChromaSettings

from config import settings
from services.pdf_service import Chunk


class RetrievalService:
    def __init__(self) -> None:
        self._client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        document_id: str,
        filename: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        """Store a document's chunks and their precomputed embeddings."""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be the same length")

        self._collection.add(
            ids=[f"{document_id}::{c.chunk_index}::{uuid.uuid4().hex[:8]}" for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=[
                {
                    "document_id": document_id,
                    "filename": filename,
                    "chunk_index": c.chunk_index,
                    "page_number": c.page_number if c.page_number is not None else -1,
                }
                for c in chunks
            ],
        )

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int | None = None,
        document_id: str | None = None,
    ) -> list[dict]:
        """Return the top_k most similar chunks, optionally scoped to one document."""
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k or settings.retrieval_top_k,
            where={"document_id": document_id} if document_id else None,
        )

        if not results["documents"] or not results["documents"][0]:
            return []

        hits = []
        for text, metadata, distance in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            hits.append(
                {
                    "text": text,
                    "filename": metadata.get("filename"),
                    "page_number": metadata.get("page_number"),
                    "chunk_index": metadata.get("chunk_index"),
                    # cosine distance -> similarity score in [0, 1], higher is better
                    "score": 1.0 - distance,
                }
            )
        return hits

    def document_exists(self, document_id: str) -> bool:
        existing = self._collection.get(where={"document_id": document_id}, limit=1)
        return len(existing["ids"]) > 0

    def delete_document(self, document_id: str) -> None:
        self._collection.delete(where={"document_id": document_id})


retrieval_service = RetrievalService()
