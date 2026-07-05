"""
RAG pipeline: retrieve relevant chunks, then ask a local Ollama model to
answer grounded only in that context.

Uses Ollama (https://ollama.com) running on localhost — no API key, no
per-request cost, no data leaves the machine. Pull the model once with
`ollama pull llama3` before running the app.
"""

import httpx
import ollama

from config import settings
from services.embedding_service import embedding_service
from services.retrieval_service import retrieval_service

SYSTEM_PROMPT = (
    "You are DocuMind, a document question-answering assistant. "
    "Answer the user's question using ONLY the information in the provided "
    "context excerpts from their document. "
    "If the context does not contain enough information to answer, say so "
    "plainly instead of guessing. "
    "When useful, mention which page an answer came from. "
    "Be concise and accurate."
)


class NoRelevantContextError(Exception):
    """Raised when no chunks are found for the requested document."""


class OllamaUnavailableError(Exception):
    """Raised when the local Ollama server can't be reached, or the model isn't pulled."""


class QAService:
    def __init__(self) -> None:
        self._client = ollama.Client(host=settings.ollama_host)

    def _build_context(self, hits: list[dict]) -> str:
        sections = []
        for i, hit in enumerate(hits, start=1):
            page = hit.get("page_number")
            page_label = f", page {page}" if page and page != -1 else ""
            sections.append(f"[Excerpt {i}{page_label}]\n{hit['text']}")
        return "\n\n".join(sections)

    def answer_question(self, question: str, document_id: str) -> dict:
        """Retrieve relevant chunks for document_id and generate an answer."""
        if not retrieval_service.document_exists(document_id):
            raise NoRelevantContextError(
                f"No document found with id '{document_id}'. Upload it first via /upload."
            )

        query_embedding = embedding_service.embed_query(question)
        hits = retrieval_service.similarity_search(
            query_embedding=query_embedding,
            document_id=document_id,
        )

        if not hits:
            raise NoRelevantContextError("No relevant content found in the document.")

        context = self._build_context(hits)
        user_message = (
            f"Context excerpts from the document:\n\n{context}\n\n"
            f"Question: {question}"
        )

        try:
            response = self._client.chat(
                model=settings.ollama_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                options={"num_predict": settings.ollama_max_tokens},
            )
        except ollama.ResponseError as exc:
            raise OllamaUnavailableError(
                f"Ollama model '{settings.ollama_model}' is unavailable: {exc.error}. "
                f"Try: ollama pull {settings.ollama_model}"
            ) from exc
        except httpx.ConnectError as exc:
            raise OllamaUnavailableError(
                f"Could not reach Ollama at {settings.ollama_host}. "
                "Make sure the Ollama server is running (`ollama serve`)."
            ) from exc

        answer_text = response["message"]["content"]

        return {
            "answer": answer_text,
            "sources": [
                {
                    "page_number": hit.get("page_number"),
                    "score": round(hit.get("score", 0.0), 4),
                    "excerpt": hit["text"][:200],
                }
                for hit in hits
            ],
        }


qa_service = QAService()
