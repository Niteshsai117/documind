import re

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
    "plainly instead of guessing — never invent facts not present in the excerpts. "
    "Every excerpt is labeled with its page number: cite the page number in "
    "parentheses, e.g. '(page 3)', immediately after each claim you draw from it. "
    "If a claim draws on multiple pages, cite all of them. "
    "Keep answers short — a few sentences at most — and skip preamble like "
    "'Based on the document'; just answer directly."
)

_WHITESPACE_RE = re.compile(r"\s+")


class NoRelevantContextError(Exception):
    pass


class OllamaUnavailableError(Exception):
    pass


class QAService:
    def __init__(self) -> None:
        self._client = ollama.Client(host=settings.ollama_host)

    def _normalize_query(self, question: str) -> str:
        return _WHITESPACE_RE.sub(" ", question).strip()

    def _rerank(self, hits: list[dict], top_n: int) -> list[dict]:
        return sorted(hits, key=lambda hit: hit.get("score", 0.0), reverse=True)[:top_n]

    def _build_context(self, hits: list[dict]) -> str:
        sections = []
        for i, hit in enumerate(hits, start=1):
            page = hit.get("page_number")
            page_label = f", page {page}" if page and page != -1 else ""
            sections.append(f"[Excerpt {i}{page_label}]\n{hit['text']}")
        return "\n\n".join(sections)

    def answer_question(self, question: str, document_id: str) -> dict:
        if not retrieval_service.document_exists(document_id):
            raise NoRelevantContextError(
                f"No document found with id '{document_id}'. Upload it first via /upload."
            )

        query = self._normalize_query(question)
        query_embedding = embedding_service.embed_query(query)
        candidates = retrieval_service.similarity_search(
            query_embedding=query_embedding,
            document_id=document_id,
        )

        if not candidates:
            raise NoRelevantContextError("No relevant content found in the document.")

        hits = self._rerank(candidates, top_n=settings.rerank_top_n)

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
