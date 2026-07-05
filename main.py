"""
DocuMind — RAG-based Document Q&A System.

FastAPI application exposing:
  POST /upload  - upload a PDF, chunk it, embed it, store it in ChromaDB
  POST /ask     - ask a question about a previously uploaded document
  GET  /health  - liveness/readiness check

Swagger UI is available at /docs (FastAPI default), and a minimal
frontend is served at /.
"""

import logging
import os
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import settings
from services.embedding_service import embedding_service
from services.pdf_service import PDFProcessingError, pdf_service
from services.qa_service import NoRelevantContextError, OllamaUnavailableError, qa_service
from services.retrieval_service import retrieval_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("documind")

app = FastAPI(
    title="DocuMind",
    description="RAG-based Document Q&A System powered by a local Ollama model.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")


# --- Schemas ---

class UploadResponse(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    message: str


class AskRequest(BaseModel):
    document_id: str = Field(..., description="ID returned by /upload")
    question: str = Field(..., min_length=1, max_length=2000)


class Source(BaseModel):
    page_number: int | None
    score: float
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


class HealthResponse(BaseModel):
    status: str
    service: str


# --- Routes ---

@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    return FileResponse("static/index.html")


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="documind")


@app.post("/upload", response_model=UploadResponse, tags=["documents"])
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    if file.content_type != "application/pdf" and not (file.filename or "").lower().endswith(
        ".pdf"
    ):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    document_id = uuid.uuid4().hex
    temp_path = Path(settings.upload_dir) / f"{document_id}.pdf"

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    size = 0
    try:
        with open(temp_path, "wb") as out_file:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds the {settings.max_upload_size_mb}MB limit.",
                    )
                out_file.write(chunk)

        try:
            chunks = pdf_service.chunk_document(temp_path)
        except PDFProcessingError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        embeddings = embedding_service.embed_texts([c.text for c in chunks])
        retrieval_service.add_chunks(
            document_id=document_id,
            filename=file.filename or "document.pdf",
            chunks=chunks,
            embeddings=embeddings,
        )

        logger.info("Indexed document %s (%s) with %d chunks", document_id, file.filename, len(chunks))

        return UploadResponse(
            document_id=document_id,
            filename=file.filename or "document.pdf",
            chunk_count=len(chunks),
            message="Document indexed successfully. Use the document_id with /ask.",
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - convert unexpected failures into a 500
        logger.exception("Failed to process upload")
        raise HTTPException(status_code=500, detail=f"Failed to process document: {exc}") from exc
    finally:
        if temp_path.exists():
            os.remove(temp_path)


@app.post("/ask", response_model=AskResponse, tags=["qa"])
def ask_question(request: AskRequest) -> AskResponse:
    try:
        result = qa_service.answer_question(
            question=request.question,
            document_id=request.document_id,
        )
        return AskResponse(**result)
    except NoRelevantContextError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except OllamaUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to answer question")
        raise HTTPException(status_code=500, detail=f"Failed to answer question: {exc}") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
