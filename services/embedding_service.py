"""
Text embeddings via sentence-transformers.

The model is loaded once at process startup and reused for every embed
call — loading it per-request would add seconds of latency to every
upload and query. all-MiniLM-L6-v2 is small (~80MB), fast on CPU, and
produces 384-dimensional embeddings, which is plenty for similarity
search over document chunks.
"""

from sentence_transformers import SentenceTransformer

from config import settings


class EmbeddingService:
    def __init__(self) -> None:
        self._model = SentenceTransformer(settings.embedding_model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document chunks."""
        embeddings = self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed a single user query."""
        embedding = self._model.encode(query, show_progress_bar=False, convert_to_numpy=True)
        return embedding.tolist()


embedding_service = EmbeddingService()
