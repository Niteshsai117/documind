from sentence_transformers import SentenceTransformer

from config import settings


class EmbeddingService:
    def __init__(self) -> None:
        self._model = SentenceTransformer(settings.embedding_model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        embedding = self._model.encode(query, show_progress_bar=False, convert_to_numpy=True)
        return embedding.tolist()


embedding_service = EmbeddingService()
