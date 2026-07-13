from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    ollama_max_tokens: int = 1024

    embedding_model_name: str = "all-MiniLM-L6-v2"

    chunk_size: int = 500
    chunk_overlap: int = 50

    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "documind"

    retrieval_top_k: int = 5
    rerank_top_n: int = 3

    upload_dir: str = "./uploads"
    max_upload_size_mb: int = 25

    cors_allow_origins: list[str] = ["*"]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
