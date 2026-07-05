"""
Application configuration.

All settings are loaded from environment variables (and a local .env file
during development) via pydantic-settings. Import `settings` anywhere
in the app instead of reading os.environ directly, so there is a single
source of truth and validation happens at startup.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # --- Ollama (local LLM, no API key, no cost) ---
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    ollama_max_tokens: int = 1024

    # --- Embeddings ---
    embedding_model_name: str = "all-MiniLM-L6-v2"

    # --- Chunking ---
    chunk_size: int = 500
    chunk_overlap: int = 50

    # --- Vector store ---
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "documind"

    # --- Retrieval ---
    retrieval_top_k: int = 5
    rerank_top_n: int = 3

    # --- Uploads ---
    upload_dir: str = "./uploads"
    max_upload_size_mb: int = 25

    # --- Server ---
    cors_allow_origins: list[str] = ["*"]


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor so the .env file is only parsed once."""
    return Settings()


settings = get_settings()
