from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    environment: str = "local"
    log_level: str = "INFO"
    api_url: str = "http://127.0.0.1:8000"
    api_key: str = "change-me"
    cors_allow_origins: str = (
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:8501"
    )

    database_url: str = "postgresql+psycopg://oba:oba@localhost:5432/oba"
    redis_url: str = "redis://localhost:6379/0"

    data_dir: Path = Field(default_factory=lambda: Path("data"))
    emb_model_name: str = "intfloat/e5-base-v2"
    reranker_name: str = "BAAI/bge-reranker-base"
    emb_device: str = "cpu"
    rerank_device: str = "cpu"
    enable_embedding_writes: bool = False
    enable_postgres_native_retrieval: bool = True
    enable_semantic_retrieval: bool = True
    enable_reranking: bool = False
    retrieval_chunk_limit: int = 1000

    llm_provider: str = "ollama"
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.1:8b"
    openai_base_url: str | None = None
    openai_api_key: str | None = None
    openai_model: str | None = None

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_allow_origins.split(",") if origin.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
