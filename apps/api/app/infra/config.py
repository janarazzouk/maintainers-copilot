from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    vault_addr: str = Field(
        default="http://127.0.0.1:8200",
        alias="VAULT_ADDR",
    )
    vault_dev_root_token_id: str = Field(
        default="root",
        alias="VAULT_DEV_ROOT_TOKEN_ID",
    )
    vault_token: str | None = Field(default=None, alias="VAULT_TOKEN")
    vault_token_file: str | None = Field(default=None, alias="VAULT_TOKEN_FILE")

    redis_url: str | None = Field(default=None, alias="REDIS_URL")
    api_url: str | None = Field(default=None, alias="API_URL")

    jwt_access_token_minutes: int = Field(
        default=60 * 24,
        alias="JWT_ACCESS_TOKEN_MINUTES",
    )

    model_server_url: str = Field(
        default="http://127.0.0.1:8001",
        alias="MODEL_SERVER_URL",
    )

    rag_corpus_path: str = Field(
        default="data/rag/rag_issues_corpus.jsonl",
        alias="RAG_CORPUS_PATH",
    )
    rag_chunks_path: str = Field(
        default="data/rag/rag_issues_chunks.jsonl",
        alias="RAG_CHUNKS_PATH",
    )
    rag_golden_path: str = Field(
        default="data/rag/rag_golden_draft.jsonl",
        alias="RAG_GOLDEN_PATH",
    )

    embedding_model_name: str = Field(
        default="BAAI/bge-small-en-v1.5",
        alias="EMBEDDING_MODEL_NAME",
    )
    embedding_batch_size: int = Field(
        default=32,
        alias="EMBEDDING_BATCH_SIZE",
    )
    embedding_cache_dir: str = Field(
        default="/models/fastembed",
        alias="EMBEDDING_CACHE_DIR",
    )

    groq_base_url: str = Field(
        default="https://api.groq.com/openai/v1",
        alias="GROQ_BASE_URL",
    )
    groq_model: str = Field(
        default="llama-3.1-8b-instant",
        alias="GROQ_MODEL",
    )
    groq_temperature: float = Field(
        default=0.2,
        alias="GROQ_TEMPERATURE",
    )
    groq_max_tokens: int = Field(
        default=700,
        alias="GROQ_MAX_TOKENS",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()