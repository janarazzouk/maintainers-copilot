from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    vault_addr: str = Field(
        default="http://127.0.0.1:8200",
        alias="VAULT_ADDR",
    )
    vault_dev_root_token_id: str = Field(
        default="root",
        alias="VAULT_DEV_ROOT_TOKEN_ID",
    )

    redis_url: str | None = None
    api_url: str | None = None

   

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
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
    


@lru_cache
def get_settings() -> Settings:
    return Settings()