from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app.api.routers.auth import router as auth_router
from app.api.routers.model_tools import router as model_tools_router
from app.api.routers.rag import router as rag_router
from app.api.routers.widgets import admin_router as widget_admin_router
from app.api.routers.widgets import public_router as widget_public_router
from app.api.routers.chat import router as chat_router
from app.infra.redis import RedisShortTermMemory
from app.infra.config import get_settings
from app.infra.database import check_database_connection, init_database
from app.infra.embeddings import EmbeddingModel
from app.infra.groq_client import GroqLLMClient
from app.infra.minio import MinIOObjectStore
from app.infra.model_server_client import ModelServerClient
from app.infra.vault import VaultClient, resolve_vault_token


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()

    vault = VaultClient(
        addr=settings.vault_addr,
        token=resolve_vault_token(settings),
    )
    app_secrets = vault.read_app_secrets()

    jwt_signing_key = app_secrets.get("jwt_signing_key")
    if not jwt_signing_key:
        raise RuntimeError("Vault secret/app is missing jwt_signing_key.")

    init_database(str(app_secrets["database_url"]))
    check_database_connection()

    object_store = MinIOObjectStore.from_secrets(app_secrets)
    object_store.ensure_bucket()

    app.state.embedding_model = EmbeddingModel(
        model_name=settings.embedding_model_name,
        cache_dir=settings.embedding_cache_dir,
    )
    app.state.model_server_client = ModelServerClient(
        base_url=settings.model_server_url,
    )
    app.state.object_store = object_store
    app.state.jwt_signing_key = str(jwt_signing_key)
    app.state.jwt_access_token_minutes = settings.jwt_access_token_minutes
    app.state.secrets = app_secrets

    groq_api_key = app_secrets.get("groq_api_key")
    app.state.groq_llm_client = None
    if groq_api_key:
        app.state.groq_llm_client = GroqLLMClient(
            api_key=str(groq_api_key),
            base_url=settings.groq_base_url,
            model=settings.groq_model,
            temperature=settings.groq_temperature,
            max_tokens=settings.groq_max_tokens,
        )

    short_term_memory = RedisShortTermMemory(
        redis_url=settings.redis_url,
        ttl_seconds=settings.chat_short_term_ttl_seconds,
    )
    short_term_memory.ping()
    app.state.short_term_memory = short_term_memory    

    yield


app = FastAPI(
    title="Maintainer's Copilot API",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "api"}


@app.get("/debug/secrets-check")
def secrets_check() -> dict[str, object]:
    return {
        "vault_loaded": True,
        "available_secret_keys": sorted(app.state.secrets.keys()),
    }


@app.get("/debug/db-check")
def db_check() -> dict[str, bool]:
    check_database_connection()
    return {"database_connected": True}


@app.get("/debug/minio-check")
def minio_check() -> dict[str, object]:
    app.state.object_store.ensure_bucket()
    return {"minio_connected": True, "bucket": app.state.object_store.bucket}


@app.get("/debug/redis-check")
def redis_check() -> dict[str, object]:
    app.state.short_term_memory.ping()
    return {
        "redis_connected": True,
        "chat_short_term_ttl_seconds": app.state.short_term_memory.ttl_seconds,
    }


app.include_router(auth_router)
app.include_router(model_tools_router)
app.include_router(rag_router)
app.include_router(widget_admin_router)
app.include_router(widget_public_router)
app.include_router(chat_router)