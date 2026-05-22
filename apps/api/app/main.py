from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app.infra.config import get_settings
from app.infra.database import check_database_connection, init_database
from app.infra.vault import VaultClient

from app.api.routers.model_tools import router as model_tools_router
from app.infra.model_server_client import ModelServerClient

from app.api.routers.rag import router as rag_router
from app.infra.embeddings import EmbeddingModel
from app.infra.groq_client import GroqLLMClient

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.embedding_model = EmbeddingModel(
        model_name=settings.embedding_model_name,
    )
    app.state.model_server_client = ModelServerClient(
    base_url=settings.model_server_url,
)

    vault = VaultClient(
        addr=settings.vault_addr,
        token=settings.vault_dev_root_token_id,
    )

    app_secrets = vault.read_app_secrets()
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

    init_database(app_secrets["database_url"])
    check_database_connection()

    app.state.secrets = app_secrets

    

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

app.include_router(model_tools_router)
app.include_router(rag_router)