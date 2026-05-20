from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app.infra.config import get_settings
from app.infra.vault import VaultClient
from app.infra.database import check_database_connection, init_database
from app.infra.database import check_database_connection

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()

    vault = VaultClient(
        addr=settings.vault_addr,
        token=settings.vault_dev_root_token_id,
    )

    app_secrets = vault.read_app_secrets()
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
def db_check() -> dict[str, object]:
    check_database_connection()
    return {"database_connected": True}