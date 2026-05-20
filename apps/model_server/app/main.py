"""FastAPI entrypoint for the Maintainer's Copilot model server."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routers.health import router as health_router
from app.api.routers.nlp import router as nlp_router
from app.infra.config import get_settings
from app.infra.model_loader import load_roberta_classifier


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()

    app.state.settings = settings
    app.state.model_bundle = load_roberta_classifier(settings)

    yield


app = FastAPI(
    title="Maintainer's Copilot Model Server",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(nlp_router)