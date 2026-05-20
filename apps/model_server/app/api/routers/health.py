"""Health routes for the model server."""

from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health")
def health(request: Request) -> dict[str, object]:
    model_loaded = hasattr(request.app.state, "model_bundle")

    return {
        "status": "ok" if model_loaded else "degraded",
        "service": "model_server",
        "model_loaded": model_loaded,
    }