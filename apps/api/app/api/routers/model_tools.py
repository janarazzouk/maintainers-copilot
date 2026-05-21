"""API routes that expose model-server tools to the API/chatbot layer."""

from fastapi import APIRouter, HTTPException, Request, status

from app.infra.model_server_client import ModelServerClient, ModelServerError
from app.schemas.model_tools import (
    ClassificationResponse,
    IssueTextRequest,
    NERResponse,
    SummarizeRequest,
    SummarizeResponse,
)

router = APIRouter(prefix="/tools", tags=["model-tools"])


def _get_model_client(request: Request) -> ModelServerClient:
    client = getattr(request.app.state, "model_server_client", None)

    if client is None:
        raise RuntimeError("Model server client is not initialized.")

    return client


@router.post("/classify", response_model=ClassificationResponse)
async def classify_issue(
    payload: IssueTextRequest,
    request: Request,
) -> ClassificationResponse:
    client = _get_model_client(request)

    try:
        result = await client.classify(
            title=payload.title,
            body=payload.body,
        )
    except ModelServerError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "MODEL_SERVER_UNAVAILABLE",
                "message": str(exc),
            },
        ) from exc

    return ClassificationResponse(**result)


@router.post("/ner", response_model=NERResponse)
async def extract_entities(
    payload: IssueTextRequest,
    request: Request,
) -> NERResponse:
    client = _get_model_client(request)

    try:
        result = await client.extract_entities(
            title=payload.title,
            body=payload.body,
        )
    except ModelServerError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "MODEL_SERVER_UNAVAILABLE",
                "message": str(exc),
            },
        ) from exc

    return NERResponse(**result)


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize_issue(
    payload: SummarizeRequest,
    request: Request,
) -> SummarizeResponse:
    client = _get_model_client(request)

    try:
        result = await client.summarize(
            title=payload.title,
            body=payload.body,
        )
    except ModelServerError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "MODEL_SERVER_UNAVAILABLE",
                "message": str(exc),
            },
        ) from exc

    return SummarizeResponse(**result)