from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.infra.database import get_db
from app.infra.embeddings import EmbeddingModel
from app.schemas.rag import RagQueryRequest, RagQueryResponse
from app.services.rag_service import RagService

router = APIRouter(prefix="/rag", tags=["rag"])


def _get_embedding_model(request: Request) -> EmbeddingModel:
    embedding_model = getattr(request.app.state, "embedding_model", None)

    if embedding_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "EMBEDDING_MODEL_UNAVAILABLE",
                "message": "Embedding model is not initialized.",
            },
        )

    return embedding_model


@router.post("/query", response_model=RagQueryResponse)
def query_rag(
    payload: RagQueryRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> RagQueryResponse:
    embedding_model = _get_embedding_model(request)

    service = RagService(
        db=db,
        embedding_model=embedding_model,
    )

    return service.query(payload)