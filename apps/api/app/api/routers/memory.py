from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.infra.database import get_db
from app.infra.embeddings import EmbeddingModel
from app.models.user import User
from app.schemas.memory import (
    MemoryCreateRequest,
    MemoryResponse,
    MemorySearchRequest,
    MemorySearchResult,
)
from app.services.memory_service import LongTermMemoryService, MemoryError

router = APIRouter(prefix="/memory", tags=["memory"])


def get_embedding_model(request: Request) -> EmbeddingModel:
    return request.app.state.embedding_model


@router.get("", response_model=list[MemoryResponse])
def list_memory(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    embedding_model: EmbeddingModel = Depends(get_embedding_model),
) -> list[MemoryResponse]:
    service = LongTermMemoryService(db=db, embedding_model=embedding_model)
    memories = service.list_memories(user=current_user)
    return [MemoryResponse.model_validate(memory) for memory in memories]


@router.post("", response_model=MemoryResponse, status_code=status.HTTP_201_CREATED)
def create_memory(
    payload: MemoryCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    embedding_model: EmbeddingModel = Depends(get_embedding_model),
) -> MemoryResponse:
    service = LongTermMemoryService(db=db, embedding_model=embedding_model)

    try:
        memory = service.write_memory(
            user=current_user,
            memory_type=payload.memory_type,
            content=payload.content,
            reason=payload.reason,
            metadata=payload.metadata,
        )
        db.commit()
        db.refresh(memory)
    except MemoryError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "MEMORY_WRITE_FAILED", "message": str(exc)},
        ) from exc

    return MemoryResponse.model_validate(memory)


@router.post("/search", response_model=list[MemorySearchResult])
def search_memory(
    payload: MemorySearchRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    embedding_model: EmbeddingModel = Depends(get_embedding_model),
) -> list[MemorySearchResult]:
    service = LongTermMemoryService(db=db, embedding_model=embedding_model)
    results = service.search_memories(
        user=current_user,
        query=payload.query,
        limit=payload.limit,
    )
    return [MemorySearchResult(**item) for item in results]


@router.delete("/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_memory(
    memory_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    embedding_model: EmbeddingModel = Depends(get_embedding_model),
) -> None:
    service = LongTermMemoryService(db=db, embedding_model=embedding_model)
    deleted = service.delete_memory(user=current_user, memory_id=memory_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "MEMORY_NOT_FOUND", "message": "Memory not found."},
        )

    db.commit()