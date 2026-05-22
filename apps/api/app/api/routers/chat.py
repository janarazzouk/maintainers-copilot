from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.infra.database import get_db
from app.infra.redis import RedisShortTermMemory
from app.models.user import User
from app.schemas.chat import (
    ChatMessageResponse,
    ChatRequest,
    ChatResponse,
    ConversationResponse,
)
from app.services.chat_service import ChatError, ChatService

router = APIRouter(prefix="/chat", tags=["chat"])


def get_short_term_memory(request: Request) -> RedisShortTermMemory:
    return request.app.state.short_term_memory


@router.post("/message", response_model=ChatResponse)
def send_chat_message(
    payload: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    short_term_memory: RedisShortTermMemory = Depends(get_short_term_memory),
) -> ChatResponse:
    service = ChatService(db=db, short_term_memory=short_term_memory)

    try:
        conversation_id, user_message, assistant_message, answer, recent_messages = service.handle_message(
            user=current_user,
            payload=payload,
        )
    except ChatError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "CHAT_ERROR", "message": str(exc)},
        ) from exc

    return ChatResponse(
        conversation_id=conversation_id,
        user_message=ChatMessageResponse.model_validate(user_message),
        assistant_message=ChatMessageResponse.model_validate(assistant_message),
        answer=answer,
        recent_messages=recent_messages,
    )


@router.get("/conversations", response_model=list[ConversationResponse])
def list_conversations(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    short_term_memory: RedisShortTermMemory = Depends(get_short_term_memory),
) -> list[ConversationResponse]:
    service = ChatService(db=db, short_term_memory=short_term_memory)
    conversations = service.list_conversations(user=current_user)
    return [ConversationResponse.model_validate(item) for item in conversations]


@router.get("/conversations/{conversation_id}/messages", response_model=list[ChatMessageResponse])
def list_messages(
    conversation_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    short_term_memory: RedisShortTermMemory = Depends(get_short_term_memory),
) -> list[ChatMessageResponse]:
    service = ChatService(db=db, short_term_memory=short_term_memory)

    try:
        messages = service.get_messages_for_conversation(
            user=current_user,
            conversation_id=conversation_id,
        )
    except ChatError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "CONVERSATION_NOT_FOUND", "message": str(exc)},
        ) from exc

    return [ChatMessageResponse.model_validate(item) for item in messages]


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(
    conversation_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    short_term_memory: RedisShortTermMemory = Depends(get_short_term_memory),
) -> None:
    service = ChatService(db=db, short_term_memory=short_term_memory)

    try:
        service.soft_delete_conversation(
            user=current_user,
            conversation_id=conversation_id,
        )
    except ChatError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "CONVERSATION_NOT_FOUND", "message": str(exc)},
        ) from exc