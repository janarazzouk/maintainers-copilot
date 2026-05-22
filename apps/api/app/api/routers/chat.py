from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.infra.database import get_db
from app.infra.embeddings import EmbeddingModel
from app.infra.groq_client import GroqLLMClient
from app.infra.groq_tool_client import GroqToolCallingClient
from app.infra.minio import MinIOObjectStore
from app.infra.model_server_client import ModelServerClient
from app.infra.redis import RedisShortTermMemory
from app.models.user import User
from app.schemas.chat import (
    ChatMessageResponse,
    ChatRequest,
    ChatResponse,
    ConversationResponse,
)
from app.services.chat_service import ChatError, ChatService
from app.services.llm_chat_service import LLMChatService
from app.services.rag_snapshot_service import RagSnapshotService
from app.services.tool_service import ToolService

router = APIRouter(prefix="/chat", tags=["chat"])


def get_short_term_memory(request: Request) -> RedisShortTermMemory:
    return request.app.state.short_term_memory


def get_model_client(request: Request) -> ModelServerClient:
    return request.app.state.model_server_client


def get_embedding_model(request: Request) -> EmbeddingModel:
    return request.app.state.embedding_model


def get_llm_client(request: Request) -> GroqLLMClient | None:
    return request.app.state.groq_llm_client


def get_groq_tool_client(request: Request) -> GroqToolCallingClient | None:
    return request.app.state.groq_tool_client


def get_object_store(request: Request) -> MinIOObjectStore:
    return request.app.state.object_store


@router.post("/message", response_model=ChatResponse)
async def send_chat_message(
    payload: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    short_term_memory: RedisShortTermMemory = Depends(get_short_term_memory),
    model_client: ModelServerClient = Depends(get_model_client),
    embedding_model: EmbeddingModel = Depends(get_embedding_model),
    llm_client: GroqLLMClient | None = Depends(get_llm_client),
    groq_tool_client: GroqToolCallingClient | None = Depends(get_groq_tool_client),
    object_store: MinIOObjectStore = Depends(get_object_store),
) -> ChatResponse:
    tool_service = ToolService(
        db=db,
        model_client=model_client,
        embedding_model=embedding_model,
        llm_client=llm_client,
    )
    rag_snapshot_service = RagSnapshotService(object_store=object_store)

    llm_chat_service = None
    if groq_tool_client is not None:
        llm_chat_service = LLMChatService(
            groq_client=groq_tool_client,
            tool_service=tool_service,
        )

    service = ChatService(
        db=db,
        short_term_memory=short_term_memory,
        tool_service=tool_service,
        rag_snapshot_service=rag_snapshot_service,
        llm_chat_service=llm_chat_service,
    )

    try:
        conversation_id, user_message, assistant_message, answer, recent_messages = await service.handle_message(
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