from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    conversation_id: int | None = None
    message: str = Field(min_length=1, max_length=12000)
    repo: str | None = None


class ChatMessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    conversation_id: int
    role: str
    content: str
    retrieval_snapshot_key: str | None = None
    request_id: str | None = None
    trace_id: str | None = None
    created_at: datetime


class ConversationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    title: str
    created_at: datetime
    updated_at: datetime


class ChatResponse(BaseModel):
    conversation_id: int
    user_message: ChatMessageResponse
    assistant_message: ChatMessageResponse
    answer: str
    recent_messages: list[dict[str, Any]]