from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MemoryCreateRequest(BaseModel):
    memory_type: str = Field(default="semantic")
    content: str = Field(min_length=1, max_length=4000)
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemorySearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    limit: int = Field(default=5, ge=1, le=20)


class MemoryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    memory_type: str
    redacted_content: str
    memory_metadata: dict[str, Any]
    created_at: datetime


class MemorySearchResult(BaseModel):
    id: int
    memory_type: str
    content: str
    score: float
    metadata: dict[str, Any]
    created_at: str