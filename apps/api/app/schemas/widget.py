from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class WidgetCreate(BaseModel):
    name: str
    widget_id: str | None = None
    allowed_origins: list[str] = Field(default_factory=list)
    theme: dict[str, Any] = Field(default_factory=lambda: {"primaryColor": "#2563eb", "position": "bottom-right"})
    greeting: str = "How can I help you triage this issue?"
    enabled_tools: list[str] = Field(default_factory=lambda: ["rag_search", "classify_issue", "extract_entities", "summarize_thread"])
    is_active: bool = True


class WidgetUpdate(BaseModel):
    name: str | None = None
    allowed_origins: list[str] | None = None
    theme: dict[str, Any] | None = None
    greeting: str | None = None
    enabled_tools: list[str] | None = None
    is_active: bool | None = None


class WidgetResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    widget_id: str
    name: str
    allowed_origins: list[str]
    theme: dict[str, Any]
    greeting: str
    enabled_tools: list[str]
    is_active: bool
    created_at: datetime
    updated_at: datetime


class PublicWidgetConfig(BaseModel):
    widget_id: str
    theme: dict[str, Any]
    greeting: str
    enabled_tools: list[str]