"""Schemas for API endpoints that wrap the model server."""

from pydantic import BaseModel, Field


class IssueTextRequest(BaseModel):
    title: str = Field(..., min_length=1)
    body: str = ""


class ClassificationResponse(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]


class Entity(BaseModel):
    text: str
    type: str


class NERResponse(BaseModel):
    entities: list[Entity]


class SummarizeRequest(BaseModel):
    title: str = Field(..., min_length=1)
    body: str = Field(..., min_length=1)


class SummarizeResponse(BaseModel):
    summary: str