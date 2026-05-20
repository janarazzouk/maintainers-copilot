"""Pydantic schemas for NLP model-server endpoints."""

from pydantic import BaseModel, Field


class IssueTextRequest(BaseModel):
    """Input used by classifier and NER endpoints."""

    title: str = Field(..., min_length=1)
    body: str = ""


class ClassificationResponse(BaseModel):
    """Classifier response returned by /classify."""

    label: str
    confidence: float
    probabilities: dict[str, float]


class Entity(BaseModel):
    """One extracted code-shaped entity."""

    text: str
    type: str


class NERResponse(BaseModel):
    """NER response returned by /ner."""

    entities: list[Entity]


class SummarizeRequest(BaseModel):
    """Input used by the summarizer endpoint."""

    title: str = Field(..., min_length=1)
    body: str = Field(..., min_length=1)


class SummarizeResponse(BaseModel):
    """Summarizer response returned by /summarize."""

    summary: str