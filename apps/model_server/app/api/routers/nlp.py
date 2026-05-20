"""NLP routes for classifier, NER, and summarizer."""

from fastapi import APIRouter, Request

from app.infra.model_loader import ModelBundle
from app.schemas.nlp import (
    ClassificationResponse,
    IssueTextRequest,
    NERResponse,
    SummarizeRequest,
    SummarizeResponse,
)
from app.services.classifier import classify_issue
from app.services.ner import extract_entities
from app.services.summarizer import summarize_issue

router = APIRouter(tags=["nlp"])


def _get_model_bundle(request: Request) -> ModelBundle:
    model_bundle = getattr(request.app.state, "model_bundle", None)

    if model_bundle is None:
        raise RuntimeError("RoBERTa classifier is not loaded.")

    return model_bundle


@router.post("/classify", response_model=ClassificationResponse)
def classify(
    request_body: IssueTextRequest,
    request: Request,
) -> ClassificationResponse:
    model_bundle = _get_model_bundle(request)
    return classify_issue(request_body, model_bundle)


@router.post("/ner", response_model=NERResponse)
def ner(request_body: IssueTextRequest) -> NERResponse:
    return extract_entities(request_body)


@router.post("/summarize", response_model=SummarizeResponse)
def summarize(request_body: SummarizeRequest) -> SummarizeResponse:
    return summarize_issue(request_body)