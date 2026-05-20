"""Basic local issue summarization service.

This does not use an LLM. It cleans the issue text and returns a shortened,
chatbot-friendly summary.
"""

from app.schemas.nlp import SummarizeRequest, SummarizeResponse
from app.services.text_preprocessing import clean_issue_text


def _compact_whitespace(text: str) -> str:
    """Turn multiline issue text into one readable sentence-like block."""

    return " ".join(text.split())


def summarize_issue(request: SummarizeRequest) -> SummarizeResponse:
    """Return a clean local summary without calling an LLM."""

    title = clean_issue_text(request.title)
    body = clean_issue_text(request.body)

    title = _compact_whitespace(title)
    body = _compact_whitespace(body)

    max_body_chars = 650

    if len(body) > max_body_chars:
        body = body[:max_body_chars].rsplit(" ", 1)[0] + "..."

    if body:
        summary = f"{title}: {body}"
    else:
        summary = title

    return SummarizeResponse(summary=summary)