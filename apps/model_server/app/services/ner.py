"""Simple code-shaped entity extraction for GitHub issues.

This is integration-level NER. It extracts useful maintainer entities such as
file paths, functions, versions, URLs, and error-like constants.
"""

import re

from app.schemas.nlp import Entity, IssueTextRequest, NERResponse


CODE_ENTITY_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        "FILE_PATH",
        r"\b[\w\-./]+?\.(py|js|ts|tsx|jsx|md|json|yaml|yml|toml|txt|java|go|rs|cpp|c|h)\b",
    ),
    (
        "FUNCTION",
        r"\b[a-zA-Z_][a-zA-Z0-9_]*\(\)",
    ),
    (
        "ERROR_CODE",
        r"\b[A-Z][A-Z0-9_]*ERROR[A-Z0-9_]*\b",
    ),
    (
        "VERSION",
        r"\bv?\d+\.\d+(\.\d+)?\b",
    ),
    (
        "URL",
        r"https?://[^\s)>\]]+",
    ),
    (
        "PACKAGE",
        r"\b(@[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+|[a-zA-Z0-9_.-]+==\d+\.\d+(\.\d+)?)\b",
    ),
)


def extract_entities(request: IssueTextRequest) -> NERResponse:
    """Extract code-shaped entities from issue title and body."""

    text = f"{request.title}\n{request.body}"

    entities: list[Entity] = []
    seen: set[tuple[str, str]] = set()

    for entity_type, pattern in CODE_ENTITY_PATTERNS:
        for match in re.finditer(pattern, text):
            entity_text = match.group(0)
            key = (entity_text, entity_type)

            if key in seen:
                continue

            seen.add(key)
            entities.append(
                Entity(
                    text=entity_text,
                    type=entity_type,
                )
            )

    return NERResponse(entities=entities)