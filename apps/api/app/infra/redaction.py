from __future__ import annotations

import re
from typing import Any


_SECRET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # OpenAI-style keys
    (re.compile(r"sk-[A-Za-z0-9_\-]{20,}"), "[REDACTED_OPENAI_KEY]"),

    # Groq-style keys
    (re.compile(r"gsk_[A-Za-z0-9_\-]{20,}"), "[REDACTED_GROQ_KEY]"),

    # GitHub tokens
    (re.compile(r"github_pat_[A-Za-z0-9_]{20,}"), "[REDACTED_GITHUB_TOKEN]"),
    (re.compile(r"ghp_[A-Za-z0-9]{20,}"), "[REDACTED_GITHUB_TOKEN]"),

    # Authorization headers
    (re.compile(r"Bearer\s+[A-Za-z0-9_\-\.]+"), "Bearer [REDACTED]"),

    # Database URLs
    (re.compile(r"postgresql(\+\w+)?://[^\s]+"), "[REDACTED_DATABASE_URL]"),

    # Generic key-value secrets
    (
        re.compile(r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*[^\s,;]+"),
        r"\1=[REDACTED]",
    ),
]


def redact_text(text: str) -> str:
    redacted = text

    for pattern, replacement in _SECRET_PATTERNS:
        redacted = pattern.sub(replacement, redacted)

    return redacted


def redact_value(value: Any) -> Any:
    """Recursively redact strings inside dictionaries/lists.

    Use this before sending tool inputs/outputs to logs, traces, or memory.
    """

    if isinstance(value, str):
        return redact_text(value)

    if isinstance(value, dict):
        return {
            str(key): redact_value(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [redact_value(item) for item in value]

    if isinstance(value, tuple):
        return tuple(redact_value(item) for item in value)

    return value