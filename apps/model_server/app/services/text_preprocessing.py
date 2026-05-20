"""Text preprocessing used before RoBERTa inference.

This must stay aligned with the preprocessing used during training.
"""

import re


TEMPLATE_LINES: tuple[str, ...] = (
    "Version",
    "Platform",
    "Subsystem",
    "What steps will reproduce the bug?",
    "How often does it reproduce? Is there a required condition?",
    "What is the expected behavior? Why is that the expected behavior?",
    "What do you see instead?",
    "Additional information",
)


def clean_issue_text(text: str) -> str:
    """Clean issue text the same way the training notebook did."""

    cleaned = str(text)

    for line in TEMPLATE_LINES:
        cleaned = cleaned.replace(line, " ")

    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)

    return cleaned.strip()


def build_issue_text(title: str, body: str) -> str:
    """Combine title and body exactly like training."""

    return clean_issue_text(f"{title}\n\n{body}")