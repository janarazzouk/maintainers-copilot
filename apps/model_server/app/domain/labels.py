"""Shared issue-label definitions for the model server."""

ISSUE_LABELS: tuple[str, ...] = ("bug", "feature", "docs", "question")

ID_TO_LABEL: dict[int, str] = {
    0: "bug",
    1: "feature",
    2: "docs",
    3: "question",
}

LABEL_TO_ID: dict[str, int] = {
    label: label_id for label_id, label in ID_TO_LABEL.items()
}