from __future__ import annotations

CHAT_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "classify_issue",
            "description": (
                "Classify an issue into one of the project labels such as bug, "
                "feature, docs, or question. Use this when the user asks for triage "
                "or issue type."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Short issue title.",
                    },
                    "body": {
                        "type": "string",
                        "description": "Full issue body or user-provided issue text.",
                    },
                },
                "required": ["title", "body"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_entities",
            "description": (
                "Extract code-shaped entities from issue text, such as file paths, "
                "function names, error codes, versions, packages, and stack-trace terms."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Short issue title.",
                    },
                    "body": {
                        "type": "string",
                        "description": "Full issue body or user-provided issue text.",
                    },
                },
                "required": ["title", "body"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_thread",
            "description": (
                "Summarize a long issue, thread, or maintainer conversation. "
                "Use only when the user asks for a summary or the text is long."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Issue title or summary title.",
                    },
                    "body": {
                        "type": "string",
                        "description": "Issue body, thread, or long text to summarize.",
                    },
                },
                "required": ["title", "body"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": (
                "Search the project's docs and resolved issues for relevant evidence. "
                "Use this when the user asks for related issues, fixes, explanations, "
                "maintainer answers, or grounded project knowledge."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The search question.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to retrieve. Default 5.",
                        "minimum": 1,
                        "maximum": 10,
                    },
                    "label_filter": {
                        "type": ["string", "null"],
                        "description": "Optional label filter such as bug, docs, feature, or question.",
                    },
                },
                "required": ["question"],
                "additionalProperties": False,
            },
        },
    },
]