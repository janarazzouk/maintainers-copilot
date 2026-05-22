from __future__ import annotations

CHAT_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "classify_issue",
            "description": "Classify an issue into bug, feature, docs, or question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "body": {"type": "string"},
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
            "description": "Extract code-shaped entities such as file paths, functions, errors, and versions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "body": {"type": "string"},
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
            "description": "Summarize a long issue, thread, or maintainer conversation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "body": {"type": "string"},
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
            "description": "Search project docs and resolved issues for relevant evidence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "default": 5,
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
    {
        "type": "function",
        "function": {
            "name": "write_memory",
            "description": (
                "Write durable long-term memory for the authenticated user. "
                "Only call this tool when the user explicitly asks to remember, store, save, or note something "
                "for future conversations, or when the user states a durable preference using wording like "
                "'from now on' or 'going forward'. Never store secrets, API keys, passwords, tokens, or temporary facts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_type": {
                        "type": "string",
                        "enum": ["semantic", "episodic", "procedural"],
                        "default": "semantic",
                    },
                    "content": {
                        "type": "string",
                        "description": "The durable memory to store.",
                    },
                    "reason": {
                        "type": ["string", "null"],
                        "description": "Why this memory is useful in future conversations.",
                    },
                },
                "required": ["memory_type", "content"],
                "additionalProperties": False,
            },
        },
    },
]