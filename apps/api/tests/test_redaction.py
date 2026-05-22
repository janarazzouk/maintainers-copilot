from app.infra.redaction import redact_text, redact_value


def test_redacts_openai_style_key() -> None:
    text = "User pasted key sk-abcdefghijklmnopqrstuvwxyz1234567890"

    redacted = redact_text(text)

    assert "sk-abcdefghijklmnopqrstuvwxyz1234567890" not in redacted
    assert "[REDACTED_OPENAI_KEY]" in redacted


def test_redacts_groq_key() -> None:
    text = "GROQ_API_KEY=gsk_abcdefghijklmnopqrstuvwxyz1234567890"

    redacted = redact_text(text)

    assert "gsk_abcdefghijklmnopqrstuvwxyz1234567890" not in redacted
    assert "[REDACTED_GROQ_KEY]" in redacted or "GROQ_API_KEY=[REDACTED]" in redacted


def test_redacts_github_tokens() -> None:
    text = "Token ghp_abcdefghijklmnopqrstuvwxyz123456"

    redacted = redact_text(text)

    assert "ghp_abcdefghijklmnopqrstuvwxyz123456" not in redacted
    assert "[REDACTED_GITHUB_TOKEN]" in redacted


def test_redacts_bearer_token() -> None:
    text = "Authorization: Bearer abc.def.ghi"

    redacted = redact_text(text)

    assert "abc.def.ghi" not in redacted
    assert "Bearer [REDACTED]" in redacted


def test_redacts_database_url() -> None:
    text = "DATABASE_URL=postgresql+psycopg2://postgres:postgres@db:5432/maintainers_copilot"

    redacted = redact_text(text)

    assert "postgres:postgres@db" not in redacted
    assert "[REDACTED_DATABASE_URL]" in redacted


def test_redacts_nested_tool_payload() -> None:
    payload = {
        "message": "Here is my token ghp_abcdefghijklmnopqrstuvwxyz123456",
        "headers": {
            "Authorization": "Bearer abc.def.ghi",
        },
        "items": [
            "password=supersecret",
            "safe text",
        ],
    }

    redacted = redact_value(payload)

    assert "ghp_abcdefghijklmnopqrstuvwxyz123456" not in str(redacted)
    assert "abc.def.ghi" not in str(redacted)
    assert "supersecret" not in str(redacted)
    assert "[REDACTED_GITHUB_TOKEN]" in str(redacted)
    assert "Bearer [REDACTED]" in str(redacted)
    assert "password=[REDACTED]" in str(redacted)