from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    title: Mapped[str] = mapped_column(String(255), default="New conversation", nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    conversation_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("conversations.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    role: Mapped[str] = mapped_column(String(30), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    redacted_content: Mapped[str | None] = mapped_column(Text, nullable=True)

    message_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    retrieval_snapshot_key: Mapped[str | None] = mapped_column(String(500), nullable=True)

    request_id: Mapped[str | None] = mapped_column(String(120), nullable=True)
    trace_id: Mapped[str | None] = mapped_column(String(120), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class ToolCall(Base):
    __tablename__ = "tool_calls"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    conversation_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("conversations.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    message_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("chat_messages.id", ondelete="SET NULL"),
        nullable=True,
    )

    tool_name: Mapped[str] = mapped_column(String(120), nullable=False)
    input_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    output_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)

    status: Mapped[str] = mapped_column(String(40), default="success", nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    request_id: Mapped[str | None] = mapped_column(String(120), nullable=True)
    trace_id: Mapped[str | None] = mapped_column(String(120), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)