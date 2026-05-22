"""add chat tables

Revision ID: b8e2d7a91c40
Revises: a3b57e4d0c21
Create Date: 2026-05-22
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "b8e2d7a91c40"
down_revision: Union[str, Sequence[str], None] = "a3b57e4d0c21"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "conversations",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.Column("deleted_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_conversations_id"), "conversations", ["id"], unique=False)
    op.create_index(op.f("ix_conversations_user_id"), "conversations", ["user_id"], unique=False)

    op.create_table(
        "chat_messages",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("conversation_id", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(length=30), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("redacted_content", sa.Text(), nullable=True),
        sa.Column(
            "message_metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("retrieval_snapshot_key", sa.String(length=500), nullable=True),
        sa.Column("request_id", sa.String(length=120), nullable=True),
        sa.Column("trace_id", sa.String(length=120), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_chat_messages_id"), "chat_messages", ["id"], unique=False)
    op.create_index(
        op.f("ix_chat_messages_conversation_id"),
        "chat_messages",
        ["conversation_id"],
        unique=False,
    )

    op.create_table(
        "tool_calls",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("conversation_id", sa.Integer(), nullable=False),
        sa.Column("message_id", sa.Integer(), nullable=True),
        sa.Column("tool_name", sa.String(length=120), nullable=False),
        sa.Column(
            "input_json",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "output_json",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("status", sa.String(length=40), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("request_id", sa.String(length=120), nullable=True),
        sa.Column("trace_id", sa.String(length=120), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["message_id"], ["chat_messages.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_tool_calls_id"), "tool_calls", ["id"], unique=False)
    op.create_index(op.f("ix_tool_calls_conversation_id"), "tool_calls", ["conversation_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_tool_calls_conversation_id"), table_name="tool_calls")
    op.drop_index(op.f("ix_tool_calls_id"), table_name="tool_calls")
    op.drop_table("tool_calls")

    op.drop_index(op.f("ix_chat_messages_conversation_id"), table_name="chat_messages")
    op.drop_index(op.f("ix_chat_messages_id"), table_name="chat_messages")
    op.drop_table("chat_messages")

    op.drop_index(op.f("ix_conversations_user_id"), table_name="conversations")
    op.drop_index(op.f("ix_conversations_id"), table_name="conversations")
    op.drop_table("conversations")