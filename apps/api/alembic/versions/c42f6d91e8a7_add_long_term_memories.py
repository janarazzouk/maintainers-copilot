"""add long term memories

Revision ID: c42f6d91e8a7
Revises: b8e2d7a91c40
Create Date: 2026-05-22
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql


revision: str = "c42f6d91e8a7"
down_revision: Union[str, Sequence[str], None] = "b8e2d7a91c40"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "long_term_memories",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("memory_type", sa.String(length=40), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("redacted_content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(384), nullable=False),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.Column("deleted_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index(
        op.f("ix_long_term_memories_id"),
        "long_term_memories",
        ["id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_long_term_memories_user_id"),
        "long_term_memories",
        ["user_id"],
        unique=False,
    )

    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_long_term_memories_embedding_hnsw
        ON long_term_memories
        USING hnsw (embedding vector_cosine_ops)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_long_term_memories_embedding_hnsw")
    op.drop_index(op.f("ix_long_term_memories_user_id"), table_name="long_term_memories")
    op.drop_index(op.f("ix_long_term_memories_id"), table_name="long_term_memories")
    op.drop_table("long_term_memories")