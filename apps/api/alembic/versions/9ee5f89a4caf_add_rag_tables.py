"""add rag tables

Revision ID: 9ee5f89a4caf
Revises: 1f79eb0b131f
Create Date: 2026-05-21 20:00:47.748627
"""

from typing import Sequence, Union

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql


revision: str = '9ee5f89a4caf'
down_revision: Union[str, Sequence[str], None] = '1f79eb0b131f'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "rag_documents",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("doc_id", sa.String(length=120), nullable=False),
        sa.Column("source_type", sa.String(length=80), nullable=False),
        sa.Column("repo", sa.String(length=255), nullable=False),
        sa.Column("issue_id", sa.Integer(), nullable=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("final_label", sa.String(length=80), nullable=True),
        sa.Column("state", sa.String(length=80), nullable=True),
        sa.Column("issue_created_at", sa.DateTime(), nullable=True),
        sa.Column("issue_closed_at", sa.DateTime(), nullable=True),
        sa.Column("problem_summary", sa.Text(), nullable=True),
        sa.Column("maintainer_answer", sa.Text(), nullable=True),
        sa.Column("resolution_type", sa.String(length=120), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column(
            "raw_metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("doc_id", name="uq_rag_documents_doc_id"),
    )

    op.create_index(
        op.f("ix_rag_documents_id"),
        "rag_documents",
        ["id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_rag_documents_doc_id"),
        "rag_documents",
        ["doc_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_rag_documents_issue_id"),
        "rag_documents",
        ["issue_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_rag_documents_final_label"),
        "rag_documents",
        ["final_label"],
        unique=False,
    )
    op.create_index(
        op.f("ix_rag_documents_state"),
        "rag_documents",
        ["state"],
        unique=False,
    )
    op.create_index(
        op.f("ix_rag_documents_resolution_type"),
        "rag_documents",
        ["resolution_type"],
        unique=False,
    )

    op.create_table(
        "rag_chunks",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("chunk_id", sa.String(length=160), nullable=False),
        sa.Column("doc_id", sa.String(length=120), nullable=False),
        sa.Column("source_type", sa.String(length=80), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("chunk_text", sa.Text(), nullable=False),
        sa.Column("final_label", sa.String(length=80), nullable=True),
        sa.Column("issue_id", sa.Integer(), nullable=True),
        sa.Column(
            "raw_metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("embedding", Vector(384), nullable=True),
        sa.ForeignKeyConstraint(
            ["doc_id"],
            ["rag_documents.doc_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("chunk_id", name="uq_rag_chunks_chunk_id"),
    )

    op.create_index(
        op.f("ix_rag_chunks_id"),
        "rag_chunks",
        ["id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_rag_chunks_chunk_id"),
        "rag_chunks",
        ["chunk_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_rag_chunks_doc_id"),
        "rag_chunks",
        ["doc_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_rag_chunks_final_label"),
        "rag_chunks",
        ["final_label"],
        unique=False,
    )
    op.create_index(
        op.f("ix_rag_chunks_issue_id"),
        "rag_chunks",
        ["issue_id"],
        unique=False,
    )

    # Useful for sparse keyword search later.
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_rag_chunks_text_fts
        ON rag_chunks
        USING gin (to_tsvector('english', chunk_text));
        """
    )

    # Useful for dense vector search later.
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_rag_chunks_embedding_hnsw
        ON rag_chunks
        USING hnsw (embedding vector_cosine_ops);
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_rag_chunks_embedding_hnsw")
    op.execute("DROP INDEX IF EXISTS ix_rag_chunks_text_fts")

    op.drop_index(op.f("ix_rag_chunks_issue_id"), table_name="rag_chunks")
    op.drop_index(op.f("ix_rag_chunks_final_label"), table_name="rag_chunks")
    op.drop_index(op.f("ix_rag_chunks_doc_id"), table_name="rag_chunks")
    op.drop_index(op.f("ix_rag_chunks_chunk_id"), table_name="rag_chunks")
    op.drop_index(op.f("ix_rag_chunks_id"), table_name="rag_chunks")
    op.drop_table("rag_chunks")

    op.drop_index(op.f("ix_rag_documents_resolution_type"), table_name="rag_documents")
    op.drop_index(op.f("ix_rag_documents_state"), table_name="rag_documents")
    op.drop_index(op.f("ix_rag_documents_final_label"), table_name="rag_documents")
    op.drop_index(op.f("ix_rag_documents_issue_id"), table_name="rag_documents")
    op.drop_index(op.f("ix_rag_documents_doc_id"), table_name="rag_documents")
    op.drop_index(op.f("ix_rag_documents_id"), table_name="rag_documents")
    op.drop_table("rag_documents")