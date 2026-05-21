from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base


class RagDocument(Base):
    """Parent RAG document.

    One row = one resolved GitHub issue from rag_issues_corpus.jsonl.
    The retriever searches chunks, then uses doc_id to fetch this parent document.
    """

    __tablename__ = "rag_documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    doc_id: Mapped[str] = mapped_column(
        String(120),
        unique=True,
        index=True,
        nullable=False,
    )

    source_type: Mapped[str] = mapped_column(String(80), nullable=False)
    repo: Mapped[str] = mapped_column(String(255), nullable=False)

    issue_id: Mapped[int | None] = mapped_column(Integer, index=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False)

    final_label: Mapped[str | None] = mapped_column(String(80), index=True)
    state: Mapped[str | None] = mapped_column(String(80), index=True)

    issue_created_at: Mapped[datetime | None] = mapped_column(DateTime)
    issue_closed_at: Mapped[datetime | None] = mapped_column(DateTime)

    problem_summary: Mapped[str | None] = mapped_column(Text)
    maintainer_answer: Mapped[str | None] = mapped_column(Text)
    resolution_type: Mapped[str | None] = mapped_column(String(120), index=True)

    text: Mapped[str] = mapped_column(Text, nullable=False)

    raw_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
    )

    chunks: Mapped[list["RagChunk"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class RagChunk(Base):
    """Searchable RAG chunk.

    One row = one chunk from rag_issues_chunks.jsonl.
    This is what we embed and retrieve.
    """

    __tablename__ = "rag_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    chunk_id: Mapped[str] = mapped_column(
        String(160),
        unique=True,
        index=True,
        nullable=False,
    )

    doc_id: Mapped[str] = mapped_column(
        String(120),
        ForeignKey("rag_documents.doc_id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    source_type: Mapped[str] = mapped_column(String(80), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False)

    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)

    final_label: Mapped[str | None] = mapped_column(String(80), index=True)
    issue_id: Mapped[int | None] = mapped_column(Integer, index=True)

    raw_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
    )

    # BAAI/bge-small-en-v1.5 and all-MiniLM-L6-v2 both use 384 dimensions.
    embedding: Mapped[list[float] | None] = mapped_column(Vector(384))

    document: Mapped[RagDocument] = relationship(back_populates="chunks")