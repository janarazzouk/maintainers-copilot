import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.repositories.rag_repository import RagRepository


@dataclass(frozen=True)
class RagIngestionResult:
    documents_loaded: int
    chunks_loaded: int
    total_documents: int
    total_chunks: int


class RagIngestionService:
    """Loads prepared RAG JSONL files into Postgres.

    This does not chunk again.
    Your corpus and chunks are already prepared.
    """

    def __init__(self, db: Session) -> None:
        self.db = db
        self.repository = RagRepository(db)

    def ingest(self, corpus_path: Path, chunks_path: Path) -> RagIngestionResult:
        documents_loaded = self._ingest_documents(corpus_path)
        chunks_loaded = self._ingest_chunks(chunks_path)

        self.db.commit()

        return RagIngestionResult(
            documents_loaded=documents_loaded,
            chunks_loaded=chunks_loaded,
            total_documents=self.repository.count_documents(),
            total_chunks=self.repository.count_chunks(),
        )

    def _ingest_documents(self, corpus_path: Path) -> int:
        count = 0

        for record in self._read_jsonl(corpus_path):
            data = self._map_document(record)
            self.repository.upsert_document(data)
            count += 1

        # Flush here so chunks can safely reference these doc_ids.
        self.db.flush()
        return count

    def _ingest_chunks(self, chunks_path: Path) -> int:
        count = 0

        for record in self._read_jsonl(chunks_path):
            data = self._map_chunk(record)

            if not self.repository.document_exists(data["doc_id"]):
                raise ValueError(
                    f"Chunk {data['chunk_id']} references missing doc_id "
                    f"{data['doc_id']}"
                )

            self.repository.upsert_chunk(data)
            count += 1

        return count

    def _map_document(self, record: dict[str, Any]) -> dict[str, Any]:
        metadata = record.get("metadata") or {}
        resolution = record.get("resolution") or {}

        return {
            "doc_id": record["doc_id"],
            "source_type": record.get("source_type")
            or metadata.get("source_type")
            or "resolved_issue",
            "repo": record.get("repo") or metadata.get("repo") or "nodejs/node",
            "issue_id": record.get("issue_id") or metadata.get("issue_id"),
            "title": record.get("title") or "",
            "url": record.get("url") or "",
            "final_label": record.get("final_label") or metadata.get("final_label"),
            "state": record.get("state"),
            "issue_created_at": self._parse_datetime(record.get("created_at")),
            "issue_closed_at": self._parse_datetime(record.get("closed_at")),
            "problem_summary": record.get("problem_summary"),
            "maintainer_answer": record.get("maintainer_answer"),
            "resolution_type": (
                metadata.get("resolution_type")
                or resolution.get("resolution_type")
            ),
            "text": record.get("text") or "",
            "raw_metadata": {
                **metadata,
                "labels": record.get("labels") or metadata.get("labels") or [],
                "expected_behavior": record.get("expected_behavior"),
                "observed_behavior": record.get("observed_behavior"),
                "resolution": resolution,
                "has_maintainer_answer": record.get("has_maintainer_answer"),
                "has_linked_fix": record.get("has_linked_fix"),
            },
        }

    def _map_chunk(self, record: dict[str, Any]) -> dict[str, Any]:
        metadata = record.get("metadata") or {}

        return {
            "chunk_id": record["chunk_id"],
            "doc_id": record["doc_id"],
            "source_type": record.get("source_type")
            or metadata.get("source_type")
            or "resolved_issue",
            "title": record.get("title") or metadata.get("title") or "",
            "url": record.get("url") or metadata.get("url") or "",
            "chunk_index": int(record.get("chunk_index") or metadata.get("chunk_index") or 0),
            "chunk_text": record.get("chunk_text") or "",
            "final_label": metadata.get("final_label"),
            "issue_id": metadata.get("issue_id"),
            "raw_metadata": metadata,
        }

    def _read_jsonl(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"RAG file not found: {path}")

        records: list[dict[str, Any]] = []

        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {path} at line {line_number}"
                    ) from exc

        return records

    def _parse_datetime(self, value: str | None) -> datetime | None:
        if not value:
            return None

        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))

        # DB column is normal DateTime, so store UTC as naive datetime.
        return parsed.replace(tzinfo=None)