from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.memory import LongTermMemory


class MemoryRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create_memory(
        self,
        *,
        user_id: int,
        memory_type: str,
        content: str,
        redacted_content: str,
        embedding: list[float],
        memory_metadata: dict[str, Any] | None = None,
    ) -> LongTermMemory:
        memory = LongTermMemory(
            user_id=user_id,
            memory_type=memory_type,
            content=content,
            redacted_content=redacted_content,
            embedding=embedding,
            memory_metadata=memory_metadata or {},
        )
        self.db.add(memory)
        self.db.flush()
        return memory

    def list_memories_for_user(
        self,
        *,
        user_id: int,
        limit: int = 50,
    ) -> list[LongTermMemory]:
        statement = (
            select(LongTermMemory)
            .where(
                LongTermMemory.user_id == user_id,
                LongTermMemory.deleted_at.is_(None),
            )
            .order_by(LongTermMemory.created_at.desc())
            .limit(limit)
        )
        return list(self.db.scalars(statement).all())

    def find_exact_memory_for_user(
        self,
        *,
        user_id: int,
        normalized_content: str,
    ) -> LongTermMemory | None:
        memories = self.list_memories_for_user(user_id=user_id, limit=200)

        for memory in memories:
            existing_normalized = self._normalize_for_dedup(memory.redacted_content)
            if existing_normalized == normalized_content:
                return memory

        return None

    def search_memories_for_user(
        self,
        *,
        user_id: int,
        query_embedding: list[float],
        limit: int = 5,
    ) -> list[tuple[LongTermMemory, float]]:
        distance = LongTermMemory.embedding.cosine_distance(query_embedding)

        statement = (
            select(LongTermMemory, distance.label("distance"))
            .where(
                LongTermMemory.user_id == user_id,
                LongTermMemory.deleted_at.is_(None),
            )
            .order_by(distance)
            .limit(limit)
        )

        rows = self.db.execute(statement).all()
        return [(row[0], float(row[1])) for row in rows]

    def mark_duplicate_attempt(
        self,
        *,
        memory: LongTermMemory,
        duplicate_content: str,
        duplicate_reason: str | None,
        duplicate_similarity: float | None,
    ) -> LongTermMemory:
        metadata = dict(memory.memory_metadata or {})

        duplicate_count = int(metadata.get("duplicate_attempt_count") or 0)
        duplicate_count += 1

        metadata["duplicate_attempt_count"] = duplicate_count
        metadata["last_duplicate_content"] = duplicate_content
        metadata["last_duplicate_reason"] = duplicate_reason
        metadata["last_duplicate_similarity"] = duplicate_similarity
        metadata["last_duplicate_at"] = datetime.utcnow().isoformat()

        memory.memory_metadata = metadata
        memory.updated_at = datetime.utcnow()

        self.db.flush()
        return memory

    def soft_delete_memory(
        self,
        *,
        memory_id: int,
        user_id: int,
    ) -> bool:
        memory = self.db.scalar(
            select(LongTermMemory).where(
                LongTermMemory.id == memory_id,
                LongTermMemory.user_id == user_id,
                LongTermMemory.deleted_at.is_(None),
            )
        )

        if memory is None:
            return False

        memory.deleted_at = datetime.utcnow()
        self.db.flush()
        return True

    def _normalize_for_dedup(self, text: str) -> str:
        return " ".join(text.lower().strip().split())