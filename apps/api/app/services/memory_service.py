from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.infra.embeddings import EmbeddingModel
from app.infra.redaction import redact_text
from app.models.memory import LongTermMemory
from app.models.user import User
from app.repositories.audit_repository import AuditRepository
from app.repositories.memory_repository import MemoryRepository


class MemoryError(RuntimeError):
    pass


class LongTermMemoryService:
    def __init__(
        self,
        *,
        db: Session,
        embedding_model: EmbeddingModel,
    ) -> None:
        self.db = db
        self.embedding_model = embedding_model
        self.memories = MemoryRepository(db)
        self.audit = AuditRepository(db)

    def write_memory(
        self,
        *,
        user: User,
        memory_type: str,
        content: str,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LongTermMemory:
        cleaned_content = content.strip()
        if not cleaned_content:
            raise MemoryError("Memory content is required.")

        if memory_type not in {"semantic", "episodic", "procedural"}:
            raise MemoryError("memory_type must be semantic, episodic, or procedural.")

        redacted_content = redact_text(cleaned_content)
        embedding = self._embed_text(redacted_content)

        merged_metadata = {
            "reason": reason,
            **(metadata or {}),
        }

        memory = self.memories.create_memory(
            user_id=user.id,
            memory_type=memory_type,
            content=cleaned_content,
            redacted_content=redacted_content,
            embedding=embedding,
            memory_metadata=merged_metadata,
        )

        self.audit.create(
            actor=user.email,
            action="memory.write",
            target=f"memory:{memory.id}",
        )

        return memory

    def list_memories(
        self,
        *,
        user: User,
        limit: int = 50,
    ) -> list[LongTermMemory]:
        return self.memories.list_memories_for_user(user_id=user.id, limit=limit)

    def search_memories(
        self,
        *,
        user: User,
        query: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        if not query.strip():
            return []

        query_embedding = self._embed_text(redact_text(query))
        rows = self.memories.search_memories_for_user(
            user_id=user.id,
            query_embedding=query_embedding,
            limit=limit,
        )

        results: list[dict[str, Any]] = []
        for memory, distance in rows:
            results.append(
                {
                    "id": memory.id,
                    "memory_type": memory.memory_type,
                    "content": memory.redacted_content,
                    "score": round(1.0 - distance, 4),
                    "metadata": memory.memory_metadata,
                    "created_at": memory.created_at.isoformat(),
                }
            )

        return results

    def delete_memory(
        self,
        *,
        user: User,
        memory_id: int,
    ) -> bool:
        deleted = self.memories.soft_delete_memory(
            memory_id=memory_id,
            user_id=user.id,
        )

        if deleted:
            self.audit.create(
                actor=user.email,
                action="memory.delete",
                target=f"memory:{memory_id}",
            )

        return deleted

    def _embed_text(self, text: str) -> list[float]:
        """Adapter around your existing EmbeddingModel.

        This is intentionally defensive because different wrappers expose
        embed(), embed_query(), encode(), etc.
        """

        method_names = [
            "embed_query",
            "embed_text",
            "embed",
            "encode",
        ]

        last_error: Exception | None = None

        for method_name in method_names:
            method = getattr(self.embedding_model, method_name, None)
            if method is None:
                continue

            try:
                value = method(text)
                return self._normalize_embedding(value)
            except TypeError as exc:
                last_error = exc
                try:
                    value = method([text])
                    return self._normalize_embedding(value)
                except Exception as inner_exc:
                    last_error = inner_exc
            except Exception as exc:
                last_error = exc

        raise MemoryError(f"Could not embed memory text: {last_error}")

    def _normalize_embedding(self, value: Any) -> list[float]:
        if hasattr(value, "tolist"):
            value = value.tolist()

        # Generator from some embedding libraries.
        if not isinstance(value, list):
            value = list(value)

        # Some libraries return [[...]] for one input.
        if value and isinstance(value[0], list):
            value = value[0]

        return [float(item) for item in value]