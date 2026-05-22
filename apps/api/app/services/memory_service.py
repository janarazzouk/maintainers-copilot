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
    # Similarity is 1 - cosine distance.
    # 0.88 is strict enough to catch duplicates like:
    # "changed-files-only patches preferred"
    # and
    # "The user prefers changed-files-only patches, not full project rewrites."
    # without merging unrelated preferences too aggressively.
    DEDUP_SIMILARITY_THRESHOLD = 0.88

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
        normalized_content = self._normalize_for_dedup(redacted_content)
        embedding = self._embed_text(redacted_content)

        exact_duplicate = self.memories.find_exact_memory_for_user(
            user_id=user.id,
            normalized_content=normalized_content,
        )
        if exact_duplicate is not None:
            memory = self.memories.mark_duplicate_attempt(
                memory=exact_duplicate,
                duplicate_content=redacted_content,
                duplicate_reason=reason,
                duplicate_similarity=1.0,
            )
            self.audit.create(
                actor=user.email,
                action="memory.write_duplicate_skipped",
                target=f"memory:{memory.id}",
            )
            return memory

        semantic_duplicate = self._find_semantic_duplicate(
            user=user,
            embedding=embedding,
        )
        if semantic_duplicate is not None:
            memory, similarity = semantic_duplicate
            memory = self.memories.mark_duplicate_attempt(
                memory=memory,
                duplicate_content=redacted_content,
                duplicate_reason=reason,
                duplicate_similarity=similarity,
            )
            self.audit.create(
                actor=user.email,
                action="memory.write_duplicate_skipped",
                target=f"memory:{memory.id}",
            )
            return memory

        merged_metadata = {
            "reason": reason,
            "dedup_similarity_threshold": self.DEDUP_SIMILARITY_THRESHOLD,
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
        return self.memories.list_memories_for_user(
            user_id=user.id,
            limit=limit,
        )

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

    def _find_semantic_duplicate(
        self,
        *,
        user: User,
        embedding: list[float],
    ) -> tuple[LongTermMemory, float] | None:
        rows = self.memories.search_memories_for_user(
            user_id=user.id,
            query_embedding=embedding,
            limit=5,
        )

        for memory, distance in rows:
            similarity = 1.0 - distance
            if similarity >= self.DEDUP_SIMILARITY_THRESHOLD:
                return memory, round(similarity, 4)

        return None

    def _normalize_for_dedup(self, text: str) -> str:
        return " ".join(text.lower().strip().split())

    def _embed_text(self, text: str) -> list[float]:
        method_names = [
            "embed_query",
            "embed_text",
            "embed",
            "encode",
            "embed_documents",
            "embed_texts",
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

        if not isinstance(value, list):
            value = list(value)

        if value and isinstance(value[0], list):
            value = value[0]

        return [float(item) for item in value]