from __future__ import annotations

import re
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
    # 0.82 is more practical than 0.88 for short preference memories.
    DEDUP_SIMILARITY_THRESHOLD = 0.82

    # Token overlap catches wording variants that embeddings may miss, such as:
    # "changed-files-only patches" vs "only changed files when receiving code fixes"
    DEDUP_TOKEN_OVERLAP_THRESHOLD = 0.45

    STOPWORDS = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "to",
        "for",
        "of",
        "in",
        "on",
        "when",
        "with",
        "this",
        "that",
        "is",
        "are",
        "be",
        "by",
        "as",
        "it",
        "not",
    }

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

        hybrid_duplicate = self._find_hybrid_duplicate(
            user=user,
            redacted_content=redacted_content,
            embedding=embedding,
        )
        if hybrid_duplicate is not None:
            memory, similarity, method = hybrid_duplicate
            memory = self.memories.mark_duplicate_attempt(
                memory=memory,
                duplicate_content=redacted_content,
                duplicate_reason=reason,
                duplicate_similarity=similarity,
            )

            metadata_copy = dict(memory.memory_metadata or {})
            metadata_copy["last_duplicate_method"] = method
            memory.memory_metadata = metadata_copy

            self.audit.create(
                actor=user.email,
                action="memory.write_duplicate_skipped",
                target=f"memory:{memory.id}",
            )
            return memory

        merged_metadata = {
            "reason": reason,
            "dedup_similarity_threshold": self.DEDUP_SIMILARITY_THRESHOLD,
            "dedup_token_overlap_threshold": self.DEDUP_TOKEN_OVERLAP_THRESHOLD,
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

    def _find_hybrid_duplicate(
        self,
        *,
        user: User,
        redacted_content: str,
        embedding: list[float],
    ) -> tuple[LongTermMemory, float, str] | None:
        rows = self.memories.search_memories_for_user(
            user_id=user.id,
            query_embedding=embedding,
            limit=10,
        )

        best_token_match: tuple[LongTermMemory, float] | None = None

        for memory, distance in rows:
            vector_similarity = 1.0 - distance
            if vector_similarity >= self.DEDUP_SIMILARITY_THRESHOLD:
                return memory, round(vector_similarity, 4), "vector_similarity"

            token_overlap = self._token_overlap_similarity(
                redacted_content,
                memory.redacted_content,
            )

            if best_token_match is None or token_overlap > best_token_match[1]:
                best_token_match = (memory, token_overlap)

        if best_token_match is not None:
            memory, token_overlap = best_token_match
            if token_overlap >= self.DEDUP_TOKEN_OVERLAP_THRESHOLD:
                return memory, round(token_overlap, 4), "token_overlap"

        return None

    def _token_overlap_similarity(self, left: str, right: str) -> float:
        left_tokens = self._content_tokens(left)
        right_tokens = self._content_tokens(right)

        if not left_tokens or not right_tokens:
            return 0.0

        intersection = left_tokens.intersection(right_tokens)
        smaller_size = min(len(left_tokens), len(right_tokens))

        return len(intersection) / smaller_size

    def _content_tokens(self, text: str) -> set[str]:
        normalized = text.lower()
        normalized = normalized.replace("-", " ")
        tokens = re.findall(r"[a-z0-9_]+", normalized)

        return {
            token
            for token in tokens
            if token not in self.STOPWORDS and len(token) >= 3
        }

    def _normalize_for_dedup(self, text: str) -> str:
        normalized = text.lower().strip()
        normalized = normalized.replace("-", " ")
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized

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