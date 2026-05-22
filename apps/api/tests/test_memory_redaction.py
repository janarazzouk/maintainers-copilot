from types import SimpleNamespace
from typing import Any

import pytest

from app.services import memory_service as memory_service_module
from app.services.memory_service import LongTermMemoryService


class FakeEmbeddingModel:
    def embed_query(self, text: str) -> list[float]:
        return [0.1] * 384


class FakeMemory:
    def __init__(
        self,
        *,
        memory_id: int,
        user_id: int,
        memory_type: str,
        content: str,
        redacted_content: str,
        memory_metadata: dict[str, Any],
    ) -> None:
        self.id = memory_id
        self.user_id = user_id
        self.memory_type = memory_type
        self.content = content
        self.redacted_content = redacted_content
        self.memory_metadata = memory_metadata


class FakeMemoryRepository:
    created_memory: FakeMemory | None = None

    def __init__(self, db: object) -> None:
        self.db = db

    def find_exact_memory_for_user(
        self,
        *,
        user_id: int,
        normalized_content: str,
    ) -> None:
        return None

    def search_memories_for_user(
        self,
        *,
        user_id: int,
        query_embedding: list[float],
        limit: int = 5,
    ) -> list[tuple[FakeMemory, float]]:
        return []

    def create_memory(
        self,
        *,
        user_id: int,
        memory_type: str,
        content: str,
        redacted_content: str,
        embedding: list[float],
        memory_metadata: dict[str, Any] | None = None,
    ) -> FakeMemory:
        memory = FakeMemory(
            memory_id=1,
            user_id=user_id,
            memory_type=memory_type,
            content=content,
            redacted_content=redacted_content,
            memory_metadata=memory_metadata or {},
        )
        FakeMemoryRepository.created_memory = memory
        return memory


class FakeAuditRepository:
    actions: list[tuple[str, str, str]] = []

    def __init__(self, db: object) -> None:
        self.db = db

    def create(self, *, actor: str, action: str, target: str) -> None:
        self.actions.append((actor, action, target))


@pytest.fixture(autouse=True)
def patch_repositories(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeMemoryRepository.created_memory = None
    FakeAuditRepository.actions = []

    monkeypatch.setattr(memory_service_module, "MemoryRepository", FakeMemoryRepository)
    monkeypatch.setattr(memory_service_module, "AuditRepository", FakeAuditRepository)


def test_memory_write_never_stores_raw_secret_content() -> None:
    user = SimpleNamespace(id=1, email="admin@example.com")
    service = LongTermMemoryService(
        db=object(),
        embedding_model=FakeEmbeddingModel(),
    )

    memory = service.write_memory(
        user=user,
        memory_type="semantic",
        content=(
            "Remember my GitHub token ghp_abcdefghijklmnopqrstuvwxyz123456 "
            "and Authorization: Bearer abc.def.ghi"
        ),
        reason="redaction test",
    )

    assert "ghp_abcdefghijklmnopqrstuvwxyz123456" not in memory.content
    assert "abc.def.ghi" not in memory.content

    assert "ghp_abcdefghijklmnopqrstuvwxyz123456" not in memory.redacted_content
    assert "abc.def.ghi" not in memory.redacted_content

    assert "[REDACTED_GITHUB_TOKEN]" in memory.content
    assert "Bearer [REDACTED]" in memory.content

    assert memory.content == memory.redacted_content


def test_memory_write_creates_audit_log() -> None:
    user = SimpleNamespace(id=1, email="admin@example.com")
    service = LongTermMemoryService(
        db=object(),
        embedding_model=FakeEmbeddingModel(),
    )

    memory = service.write_memory(
        user=user,
        memory_type="semantic",
        content="User prefers changed-files-only patches.",
        reason="audit test",
    )

    assert memory.id == 1
    assert FakeAuditRepository.actions == [
        ("admin@example.com", "memory.write", "memory:1")
    ]