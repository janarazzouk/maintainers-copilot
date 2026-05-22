from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.infra.minio import MinIOObjectStore
from app.models.chat import ChatMessage
from app.models.user import User
from app.services.tool_service import ToolResult


class RagSnapshotService:
    """Stores per-message RAG retrieval snapshots in MinIO.

    The online RAG index stays in Postgres/pgvector.
    MinIO stores reproducibility/debug snapshots of what was retrieved.
    """

    def __init__(self, *, object_store: MinIOObjectStore) -> None:
        self.object_store = object_store

    def save_snapshot_if_present(
        self,
        *,
        user: User,
        conversation_id: int,
        user_message: ChatMessage,
        assistant_message: ChatMessage,
        user_question: str,
        repo: str | None,
        tool_results: list[ToolResult],
    ) -> str | None:
        rag_result = self._find_successful_rag_result(tool_results)
        if rag_result is None:
            return None

        snapshot_key = self._make_snapshot_key(
            conversation_id=conversation_id,
            assistant_message_id=assistant_message.id,
        )

        payload = {
            "schema_version": "1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "user_id": user.id,
            "conversation_id": conversation_id,
            "user_message_id": user_message.id,
            "assistant_message_id": assistant_message.id,
            "repo": repo,
            "question": user_question,
            "rag_tool": {
                "tool_name": rag_result.tool_name,
                "status": rag_result.status,
                "latency_ms": rag_result.latency_ms,
                "input": rag_result.input_json,
            },
            "retrieval": self._normalize_rag_output(rag_result.output_json),
            "raw_rag_output": rag_result.output_json,
        }

        return self.object_store.put_json(key=snapshot_key, payload=payload)

    def _find_successful_rag_result(
        self,
        tool_results: list[ToolResult],
    ) -> ToolResult | None:
        for result in tool_results:
            if result.tool_name == "rag_search" and result.status == "success":
                return result
        return None

    def _make_snapshot_key(
        self,
        *,
        conversation_id: int,
        assistant_message_id: int,
    ) -> str:
        return (
            f"conversations/{conversation_id}/retrieved_chunks/"
            f"assistant_message_{assistant_message_id}.json"
        )

    def _normalize_rag_output(self, output: dict[str, Any]) -> dict[str, Any]:
        """Keep common fields easy to inspect in MinIO.

        Your RagService may return sources/chunks/results depending on its schema.
        This method preserves the full output under raw_rag_output, while also
        extracting the most useful retrieval fields into a stable shape.
        """

        sources = (
            output.get("sources")
            or output.get("retrieved_chunks")
            or output.get("chunks")
            or output.get("results")
            or []
        )

        normalized_sources: list[dict[str, Any]] = []
        if isinstance(sources, list):
            for item in sources:
                if isinstance(item, dict):
                    normalized_sources.append(
                        {
                            "chunk_id": item.get("chunk_id") or item.get("id"),
                            "document_id": item.get("document_id"),
                            "source_type": item.get("source_type"),
                            "source_id": item.get("source_id"),
                            "title": item.get("title"),
                            "url": item.get("url"),
                            "score": item.get("score"),
                            "label": item.get("label"),
                            "metadata": item.get("metadata", {}),
                            "content": (
                                item.get("content")
                                or item.get("text")
                                or item.get("chunk_text")
                                or item.get("body")
                            ),
                        }
                    )
                else:
                    normalized_sources.append({"value": str(item)})

        return {
            "answer": output.get("answer"),
            "query": output.get("query") or output.get("question"),
            "rewritten_queries": output.get("rewritten_queries"),
            "top_k": output.get("top_k"),
            "sources_count": len(normalized_sources),
            "sources": normalized_sources,
        }