from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from sqlalchemy.orm import Session

from app.infra.embeddings import EmbeddingModel
from app.infra.groq_client import GroqLLMClient
from app.infra.model_server_client import ModelServerClient, ModelServerError
from app.models.user import User
from app.schemas.rag import RagQueryRequest
from app.services.memory_service import LongTermMemoryService, MemoryError
from app.services.rag_service import RagService


@dataclass(frozen=True)
class ToolResult:
    tool_name: str
    input_json: dict[str, Any]
    output_json: dict[str, Any]
    status: str
    error_message: str | None
    latency_ms: int


class ToolService:
    def __init__(
        self,
        *,
        db: Session,
        model_client: ModelServerClient,
        embedding_model: EmbeddingModel,
        llm_client: GroqLLMClient | None,
        current_user: User | None = None,
        memory_service: LongTermMemoryService | None = None,
    ) -> None:
        self.db = db
        self.model_client = model_client
        self.embedding_model = embedding_model
        self.llm_client = llm_client
        self.current_user = current_user
        self.memory_service = memory_service

    async def execute_chat_tool(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        fallback_message: str,
        repo: str | None,
    ) -> ToolResult:
        title = str(arguments.get("title") or self._title_from_message(fallback_message))
        body = str(arguments.get("body") or fallback_message)

        if tool_name == "classify_issue":
            return await self.classify_issue(title=title, body=body)

        if tool_name == "extract_entities":
            return await self.extract_entities(title=title, body=body)

        if tool_name == "summarize_thread":
            return await self.summarize_thread(title=title, body=body)

        if tool_name == "rag_search":
            question = str(arguments.get("question") or fallback_message)
            raw_top_k = arguments.get("top_k", 5)

            try:
                top_k = max(1, min(int(raw_top_k), 10))
            except (TypeError, ValueError):
                top_k = 5

            label_filter = arguments.get("label_filter")
            if label_filter is not None:
                label_filter = str(label_filter)

            return self.rag_search(
                question=question,
                top_k=top_k,
                label_filter=label_filter,
            )

        if tool_name == "write_memory":
            return self.write_memory(arguments=arguments)

        return ToolResult(
            tool_name=tool_name,
            input_json=arguments,
            output_json={
                "available": False,
                "message": f"Unknown tool: {tool_name}",
            },
            status="error",
            error_message=f"Unknown tool: {tool_name}",
            latency_ms=0,
        )

    async def classify_issue(self, *, title: str, body: str) -> ToolResult:
        input_json = {"title": title, "body": body}
        start = perf_counter()

        try:
            output = await self.model_client.classify(title=title, body=body)
            return self._success(
                tool_name="classify_issue",
                input_json=input_json,
                output_json=output,
                start=start,
            )
        except ModelServerError as exc:
            return self._failure(
                tool_name="classify_issue",
                input_json=input_json,
                error=exc,
                start=start,
            )

    async def extract_entities(self, *, title: str, body: str) -> ToolResult:
        input_json = {"title": title, "body": body}
        start = perf_counter()

        try:
            output = await self.model_client.extract_entities(title=title, body=body)
            return self._success(
                tool_name="extract_entities",
                input_json=input_json,
                output_json=output,
                start=start,
            )
        except ModelServerError as exc:
            return self._failure(
                tool_name="extract_entities",
                input_json=input_json,
                error=exc,
                start=start,
            )

    async def summarize_thread(self, *, title: str, body: str) -> ToolResult:
        input_json = {"title": title, "body": body}
        start = perf_counter()

        try:
            output = await self.model_client.summarize(title=title, body=body)
            return self._success(
                tool_name="summarize_thread",
                input_json=input_json,
                output_json=output,
                start=start,
            )
        except ModelServerError as exc:
            return self._failure(
                tool_name="summarize_thread",
                input_json=input_json,
                error=exc,
                start=start,
            )

    def rag_search(
        self,
        *,
        question: str,
        top_k: int = 5,
        label_filter: str | None = None,
    ) -> ToolResult:
        input_json = {
            "question": question,
            "top_k": top_k,
            "label_filter": label_filter,
            "generate_answer": False,
        }
        start = perf_counter()

        try:
            service = RagService(
                db=self.db,
                embedding_model=self.embedding_model,
                llm_client=self.llm_client,
            )
            response = service.query(
                RagQueryRequest(
                    question=question,
                    top_k=top_k,
                    label_filter=label_filter,
                    generate_answer=False,
                )
            )
            return self._success(
                tool_name="rag_search",
                input_json=input_json,
                output_json=response.model_dump(mode="json"),
                start=start,
            )
        except Exception as exc:
            return self._failure(
                tool_name="rag_search",
                input_json=input_json,
                error=exc,
                start=start,
            )

    def write_memory(self, *, arguments: dict[str, Any]) -> ToolResult:
        input_json = dict(arguments)
        start = perf_counter()

        if self.current_user is None or self.memory_service is None:
            return self._failure(
                tool_name="write_memory",
                input_json=input_json,
                error=MemoryError("Memory service is not configured."),
                start=start,
            )

        memory_type = str(arguments.get("memory_type") or "semantic")
        content = str(arguments.get("content") or "").strip()

        reason = arguments.get("reason")
        if reason is not None:
            reason = str(reason)

        try:
            memory = self.memory_service.write_memory(
                user=self.current_user,
                memory_type=memory_type,
                content=content,
                reason=reason,
                metadata={
                    "source": "write_memory_tool",
                },
            )

            return self._success(
                tool_name="write_memory",
                input_json=input_json,
                output_json={
                    "memory_id": memory.id,
                    "memory_type": memory.memory_type,
                    "content": memory.redacted_content,
                    "stored": True,
                },
                start=start,
            )
        except MemoryError as exc:
            return self._failure(
                tool_name="write_memory",
                input_json=input_json,
                error=exc,
                start=start,
            )

    async def run_triage_tools(
        self,
        *,
        message: str,
        repo: str | None,
    ) -> list[ToolResult]:
        title = self._title_from_message(message)
        body = message

        results: list[ToolResult] = []
        results.append(await self.classify_issue(title=title, body=body))
        results.append(await self.extract_entities(title=title, body=body))
        results.append(self.rag_search(question=message, top_k=5, label_filter=None))

        if "summarize" in message.lower() or "summary" in message.lower():
            results.append(await self.summarize_thread(title=title, body=body))

        return results

    def _title_from_message(self, message: str) -> str:
        cleaned = " ".join(message.strip().split())
        if len(cleaned) <= 120:
            return cleaned
        return cleaned[:117] + "..."

    def _success(
        self,
        *,
        tool_name: str,
        input_json: dict[str, Any],
        output_json: dict[str, Any],
        start: float,
    ) -> ToolResult:
        return ToolResult(
            tool_name=tool_name,
            input_json=input_json,
            output_json=output_json,
            status="success",
            error_message=None,
            latency_ms=self._elapsed_ms(start),
        )

    def _failure(
        self,
        *,
        tool_name: str,
        input_json: dict[str, Any],
        error: Exception,
        start: float,
    ) -> ToolResult:
        return ToolResult(
            tool_name=tool_name,
            input_json=input_json,
            output_json={
                "available": False,
                "message": str(error),
            },
            status="error",
            error_message=str(error),
            latency_ms=self._elapsed_ms(start),
        )

    def _elapsed_ms(self, start: float) -> int:
        return int((perf_counter() - start) * 1000)