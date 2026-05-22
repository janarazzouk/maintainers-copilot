from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from sqlalchemy.orm import Session

from app.infra.embeddings import EmbeddingModel
from app.infra.groq_client import GroqLLMClient
from app.infra.model_server_client import ModelServerClient, ModelServerError
from app.schemas.rag import RagQueryRequest
from app.services.rag_service import RagService
#This file wraps your existing model-server tools and RAG pipeline. It catches tool failures so the chatbot does not crash if one tool is unavailable.

@dataclass(frozen=True)
class ToolResult:
    tool_name: str
    input_json: dict[str, Any]
    output_json: dict[str, Any]
    status: str
    error_message: str | None
    latency_ms: int


class ToolService:
    """Backend tool runner for the chatbot.

    Step 2 uses this directly from ChatService.
    Step 3 will expose these same tools to the tool-calling LLM.
    """

    def __init__(
        self,
        *,
        db: Session,
        model_client: ModelServerClient,
        embedding_model: EmbeddingModel,
        llm_client: GroqLLMClient | None,
    ) -> None:
        self.db = db
        self.model_client = model_client
        self.embedding_model = embedding_model
        self.llm_client = llm_client

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

    async def run_triage_tools(
        self,
        *,
        message: str,
        repo: str | None,
    ) -> list[ToolResult]:
        """Temporary deterministic tool plan for Step 2.

        Later, the LLM will choose tools dynamically.
        """

        title = self._title_from_message(message)
        body = message

        results: list[ToolResult] = []

        results.append(await self.classify_issue(title=title, body=body))
        results.append(await self.extract_entities(title=title, body=body))
        results.append(
            self.rag_search(
                question=message,
                top_k=5,
                label_filter=None,
            )
        )

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