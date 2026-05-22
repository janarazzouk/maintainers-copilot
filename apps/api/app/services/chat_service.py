from sqlalchemy.orm import Session

from app.infra.redis import RedisShortTermMemory
from app.models.chat import Conversation
from app.models.user import User
from app.repositories.chat_repository import ChatRepository
from app.schemas.chat import ChatRequest
from app.services.tool_service import ToolResult, ToolService

#This version saves the message, runs tools, saves tool calls, writes to Redis, and returns a real triage-style response.
class ChatError(RuntimeError):
    pass


class ChatService:
    def __init__(
        self,
        *,
        db: Session,
        short_term_memory: RedisShortTermMemory,
        tool_service: ToolService | None = None,
    ) -> None:
        self.db = db
        self.short_term_memory = short_term_memory
        self.tool_service = tool_service
        self.chat_repo = ChatRepository(db)

    def _make_title(self, message: str) -> str:
        cleaned = " ".join(message.strip().split())
        if len(cleaned) <= 60:
            return cleaned
        return cleaned[:57] + "..."

    def _get_or_create_conversation(
        self,
        *,
        user: User,
        payload: ChatRequest,
    ) -> Conversation:
        if payload.conversation_id is None:
            return self.chat_repo.create_conversation(
                user_id=user.id,
                title=self._make_title(payload.message),
            )

        conversation = self.chat_repo.get_conversation_for_user(
            conversation_id=payload.conversation_id,
            user_id=user.id,
        )
        if conversation is None:
            raise ChatError("Conversation not found.")

        return conversation

    def list_conversations(self, *, user: User) -> list[Conversation]:
        return self.chat_repo.list_conversations_for_user(user_id=user.id)

    def get_messages_for_conversation(
        self,
        *,
        user: User,
        conversation_id: int,
    ):
        conversation = self.chat_repo.get_conversation_for_user(
            conversation_id=conversation_id,
            user_id=user.id,
        )
        if conversation is None:
            raise ChatError("Conversation not found.")

        return self.chat_repo.list_messages(conversation_id=conversation.id)

    def soft_delete_conversation(
        self,
        *,
        user: User,
        conversation_id: int,
    ) -> None:
        conversation = self.chat_repo.get_conversation_for_user(
            conversation_id=conversation_id,
            user_id=user.id,
        )
        if conversation is None:
            raise ChatError("Conversation not found.")

        self.chat_repo.soft_delete_conversation(conversation=conversation)
        self.short_term_memory.clear_conversation(conversation.id)
        self.db.commit()

    async def handle_message(
        self,
        *,
        user: User,
        payload: ChatRequest,
    ) -> tuple[int, object, object, str, list[dict]]:
        conversation = self._get_or_create_conversation(user=user, payload=payload)

        user_message = self.chat_repo.create_message(
            conversation_id=conversation.id,
            role="user",
            content=payload.message,
            message_metadata={
                "repo": payload.repo,
                "source": "api",
            },
        )

        self.short_term_memory.append_message(
            conversation_id=conversation.id,
            role="user",
            content=payload.message,
        )

        tool_results: list[ToolResult] = []
        if self.tool_service is not None:
            tool_results = await self.tool_service.run_triage_tools(
                message=payload.message,
                repo=payload.repo,
            )

        for result in tool_results:
            self.chat_repo.create_tool_call(
                conversation_id=conversation.id,
                message_id=user_message.id,
                tool_name=result.tool_name,
                input_json=result.input_json,
                output_json=result.output_json,
                status=result.status,
                error_message=result.error_message,
                latency_ms=result.latency_ms,
            )

        answer = self._build_step2_answer(tool_results)

        assistant_message = self.chat_repo.create_message(
            conversation_id=conversation.id,
            role="assistant",
            content=answer,
            message_metadata={
                "mode": "step_2_tool_service_smoke_test",
                "tools_run": [
                    {
                        "tool_name": result.tool_name,
                        "status": result.status,
                        "latency_ms": result.latency_ms,
                    }
                    for result in tool_results
                ],
            },
        )

        self.short_term_memory.append_message(
            conversation_id=conversation.id,
            role="assistant",
            content=answer,
        )

        self.chat_repo.update_conversation_timestamp(conversation=conversation)
        self.db.commit()

        self.db.refresh(user_message)
        self.db.refresh(assistant_message)

        recent_messages = self.short_term_memory.get_recent_messages(conversation.id)

        return conversation.id, user_message, assistant_message, answer, recent_messages

    def _build_step2_answer(self, tool_results: list[ToolResult]) -> str:
        if not tool_results:
            return (
                "I saved your message and conversation state. "
                "No tools were available for this request yet."
            )

        lines: list[str] = [
            "I ran the backend triage tools for this message.",
            "",
        ]

        classify_result = self._find_tool(tool_results, "classify_issue")
        if classify_result is not None:
            lines.append(self._format_classification(classify_result))

        entity_result = self._find_tool(tool_results, "extract_entities")
        if entity_result is not None:
            lines.append(self._format_entities(entity_result))

        rag_result = self._find_tool(tool_results, "rag_search")
        if rag_result is not None:
            lines.append(self._format_rag(rag_result))

        summarize_result = self._find_tool(tool_results, "summarize_thread")
        if summarize_result is not None:
            lines.append(self._format_summary(summarize_result))

        failed_tools = [result for result in tool_results if result.status != "success"]
        if failed_tools:
            lines.append("")
            lines.append("Some tools were unavailable, but the chat request did not crash:")
            for result in failed_tools:
                lines.append(f"- {result.tool_name}: {result.error_message}")

        return "\n".join(lines).strip()

    def _find_tool(
        self,
        tool_results: list[ToolResult],
        tool_name: str,
    ) -> ToolResult | None:
        for result in tool_results:
            if result.tool_name == tool_name:
                return result
        return None

    def _format_classification(self, result: ToolResult) -> str:
        if result.status != "success":
            return "- Classification: unavailable."

        output = result.output_json
        label = (
            output.get("label")
            or output.get("predicted_label")
            or output.get("class_name")
            or output.get("prediction")
            or "unknown"
        )
        confidence = output.get("confidence") or output.get("score") or output.get("probability")

        if confidence is None:
            return f"- Classification: `{label}`."

        return f"- Classification: `{label}` with confidence `{confidence}`."

    def _format_entities(self, result: ToolResult) -> str:
        if result.status != "success":
            return "- Entities: unavailable."

        output = result.output_json
        entities = output.get("entities", [])

        if not entities:
            return "- Entities: no code-shaped entities found."

        previews: list[str] = []
        for entity in entities[:8]:
            if isinstance(entity, dict):
                text = entity.get("text") or entity.get("value") or str(entity)
                entity_type = entity.get("type") or entity.get("label") or "ENTITY"
                previews.append(f"{text} ({entity_type})")
            else:
                previews.append(str(entity))

        return "- Entities: " + ", ".join(previews)

    def _format_rag(self, result: ToolResult) -> str:
        if result.status != "success":
            return "- RAG: unavailable."

        output = result.output_json
        answer = output.get("answer", "No RAG answer returned.")
        sources = output.get("sources", [])

        line = f"- RAG: {answer}"

        if sources:
            top_source = sources[0]
            title = top_source.get("title", "unknown source")
            score = top_source.get("score")
            if score is not None:
                line += f" Top source: `{title}` with score `{score}`."
            else:
                line += f" Top source: `{title}`."

        return line

    def _format_summary(self, result: ToolResult) -> str:
        if result.status != "success":
            return "- Summary: unavailable."

        output = result.output_json
        summary = output.get("summary") or output.get("answer") or output.get("text")

        if not summary:
            return "- Summary: summarizer returned no summary field."

        return f"- Summary: {summary}"