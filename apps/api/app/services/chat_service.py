from sqlalchemy.orm import Session

from app.infra.redis import RedisShortTermMemory
from app.models.chat import Conversation
from app.models.user import User
from app.repositories.chat_repository import ChatRepository
from app.schemas.chat import ChatRequest
from app.services.llm_chat_service import LLMChatError, LLMChatService
from app.services.rag_snapshot_service import RagSnapshotService
from app.services.tool_service import ToolResult, ToolService

#Chat service now asks the LLM to choose tools. If Groq is unavailable, it falls back to your deterministic Step 2 plan.
class ChatError(RuntimeError):
    pass


class ChatService:
    def __init__(
        self,
        *,
        db: Session,
        short_term_memory: RedisShortTermMemory,
        tool_service: ToolService | None = None,
        rag_snapshot_service: RagSnapshotService | None = None,
        llm_chat_service: LLMChatService | None = None,
    ) -> None:
        self.db = db
        self.short_term_memory = short_term_memory
        self.tool_service = tool_service
        self.rag_snapshot_service = rag_snapshot_service
        self.llm_chat_service = llm_chat_service
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

        existing_recent_messages = self.short_term_memory.get_recent_messages(conversation.id)

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

        answer, tool_results, mode = await self._generate_answer(
            message=payload.message,
            repo=payload.repo,
            recent_messages=existing_recent_messages,
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

        assistant_message = self.chat_repo.create_message(
            conversation_id=conversation.id,
            role="assistant",
            content=answer,
            message_metadata={
                "mode": mode,
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

        if self.rag_snapshot_service is not None:
            snapshot_key = self.rag_snapshot_service.save_snapshot_if_present(
                user=user,
                conversation_id=conversation.id,
                user_message=user_message,
                assistant_message=assistant_message,
                user_question=payload.message,
                repo=payload.repo,
                tool_results=tool_results,
            )
            if snapshot_key is not None:
                self.chat_repo.set_message_retrieval_snapshot_key(
                    message=assistant_message,
                    retrieval_snapshot_key=snapshot_key,
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

    async def _generate_answer(
        self,
        *,
        message: str,
        repo: str | None,
        recent_messages: list[dict],
    ) -> tuple[str, list[ToolResult], str]:
        if self.llm_chat_service is not None:
            try:
                answer, tool_results = await self.llm_chat_service.answer_with_tools(
                    user_message=message,
                    repo=repo,
                    recent_messages=recent_messages,
                )
                return answer, tool_results, "groq_tool_calling_llm"
            except LLMChatError as exc:
                fallback_note = (
                    "The Groq tool-calling LLM was unavailable, so I fell back to the "
                    f"backend deterministic triage tools. LLM error: {exc}"
                )
                if self.tool_service is None:
                    return fallback_note, [], "llm_unavailable_no_tools"

                tool_results = await self.tool_service.run_triage_tools(
                    message=message,
                    repo=repo,
                )
                return (
                    fallback_note + "\n\n" + self._build_fallback_answer(tool_results),
                    tool_results,
                    "deterministic_tool_fallback",
                )

        if self.tool_service is not None:
            tool_results = await self.tool_service.run_triage_tools(
                message=message,
                repo=repo,
            )
            return self._build_fallback_answer(tool_results), tool_results, "deterministic_tool_fallback"

        return (
            "I saved your message, but no LLM or tools are currently configured.",
            [],
            "no_llm_no_tools",
        )

    def _build_fallback_answer(self, tool_results: list[ToolResult]) -> str:
        if not tool_results:
            return "No tools were available for this request."

        lines: list[str] = ["I ran the backend triage tools for this message.", ""]

        for result in tool_results:
            if result.tool_name == "classify_issue":
                lines.append(self._format_classification(result))
            elif result.tool_name == "extract_entities":
                lines.append(self._format_entities(result))
            elif result.tool_name == "rag_search":
                lines.append(self._format_rag(result))
            elif result.tool_name == "summarize_thread":
                lines.append(self._format_summary(result))

        failed_tools = [result for result in tool_results if result.status != "success"]
        if failed_tools:
            lines.append("")
            lines.append("Some tools were unavailable, but the chat request did not crash:")
            for result in failed_tools:
                lines.append(f"- {result.tool_name}: {result.error_message}")

        return "\n".join(lines).strip()

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