from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.chat import ChatMessage, Conversation, ToolCall


class ChatRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create_conversation(
        self,
        *,
        user_id: int,
        title: str,
    ) -> Conversation:
        conversation = Conversation(
            user_id=user_id,
            title=title,
        )
        self.db.add(conversation)
        self.db.flush()
        return conversation

    def get_conversation_for_user(
        self,
        *,
        conversation_id: int,
        user_id: int,
    ) -> Conversation | None:
        return self.db.scalar(
            select(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id,
                Conversation.deleted_at.is_(None),
            )
        )

    def list_conversations_for_user(self, *, user_id: int) -> list[Conversation]:
        statement = (
            select(Conversation)
            .where(
                Conversation.user_id == user_id,
                Conversation.deleted_at.is_(None),
            )
            .order_by(Conversation.updated_at.desc())
        )
        return list(self.db.scalars(statement).all())

    def soft_delete_conversation(
        self,
        *,
        conversation: Conversation,
    ) -> Conversation:
        conversation.deleted_at = datetime.utcnow()
        conversation.updated_at = datetime.utcnow()
        self.db.flush()
        return conversation

    def update_conversation_timestamp(self, *, conversation: Conversation) -> None:
        conversation.updated_at = datetime.utcnow()
        self.db.flush()

    def create_message(
        self,
        *,
        conversation_id: int,
        role: str,
        content: str,
        redacted_content: str | None = None,
        message_metadata: dict | None = None,
        retrieval_snapshot_key: str | None = None,
        request_id: str | None = None,
        trace_id: str | None = None,
    ) -> ChatMessage:
        message = ChatMessage(
            conversation_id=conversation_id,
            role=role,
            content=content,
            redacted_content=redacted_content,
            message_metadata=message_metadata or {},
            retrieval_snapshot_key=retrieval_snapshot_key,
            request_id=request_id,
            trace_id=trace_id,
        )
        self.db.add(message)
        self.db.flush()
        return message

    def list_messages(
        self,
        *,
        conversation_id: int,
        limit: int = 50,
    ) -> list[ChatMessage]:
        statement = (
            select(ChatMessage)
            .where(ChatMessage.conversation_id == conversation_id)
            .order_by(ChatMessage.created_at.desc())
            .limit(limit)
        )
        messages = list(self.db.scalars(statement).all())
        return list(reversed(messages))

    def create_tool_call(
        self,
        *,
        conversation_id: int,
        message_id: int | None,
        tool_name: str,
        input_json: dict,
        output_json: dict,
        status: str,
        error_message: str | None = None,
        latency_ms: int | None = None,
        request_id: str | None = None,
        trace_id: str | None = None,
    ) -> ToolCall:
        tool_call = ToolCall(
            conversation_id=conversation_id,
            message_id=message_id,
            tool_name=tool_name,
            input_json=input_json,
            output_json=output_json,
            status=status,
            error_message=error_message,
            latency_ms=latency_ms,
            request_id=request_id,
            trace_id=trace_id,
        )
        self.db.add(tool_call)
        self.db.flush()
        return tool_call
    
    def set_message_retrieval_snapshot_key(
        self,
        *,
        message: ChatMessage,
        retrieval_snapshot_key: str,
    ) -> ChatMessage:
        message.retrieval_snapshot_key = retrieval_snapshot_key
        self.db.flush()
        return message