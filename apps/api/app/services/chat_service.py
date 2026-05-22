from sqlalchemy.orm import Session

from app.infra.redis import RedisShortTermMemory
from app.models.chat import Conversation
from app.models.user import User
from app.repositories.chat_repository import ChatRepository
from app.schemas.chat import ChatRequest

#Owns chat business logic. For now, this creates/continues a conversation, saves messages, writes to Redis, and returns a placeholder answer. In the next step, we replace the placeholder with tool-calling LLM orchestration.
class ChatError(RuntimeError):
    pass


class ChatService:
    def __init__(
        self,
        *,
        db: Session,
        short_term_memory: RedisShortTermMemory,
    ) -> None:
        self.db = db
        self.short_term_memory = short_term_memory
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

    def handle_message(
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

        # Temporary answer for Step 1 only.
        # In Step 2/3, this becomes the tool-calling LLM answer.
        answer = (
            "I saved your message and conversation state. "
            "Next, I will connect this chat endpoint to the classifier, NER, summarizer, and RAG tools."
        )

        assistant_message = self.chat_repo.create_message(
            conversation_id=conversation.id,
            role="assistant",
            content=answer,
            message_metadata={
                "mode": "step_1_persistence_smoke_test",
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