import json
from typing import Any

from redis import Redis
from redis.exceptions import RedisError

#This is the Redis adapter. It stores recent conversation messages temporarily. Postgres is permanent storage; Redis is fast short-term memory.
class RedisMemoryError(RuntimeError):
    pass


class RedisShortTermMemory:
    """Redis adapter for short-term conversation state.

    This is not long-term memory. It only keeps recent chat turns for a TTL.
    Permanent messages stay in Postgres.
    """

    def __init__(self, redis_url: str, ttl_seconds: int) -> None:
        self.ttl_seconds = ttl_seconds
        self._client: Redis = Redis.from_url(redis_url, decode_responses=True)

    def ping(self) -> None:
        try:
            self._client.ping()
        except RedisError as exc:
            raise RedisMemoryError("Redis is unreachable.") from exc

    def _key(self, conversation_id: int) -> str:
        return f"chat:conversation:{conversation_id}:recent_messages"

    def get_recent_messages(self, conversation_id: int) -> list[dict[str, Any]]:
        try:
            raw = self._client.get(self._key(conversation_id))
        except RedisError as exc:
            raise RedisMemoryError("Failed to read short-term memory from Redis.") from exc

        if not raw:
            return []

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return []

        if not isinstance(data, list):
            return []

        return data

    def append_message(
        self,
        *,
        conversation_id: int,
        role: str,
        content: str,
        limit: int = 12,
    ) -> None:
        messages = self.get_recent_messages(conversation_id)
        messages.append({"role": role, "content": content})
        messages = messages[-limit:]

        try:
            self._client.set(
                self._key(conversation_id),
                json.dumps(messages, ensure_ascii=False),
                ex=self.ttl_seconds,
            )
        except RedisError as exc:
            raise RedisMemoryError("Failed to write short-term memory to Redis.") from exc

    def clear_conversation(self, conversation_id: int) -> None:
        try:
            self._client.delete(self._key(conversation_id))
        except RedisError as exc:
            raise RedisMemoryError("Failed to clear short-term memory from Redis.") from exc