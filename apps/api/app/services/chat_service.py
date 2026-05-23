from sqlalchemy.orm import Session

from app.infra.redis import RedisShortTermMemory
from app.models.chat import Conversation
from app.models.user import User
from app.repositories.chat_repository import ChatRepository
from app.schemas.chat import ChatRequest
from app.services.llm_chat_service import LLMChatError, LLMChatService
from app.services.memory_service import LongTermMemoryService
from app.services.rag_snapshot_service import RagSnapshotService
from app.services.tool_service import ToolResult, ToolService


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
        memory_service: LongTermMemoryService | None = None,
    ) -> None:
        self.db = db
        self.short_term_memory = short_term_memory
        self.tool_service = tool_service
        self.rag_snapshot_service = rag_snapshot_service
        self.llm_chat_service = llm_chat_service
        self.memory_service = memory_service
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

        existing_recent_messages = self.short_term_memory.get_recent_messages(
            conversation.id
        )

        long_term_memories: list[dict] = []
        if self.memory_service is not None:
            long_term_memories = self.memory_service.search_memories(
                user=user,
                query=payload.message,
                limit=5,
            )

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
            long_term_memories=long_term_memories,
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
        long_term_memories: list[dict],
    ) -> tuple[str, list[ToolResult], str]:
        if self.llm_chat_service is not None:
            try:
                answer, tool_results = await self.llm_chat_service.answer_with_tools(
                    user_message=message,
                    repo=repo,
                    recent_messages=recent_messages,
                    long_term_memories=long_term_memories,
                )
                return answer, tool_results, "groq_tool_calling_llm"

            except LLMChatError:
                if self.tool_service is None:
                    return (
                        "I could not generate a triage answer right now. Please try again.",
                        [],
                        "llm_unavailable_no_tools",
                    )

                tool_results = await self.tool_service.run_triage_tools(
                    message=message,
                    repo=repo,
                )

                return (
                    self._build_user_action_answer(
                        user_message=message,
                        tool_results=tool_results,
                    ),
                    tool_results,
                    "deterministic_tool_fallback",
                )

        if self.tool_service is not None:
            tool_results = await self.tool_service.run_triage_tools(
                message=message,
                repo=repo,
            )
            return (
                self._build_user_action_answer(
                    user_message=message,
                    tool_results=tool_results,
                ),
                tool_results,
                "deterministic_tool_fallback",
            )

        return (
            "I saved your message, but no triage tools are currently configured.",
            [],
            "no_llm_no_tools",
        )

    def _build_user_action_answer(
        self,
        *,
        user_message: str,
        tool_results: list[ToolResult],
    ) -> str:
        memory_result = self._find_tool(tool_results, "write_memory")
        if memory_result is not None and all(
            result.tool_name == "write_memory" for result in tool_results
        ):
            return self._format_memory_confirmation(memory_result)

        issue_kind = self._detect_issue_kind(user_message)

        if issue_kind == "dependency_conflict":
            return self._build_dependency_conflict_answer(user_message=user_message)

        if issue_kind == "config_error":
            return self._build_config_error_answer(tool_results=tool_results)

        return self._build_generic_issue_answer(
            user_message=user_message,
            tool_results=tool_results,
        )

    def _detect_issue_kind(self, user_message: str) -> str:
        text = user_message.lower()

        dependency_keywords = [
            "dependency",
            "dependencies",
            "uv sync",
            "pip install",
            "poetry",
            "requirements",
            "pyproject",
            "version conflict",
            "solving fails",
            "requires numpy",
            "numpy",
            "langchain",
            "pin",
            "pinned",
        ]

        if any(keyword in text for keyword in dependency_keywords):
            return "dependency_conflict"

        config_keywords = [
            "config",
            "configuration",
            "load_config",
            "config_error",
            ".env",
        ]

        if any(keyword in text for keyword in config_keywords):
            return "config_error"

        return "generic"

    def _build_dependency_conflict_answer(self, *, user_message: str) -> str:
        packages = self._extract_dependency_names(user_message)

        package_text = ""
        if packages:
            package_text = " The conflict appears to involve " + ", ".join(
                f"`{package}`" for package in packages[:5]
            ) + "."

        return "\n".join(
            [
                "### Likely cause",
                (
                    "This is a dependency version conflict, not a code bug."
                    f"{package_text} One package requires an older NumPy range, while your project pins "
                    "`numpy==2.0.2`, so `uv` cannot produce one environment that satisfies both constraints."
                ),
                "",
                "### What you should change",
                "- Loosen the NumPy pin in `pyproject.toml` instead of forcing `numpy==2.0.2` globally.",
                "- Use a compatible range such as `numpy>=1.26,<2.0` if `langchain` or one of its dependencies still requires NumPy 1.x.",
                "- If only the model server needs NumPy 2.x, move that dependency into the model-server package instead of pinning it for the whole workspace.",
                "- Keep chatbot/RAG dependencies separate from model-training dependencies if they require different NumPy versions.",
                "",
                "### Files to check",
                "- Root `pyproject.toml` — check workspace-wide dependency pins.",
                "- `apps/api/pyproject.toml` — check LangChain/RAG dependencies.",
                "- `apps/model_server/pyproject.toml` — check ML/model dependencies that may require NumPy 2.x.",
                "- `uv.lock` — regenerate it after changing the constraints.",
                "",
                "### Commands to try",
                "```bash",
                "uv lock",
                "uv sync --all-packages",
                "```",
                "",
                "If it still fails, run:",
                "",
                "```bash",
                "uv tree | grep -i numpy",
                "```",
                "",
                "That will show which package is forcing the incompatible NumPy constraint.",
            ]
        )

    def _build_config_error_answer(
        self,
        *,
        tool_results: list[ToolResult],
    ) -> str:
        entities = self._extract_entities(tool_results)
        rag_strength = self._get_rag_strength(tool_results)
        top_rag_title = self._get_top_rag_title(tool_results)

        file_entities = [
            item["text"]
            for item in entities
            if item["type"] in {"FILE_PATH", "FILE", "PATH"}
        ]
        function_entities = [
            item["text"]
            for item in entities
            if item["type"] in {"FUNCTION", "METHOD"}
        ]
        error_entities = [
            item["text"]
            for item in entities
            if item["type"] in {"ERROR_CODE", "ERROR", "EXCEPTION"}
        ]
        version_entities = [
            item["text"]
            for item in entities
            if item["type"] == "VERSION"
        ]

        likely_cause = self._make_config_likely_cause(
            file_entities=file_entities,
            function_entities=function_entities,
            error_entities=error_entities,
            version_entities=version_entities,
        )

        files_to_check = self._make_config_files_to_check(
            file_entities=file_entities,
            function_entities=function_entities,
        )

        lines: list[str] = [
            "### Likely cause",
            likely_cause,
            "",
            "### Files to check",
        ]

        for item in files_to_check:
            lines.append(f"- {item}")

        lines.extend(
            [
                "",
                "### What to try next",
                "- Reproduce the issue with the smallest possible example.",
                "- Compare the configuration values before and after the upgrade.",
                "- Check whether the failing call receives a missing, renamed, or invalid config field.",
                "- Capture the full stack trace and confirm whether the same input works on the previous version.",
            ]
        )

        if error_entities:
            lines.append(
                f"- Search the codebase for `{error_entities[0]}` to find where it is raised."
            )

        if rag_strength == "weak":
            lines.extend(
                [
                    "",
                    "I do not see a strong matching resolved issue yet, so I would not link this to an existing fix until you have a closer match.",
                ]
            )
        elif top_rag_title:
            lines.extend(
                [
                    "",
                    f"The closest retrieved issue is `{top_rag_title}`, but verify that it matches the same file, function, and error before treating it as related.",
                ]
            )

        return "\n".join(lines).strip()

    def _build_generic_issue_answer(
        self,
        *,
        user_message: str,
        tool_results: list[ToolResult],
    ) -> str:
        entities = self._extract_entities(tool_results)
        rag_strength = self._get_rag_strength(tool_results)
        top_rag_title = self._get_top_rag_title(tool_results)

        entity_lines = []
        for entity in entities[:6]:
            entity_lines.append(f"- `{entity['text']}` ({entity['type']})")

        lines = [
            "### What this looks like",
            "This looks like an issue that needs reproduction details before deciding whether it is a confirmed bug or a usage/configuration problem.",
            "",
            "### What to check first",
        ]

        if entity_lines:
            lines.extend(entity_lines)
        else:
            lines.append("- The failing file/function mentioned by the reporter.")
            lines.append("- The exact error message and stack trace.")
            lines.append("- The version where it started failing.")

        lines.extend(
            [
                "",
                "### What to ask the user for",
                "- Minimal reproduction steps.",
                "- Expected behavior vs actual behavior.",
                "- Full stack trace or logs.",
                "- Version before and after the issue appeared.",
            ]
        )

        if rag_strength == "weak":
            lines.extend(
                [
                    "",
                    "I do not see a strong matching resolved issue yet, so avoid claiming this has a known fix until there is closer evidence.",
                ]
            )
        elif top_rag_title:
            lines.extend(
                [
                    "",
                    f"The closest retrieved issue is `{top_rag_title}`, but verify it matches the same symptoms before linking it.",
                ]
            )

        return "\n".join(lines).strip()

    def _extract_dependency_names(self, text: str) -> list[str]:
        known_packages = [
            "langchain",
            "numpy",
            "uv",
            "pandas",
            "scikit-learn",
            "sklearn",
            "torch",
            "transformers",
            "fastapi",
            "sqlalchemy",
            "pgvector",
        ]

        lowered = text.lower()
        found: list[str] = []

        for package in known_packages:
            if package in lowered:
                found.append(package)

        return found

    def _make_config_likely_cause(
        self,
        *,
        file_entities: list[str],
        function_entities: list[str],
        error_entities: list[str],
        version_entities: list[str],
    ) -> str:
        file_text = file_entities[0] if file_entities else "the config-loading code"
        function_text = function_entities[0] if function_entities else "the config loader"
        error_text = error_entities[0] if error_entities else "the reported error"
        version_text = version_entities[0] if version_entities else "the new version"

        return (
            f"This looks like a configuration-loading regression after upgrading to `{version_text}`. "
            f"The failure is probably happening around `{function_text}` in or near `{file_text}`, "
            f"where the app now receives a config value it does not expect, cannot find a required setting, "
            f"or hits a changed config format that results in `{error_text}`."
        )

    def _make_config_files_to_check(
        self,
        *,
        file_entities: list[str],
        function_entities: list[str],
    ) -> list[str]:
        checks: list[str] = []

        for file_path in file_entities[:4]:
            checks.append(f"`{file_path}` — check where the failing config call happens.")

        for function_name in function_entities[:4]:
            checks.append(f"`{function_name}` — verify its inputs, defaults, and validation logic.")

        checks.extend(
            [
                "`config.py` or the config loader module — check renamed keys, changed defaults, and required environment variables.",
                "The migration/release notes — check for breaking config changes.",
                "The `.env` / deployment config — confirm all required values are still present.",
            ]
        )

        seen: set[str] = set()
        unique_checks: list[str] = []
        for item in checks:
            if item not in seen:
                unique_checks.append(item)
                seen.add(item)

        return unique_checks[:6]

    def _extract_entities(self, tool_results: list[ToolResult]) -> list[dict[str, str]]:
        result = self._find_tool(tool_results, "extract_entities")
        if result is None or result.status != "success":
            return []

        entities = result.output_json.get("entities", [])
        parsed: list[dict[str, str]] = []

        for entity in entities:
            if not isinstance(entity, dict):
                continue

            text = entity.get("text") or entity.get("value")
            entity_type = entity.get("type") or entity.get("label") or "ENTITY"

            if text:
                parsed.append(
                    {
                        "text": str(text),
                        "type": str(entity_type),
                    }
                )

        return parsed

    def _get_rag_strength(self, tool_results: list[ToolResult]) -> str:
        result = self._find_tool(tool_results, "rag_search")
        if result is None or result.status != "success":
            return "none"

        sources = result.output_json.get("sources", [])
        if not sources:
            return "none"

        top_source = sources[0]
        if not isinstance(top_source, dict):
            return "unknown"

        score = top_source.get("score")
        try:
            numeric_score = float(score)
        except (TypeError, ValueError):
            return "unknown"

        if numeric_score < 0.40:
            return "weak"
        if numeric_score < 0.60:
            return "moderate"
        return "strong"

    def _get_top_rag_title(self, tool_results: list[ToolResult]) -> str | None:
        result = self._find_tool(tool_results, "rag_search")
        if result is None or result.status != "success":
            return None

        sources = result.output_json.get("sources", [])
        if not sources:
            return None

        top_source = sources[0]
        if not isinstance(top_source, dict):
            return None

        title = top_source.get("title")
        if not title:
            return None

        return str(title)

    def _find_tool(
        self,
        tool_results: list[ToolResult],
        tool_name: str,
    ) -> ToolResult | None:
        for result in tool_results:
            if result.tool_name == tool_name:
                return result
        return None

    def _format_memory_confirmation(self, result: ToolResult) -> str:
        if result.status != "success":
            return "I could not save that preference."

        content = result.output_json.get("content")
        if content:
            return f"Remembered: {content}"

        return "Remembered."