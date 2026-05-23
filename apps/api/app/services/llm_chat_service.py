from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.infra.groq_tool_client import GroqToolCallingClient, GroqToolCallingError
from app.services.chat_tool_definitions import CHAT_TOOLS
from app.services.tool_service import ToolResult, ToolService


class LLMChatError(RuntimeError):
    pass


class LLMChatService:
    """One Groq LLM that chooses local backend tools."""

    def __init__(
        self,
        *,
        groq_client: GroqToolCallingClient,
        tool_service: ToolService,
    ) -> None:
        self.groq_client = groq_client
        self.tool_service = tool_service
        self.system_prompt = self._load_system_prompt()

    async def answer_with_tools(
        self,
        *,
        user_message: str,
        repo: str | None,
        recent_messages: list[dict[str, Any]],
        long_term_memories: list[dict[str, Any]],
    ) -> tuple[str, list[ToolResult]]:
        messages = self._build_initial_messages(
            user_message=user_message,
            repo=repo,
            recent_messages=recent_messages,
            long_term_memories=long_term_memories,
        )

        try:
            first_response = await self.groq_client.create_chat_completion(
                messages=messages,
                tools=CHAT_TOOLS,
                tool_choice="auto",
            )
        except GroqToolCallingError as exc:
            raise LLMChatError(str(exc)) from exc

        assistant_message = self._extract_assistant_message(first_response)
        tool_calls = assistant_message.get("tool_calls") or []

        if not tool_calls:
            content = assistant_message.get("content") or ""
            if content.strip():
                return content.strip(), []
            return "I could not generate a response for this request.", []

        messages.append(assistant_message)

        tool_results: list[ToolResult] = []

        for tool_call in tool_calls[:4]:
            function = tool_call.get("function") or {}
            tool_name = function.get("name")
            raw_arguments = function.get("arguments") or "{}"

            if not tool_name:
                continue

            arguments = self._parse_tool_arguments(raw_arguments)

            result = await self.tool_service.execute_chat_tool(
                tool_name=str(tool_name),
                arguments=arguments,
                fallback_message=user_message,
                repo=repo,
            )
            tool_results.append(result)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id"),
                    "name": str(tool_name),
                    "content": self._tool_result_to_content(result),
                }
            )

        messages.append(
            {
                "role": "system",
                "content": self._build_final_answer_policy(tool_results),
            }
        )

        try:
            final_response = await self.groq_client.create_chat_completion(
                messages=messages,
            )
        except GroqToolCallingError as exc:
            raise LLMChatError(str(exc)) from exc

        final_message = self._extract_assistant_message(final_response)
        answer = final_message.get("content") or ""

        if not answer.strip():
            answer = self._fallback_answer_from_tools(tool_results)

        return answer.strip(), tool_results

    def _build_initial_messages(
        self,
        *,
        user_message: str,
        repo: str | None,
        recent_messages: list[dict[str, Any]],
        long_term_memories: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": self.system_prompt,
            }
        ]

        if long_term_memories:
            memory_lines = [
                "Relevant long-term memory for this authenticated user:",
            ]
            for memory in long_term_memories:
                memory_lines.append(
                    f"- [{memory.get('memory_type')}] {memory.get('content')}"
                )

            messages.append(
                {
                    "role": "system",
                    "content": "\n".join(memory_lines),
                }
            )

        for item in recent_messages[-8:]:
            role = item.get("role")
            content = item.get("content")
            if role in {"user", "assistant"} and isinstance(content, str):
                messages.append({"role": role, "content": content})

        repo_context = f"\n\nRepo context: {repo}" if repo else ""
        messages.append(
            {
                "role": "user",
                "content": f"{user_message}{repo_context}",
            }
        )

        return messages

    def _build_final_answer_policy(self, tool_results: list[ToolResult]) -> str:
        issue_type = self._extract_classifier_label(tool_results)
        entities = self._extract_entities(tool_results)
        rag_strength = self._assess_rag_strength(tool_results)
        rag_top_sources = self._extract_rag_top_sources(tool_results)

        lines = [
            "Write the final answer as a normal helpful maintainer response.",
            "",
            "Important:",
            "- Do not mention tool calls, classifier, NER, RAG, scores, backend fallback, or LLM/provider errors.",
            "- Do not format the answer as a backend/debug report.",
            "- Tell the user what they should do next.",
            "- Keep it concise and practical.",
            "",
            "Use this answer shape:",
            "### Likely cause",
            "Explain the likely cause in plain language.",
            "",
            "### Files to check",
            "List likely files/functions from the extracted entities.",
            "",
            "### What to try next",
            "Give concrete next steps.",
            "",
        ]

        if issue_type:
            lines.append(
                f"Internal context only: the classifier predicted `{issue_type}`. Do not expose this as a classifier result."
            )

        if entities:
            lines.append("")
            lines.append("Internal extracted entities:")
            for entity in entities[:8]:
                lines.append(f"- {entity}")

        lines.append("")
        lines.append(f"Internal retrieval evidence strength: {rag_strength}.")

        if rag_top_sources:
            lines.append("Internal retrieved sources:")
            for source in rag_top_sources[:5]:
                title = source.get("title") or "unknown title"
                score = source.get("score")
                lines.append(f"- {title} | score={score}")

        if rag_strength in {"weak", "none"}:
            lines.append(
                "Because evidence is weak, say: I do not see a strong matching resolved issue yet."
            )
            lines.append(
                "Do not claim the retrieved issue is a confirmed match."
            )

        return "\n".join(lines)

    def _extract_classifier_label(self, tool_results: list[ToolResult]) -> str | None:
        result = self._find_tool(tool_results, "classify_issue")
        if result is None or result.status != "success":
            return None

        output = result.output_json
        label = (
            output.get("label")
            or output.get("predicted_label")
            or output.get("class_name")
            or output.get("prediction")
        )

        if label is None:
            return None

        return str(label)

    def _extract_entities(self, tool_results: list[ToolResult]) -> list[str]:
        result = self._find_tool(tool_results, "extract_entities")
        if result is None or result.status != "success":
            return []

        output = result.output_json
        entities = output.get("entities", [])

        formatted: list[str] = []
        for entity in entities:
            if isinstance(entity, dict):
                text = entity.get("text") or entity.get("value")
                entity_type = entity.get("type") or entity.get("label")
                if text and entity_type:
                    formatted.append(f"{text} ({entity_type})")
                elif text:
                    formatted.append(str(text))
            else:
                formatted.append(str(entity))

        return formatted

    def _assess_rag_strength(self, tool_results: list[ToolResult]) -> str:
        result = self._find_tool(tool_results, "rag_search")
        if result is None or result.status != "success":
            return "none"

        sources = self._get_rag_sources(result.output_json)
        if not sources:
            return "none"

        scores: list[float] = []
        for source in sources[:5]:
            if not isinstance(source, dict):
                continue
            raw_score = source.get("score")
            try:
                if raw_score is not None:
                    scores.append(float(raw_score))
            except (TypeError, ValueError):
                continue

        if not scores:
            return "unknown"

        top_score = max(scores)

        if top_score < 0.40:
            return "weak"
        if top_score < 0.60:
            return "moderate"
        return "strong"

    def _extract_rag_top_sources(
        self,
        tool_results: list[ToolResult],
    ) -> list[dict[str, Any]]:
        result = self._find_tool(tool_results, "rag_search")
        if result is None or result.status != "success":
            return []

        sources = self._get_rag_sources(result.output_json)

        normalized: list[dict[str, Any]] = []
        for source in sources[:5]:
            if not isinstance(source, dict):
                continue

            normalized.append(
                {
                    "title": source.get("title") or source.get("source_title"),
                    "score": source.get("score"),
                    "source_id": source.get("source_id") or source.get("issue_number"),
                }
            )

        return normalized

    def _get_rag_sources(self, output: dict[str, Any]) -> list[Any]:
        sources = (
            output.get("sources")
            or output.get("retrieved_chunks")
            or output.get("chunks")
            or output.get("results")
            or []
        )

        if isinstance(sources, list):
            return sources

        return []

    def _find_tool(
        self,
        tool_results: list[ToolResult],
        tool_name: str,
    ) -> ToolResult | None:
        for result in tool_results:
            if result.tool_name == tool_name:
                return result
        return None

    def _extract_assistant_message(self, response: dict[str, Any]) -> dict[str, Any]:
        try:
            message = response["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMChatError(f"Unexpected Groq response shape: {response}") from exc

        if not isinstance(message, dict):
            raise LLMChatError(f"Unexpected Groq message shape: {message}")

        return message

    def _parse_tool_arguments(self, raw_arguments: str) -> dict[str, Any]:
        try:
            parsed = json.loads(raw_arguments)
        except json.JSONDecodeError:
            return {}

        if not isinstance(parsed, dict):
            return {}

        return parsed

    def _tool_result_to_content(self, result: ToolResult) -> str:
        payload = {
            "tool_name": result.tool_name,
            "status": result.status,
            "error_message": result.error_message,
            "latency_ms": result.latency_ms,
            "input": result.input_json,
            "output": result.output_json,
        }

        text = json.dumps(payload, ensure_ascii=False, default=str)

        if len(text) > 12000:
            text = text[:12000] + "...[truncated]"

        return text

    def _fallback_answer_from_tools(self, tool_results: list[ToolResult]) -> str:
        if not tool_results:
            return "I could not find enough information to triage this issue."

        return "I found some issue details, but I could not generate a full answer. Ask for the exact error, stack trace, affected version, and a minimal reproduction."

    def _load_system_prompt(self) -> str:
        prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "chat_system.md"
        return prompt_path.read_text(encoding="utf-8")