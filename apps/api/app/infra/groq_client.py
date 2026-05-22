from dataclasses import dataclass
from typing import Any

import httpx


class GroqLLMError(RuntimeError):
    """Raised when Groq cannot generate an answer."""


@dataclass(frozen=True)
class GroqLLMResponse:
    content: str
    model: str
    usage: dict[str, Any]


class GroqLLMClient:
    """Small Groq chat-completions client.

    Uses Groq's OpenAI-compatible endpoint:
    /chat/completions
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate_rag_answer(
        self,
        question: str,
        context_blocks: list[str],
    ) -> GroqLLMResponse:
        if not context_blocks:
            return GroqLLMResponse(
                content="I could not find enough retrieved context to answer this question.",
                model=self.model,
                usage={},
            )

        system_prompt = (
            "You are Maintainer's Copilot, an assistant for open-source maintainers. "
            "Answer using only the provided retrieved GitHub issue context. "
            "Treat Source 1 as the primary issue unless another source clearly has the same exact error, API, or issue topic. "
            "Do not combine resolutions from unrelated issues. "
            "Do not invent fixes, PRs, versions, or code changes. "
            "If the retrieved context does not contain the final fix, say that the issue was linked to a fix but the provided context does not show the exact patch. "
            "Prefer a concise answer with: problem, maintainer context, resolution, and source URL."
        )

        user_prompt = (
            f"Question:\n{question}\n\n"
            "Retrieved context:\n"
            + "\n\n---\n\n".join(context_blocks)
            + "\n\nWrite the final answer for the maintainer."
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        url = f"{self.base_url}/chat/completions"

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

        except httpx.HTTPStatusError as exc:
            raise GroqLLMError(
                f"Groq returned HTTP {exc.response.status_code}: {exc.response.text}"
            ) from exc

        except httpx.RequestError as exc:
            raise GroqLLMError(
                f"Could not reach Groq at {url}"
            ) from exc

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise GroqLLMError("Groq response did not contain an answer.") from exc

        return GroqLLMResponse(
            content=content,
            model=str(data.get("model") or self.model),
            usage=dict(data.get("usage") or {}),
        )