from __future__ import annotations

import asyncio
import os
import re
from typing import Any

import httpx


class GroqToolCallingError(RuntimeError):
    pass


class GroqToolCallingClient:
    """Groq OpenAI-compatible chat client for local tool calling.

    The chatbot should not request huge outputs, because Groq free/on-demand
    tiers can rate-limit by tokens-per-minute. Keep this client intentionally
    smaller than general generation.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature

        # Important:
        # Your old request asked for ~4100 tokens, which triggered Groq 429.
        # This clamps chatbot responses to a safer demo-friendly size.
        default_chat_max_tokens = int(os.getenv("GROQ_CHAT_MAX_TOKENS", "900"))
        self.max_tokens = min(max_tokens, default_chat_max_tokens)

        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    async def create_chat_completion(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if tools is not None:
            payload["tools"] = tools

        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        url = f"{self.base_url}/chat/completions"

        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                    response = await client.post(url, headers=headers, json=payload)

                if response.status_code == 429:
                    last_error = GroqToolCallingError(
                        "Groq rate limit reached. Please retry in a moment."
                    )

                    if attempt < self.max_retries:
                        await asyncio.sleep(self._retry_delay_seconds(response))
                        continue

                    raise last_error

                response.raise_for_status()
                return response.json()

            except httpx.HTTPStatusError as exc:
                # Do not expose raw Groq response body to the user.
                raise GroqToolCallingError(
                    f"Groq API returned HTTP {exc.response.status_code}."
                ) from exc

            except httpx.HTTPError as exc:
                last_error = exc

                if attempt < self.max_retries:
                    await asyncio.sleep(0.5)
                    continue

                raise GroqToolCallingError(
                    "Groq API request failed. Please retry in a moment."
                ) from exc

        raise GroqToolCallingError(str(last_error) if last_error else "Groq request failed.")

    def _retry_delay_seconds(self, response: httpx.Response) -> float:
        retry_after = response.headers.get("retry-after")
        if retry_after:
            try:
                return min(float(retry_after), 3.0)
            except ValueError:
                pass

        # Groq sometimes says: "Please try again in 740ms"
        match = re.search(r"try again in (\d+)ms", response.text)
        if match:
            return min(int(match.group(1)) / 1000.0, 3.0)

        return 1.0