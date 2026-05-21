"""HTTP client adapter for the model_server service."""

from typing import Any

import httpx


class ModelServerError(RuntimeError):
    """Raised when the model server cannot be reached or returns an error."""


class ModelServerClient:
    """Small async HTTP client for model_server."""

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")

    async def classify(self, title: str, body: str) -> dict[str, Any]:
        return await self._post(
            path="/classify",
            payload={"title": title, "body": body},
        )

    async def extract_entities(self, title: str, body: str) -> dict[str, Any]:
        return await self._post(
            path="/ner",
            payload={"title": title, "body": body},
        )

    async def summarize(self, title: str, body: str) -> dict[str, Any]:
        return await self._post(
            path="/summarize",
            payload={"title": title, "body": body},
        )

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base_url}{path}"

        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                return response.json()

        except httpx.HTTPStatusError as exc:
            raise ModelServerError(
                f"Model server returned HTTP {exc.response.status_code} for {path}"
            ) from exc

        except httpx.RequestError as exc:
            raise ModelServerError(
                f"Could not reach model server at {url}"
            ) from exc