from __future__ import annotations

import os
from typing import Any

import requests

#Purpose: one clean client for calling your FastAPI backend.

class ApiError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class APIClient:
    def __init__(self, *, token: str | None = None) -> None:
        self.base_url = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
        self.token = token

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
        }

        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        return headers

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
    ) -> Any:
        url = f"{self.base_url}{path}"

        try:
            response = requests.request(
                method=method,
                url=url,
                headers=self._headers(),
                json=json,
                timeout=60,
            )
        except requests.RequestException as exc:
            raise ApiError(f"Could not connect to API at {self.base_url}: {exc}") from exc

        if response.status_code == 204:
            return None

        try:
            payload = response.json()
        except ValueError:
            payload = {"detail": response.text}

        if response.status_code >= 400:
            message = self._extract_error_message(payload)
            raise ApiError(message, status_code=response.status_code)

        return payload

    def _extract_error_message(self, payload: Any) -> str:
        if isinstance(payload, dict):
            detail = payload.get("detail")

            if isinstance(detail, dict):
                return str(detail.get("message") or detail.get("code") or detail)

            if isinstance(detail, str):
                return detail

            return str(payload)

        return str(payload)

    def login(self, *, email: str, password: str) -> dict[str, Any]:
        return self._request(
            "POST",
            "/auth/login",
            json={
                "email": email,
                "password": password,
            },
        )

    def me(self) -> dict[str, Any]:
        return self._request("GET", "/auth/me")

    def send_chat_message(
        self,
        *,
        message: str,
        repo: str | None = None,
        conversation_id: int | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "message": message,
            "repo": repo,
        }

        if conversation_id is not None:
            payload["conversation_id"] = conversation_id

        return self._request(
            "POST",
            "/chat/message",
            json=payload,
        )

    def list_conversations(self) -> list[dict[str, Any]]:
        return self._request("GET", "/chat/conversations")

    def list_messages(self, *, conversation_id: int) -> list[dict[str, Any]]:
        return self._request(
            "GET",
            f"/chat/conversations/{conversation_id}/messages",
        )

    def delete_conversation(self, *, conversation_id: int) -> None:
        self._request(
            "DELETE",
            f"/chat/conversations/{conversation_id}",
        )

    def list_memories(self) -> list[dict[str, Any]]:
        return self._request("GET", "/memory")

    def create_memory(
        self,
        *,
        memory_type: str,
        content: str,
        reason: str | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/memory",
            json={
                "memory_type": memory_type,
                "content": content,
                "reason": reason,
            },
        )

    def search_memory(
        self,
        *,
        query: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        return self._request(
            "POST",
            "/memory/search",
            json={
                "query": query,
                "limit": limit,
            },
        )

    def delete_memory(self, *, memory_id: int) -> None:
        self._request("DELETE", f"/memory/{memory_id}")

    def list_widgets(self) -> list[dict[str, Any]]:
        return self._request("GET", "/admin/widgets")

    def create_widget(
        self,
        *,
        name: str,
        allowed_origins: list[str],
        theme: dict[str, Any],
        greeting: str,
        enabled_tools: list[str],
        is_active: bool = True,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/admin/widgets",
            json={
                "name": name,
                "allowed_origins": allowed_origins,
                "theme": theme,
                "greeting": greeting,
                "enabled_tools": enabled_tools,
                "is_active": is_active,
            },
        )

    def get_public_widget_config(self, *, widget_id: str) -> dict[str, Any]:
        return self._request("GET", f"/widgets/{widget_id}/config")