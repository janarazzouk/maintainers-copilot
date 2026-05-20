"""Vault adapter for model-server secret loading."""

from typing import Any

import hvac


class VaultError(RuntimeError):
    """Raised when Vault secrets cannot be loaded."""


class VaultClient:
    """Small wrapper around HashiCorp Vault KV v2."""

    def __init__(self, addr: str, token: str) -> None:
        self._client = hvac.Client(url=addr, token=token)

    def check_available(self) -> None:
        if not self._client.is_authenticated():
            raise VaultError("Vault is unreachable or token is invalid.")

    def read_app_secrets(self, path: str = "app") -> dict[str, str]:
        self.check_available()

        try:
            response: dict[str, Any] = self._client.secrets.kv.v2.read_secret_version(
                path=path,
                mount_point="secret",
            )
        except Exception as exc:
            raise VaultError(f"Failed to read Vault secret/app: {exc}") from exc

        data = response.get("data", {}).get("data", {})

        if not isinstance(data, dict):
            raise VaultError("Vault secret/app did not return a dictionary.")

        return {str(key): str(value) for key, value in data.items()}