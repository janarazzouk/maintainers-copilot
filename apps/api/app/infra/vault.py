from typing import Any
from pathlib import Path
import hvac


class VaultError(RuntimeError):
    pass


class VaultClient:
    def __init__(self, addr: str, token: str) -> None:
        self._client = hvac.Client(url=addr, token=token)

    def check_available(self) -> None:
        try:
            if not self._client.is_authenticated():
                raise VaultError("Vault authentication failed.")
        except Exception as exc:
            raise VaultError("Vault is unreachable or authentication failed.") from exc

    def read_app_secrets(self) -> dict[str, Any]:
        self.check_available()

        try:
            response = self._client.secrets.kv.v2.read_secret_version(
                mount_point="secret",
                path="app",
            )
            return dict(response["data"]["data"])
        except Exception as exc:
            raise VaultError("Failed to read secret/app from Vault.") from exc
        
def resolve_vault_token(settings: Any) -> str:
    token_file = getattr(settings, "vault_token_file", None)

    if token_file:
        path = Path(token_file)
        if path.exists():
            token = path.read_text(encoding="utf-8").strip()
            if token:
                return token

    token = getattr(settings, "vault_token", None)
    if token:
        return str(token)

    return str(settings.vault_dev_root_token_id)