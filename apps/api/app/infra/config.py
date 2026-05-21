from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    vault_addr: str = Field(
        default="http://127.0.0.1:8200",
        alias="VAULT_ADDR",
    )
    vault_dev_root_token_id: str = Field(
        default="root",
        alias="VAULT_DEV_ROOT_TOKEN_ID",
    )

    redis_url: str | None = None
    api_url: str | None = None

   

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )
    model_server_url: str = Field(
        default="http://127.0.0.1:8001",
        alias="MODEL_SERVER_URL",
    )
    


@lru_cache
def get_settings() -> Settings:
    return Settings()