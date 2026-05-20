from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    vault_addr: str
    vault_dev_root_token_id: str

    redis_url: str | None = None
    model_server_url: str | None = None
    api_url: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()