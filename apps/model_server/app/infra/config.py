"""Model-server configuration."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-driven model-server settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    model_dir: Path = Field(
        default=Path("artifacts/roberta_issue_classifier"),
        alias="MODEL_SERVER_MODEL_DIR",
    )

    max_length: int = Field(
        default=512,
        alias="MODEL_SERVER_MAX_LENGTH",
    )

    device: str = Field(
        default="auto",
        alias="MODEL_SERVER_DEVICE",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()