"""
Application configuration loaded from environment variables.
Uses pydantic-settings for validation and .env support.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for API, auth, storage, and optional cloud/LLM."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "AI Medical Diagnosis Assistant"
    debug: bool = False
    api_prefix: str = "/api/v1"

    # JWT
    secret_key: str = "change-me-in-production-use-openssl-rand-hex-32"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24

    # Paths (resolved relative to backend cwd)
    database_url: str = "sqlite+aiosqlite:///./data/medical_ai.db"
    upload_dir: Path = Path("data/uploads")
    heatmap_dir: Path = Path("data/heatmaps")
    mask_dir: Path = Path("data/masks")
    ml_checkpoints_dir: Path = Path("../ml_models/checkpoints")

    # Optional OpenAI for LLM explanations / chatbot
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"

    # CORS
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
