from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime configuration for the OCR application."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_name: str = "Handwriting OCR"
    app_version: str = "0.1.0"
    upload_dir: Path = Field(default=BASE_DIR / "data" / "uploads")
    processed_dir: Path = Field(default=BASE_DIR / "data" / "processed")
    feedback_dir: Path = Field(default=BASE_DIR / "data" / "training" / "user_feedback")
    database_url: str = Field(default=f"sqlite:///{BASE_DIR / 'data' / 'ocr_history.db'}")
    trocr_model_name: str = "kazars24/trocr-base-handwritten-ru"
    local_mixed_model_dir: Path = Field(default=BASE_DIR / "models" / "trocr-cyrillic-mixed")
    local_finetuned_model_dir: Path = Field(default=BASE_DIR / "models" / "trocr-cyrillic-finetuned")
    prefer_local_finetuned_model: bool = True
    max_upload_size_mb: int = 10
    recognizer_backend: str = "trocr"
    generation_max_new_tokens: int = 128


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    settings.feedback_dir.mkdir(parents=True, exist_ok=True)
    return settings
