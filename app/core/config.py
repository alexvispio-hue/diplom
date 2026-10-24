from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[2]
QWEN_MODEL_PRESETS = {
    "main": "Qwen/Qwen3-VL-2B-Instruct",
    "light": "Qwen/Qwen3-VL-2B-Instruct",
    "experimental_4b": "Qwen/Qwen3-VL-4B-Instruct",
    "research_8b": "Qwen/Qwen3-VL-8B-Instruct",
    "fallback_7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "fallback_3b": "Qwen/Qwen2.5-VL-3B-Instruct",
}


class Settings(BaseSettings):
    """Runtime configuration for the OCR application."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_name: str = "Handwriting OCR"
    app_version: str = "0.4.0-dev"
    upload_dir: Path = Field(default=BASE_DIR / "data" / "uploads")
    processed_dir: Path = Field(default=BASE_DIR / "data" / "processed")
    feedback_dir: Path = Field(default=BASE_DIR / "data" / "training" / "user_feedback")
    jobs_dir: Path = Field(default=BASE_DIR / "data" / "runtime" / "jobs")
    exports_dir: Path = Field(default=BASE_DIR / "data" / "runtime" / "exports")
    logs_dir: Path = Field(default=BASE_DIR / "data" / "runtime" / "logs")
    user_dictionary_path: Path = Field(default=BASE_DIR / "data" / "runtime" / "user_dictionary.json")
    models_dir: Path = Field(default=BASE_DIR / "models")
    model_registry_path: Path = Field(default=BASE_DIR / "data" / "runtime" / "model_registry.json")
    model_download_state_path: Path = Field(default=BASE_DIR / "data" / "runtime" / "model_downloads.json")
    hf_home: Path = Field(default=BASE_DIR / "data" / "runtime" / "huggingface")
    database_url: str = Field(default=f"sqlite:///{BASE_DIR / 'data' / 'ocr_history.db'}")
    trocr_model_name: str = "kazars24/trocr-base-handwritten-ru"
    local_mixed_epoch_2_model_dir: Path = Field(default=BASE_DIR / "models" / "trocr-cyrillic-mixed-epoch-2")
    local_mixed_model_dir: Path = Field(default=BASE_DIR / "models" / "trocr-cyrillic-mixed")
    local_finetuned_model_dir: Path = Field(default=BASE_DIR / "models" / "trocr-cyrillic-finetuned")
    prefer_local_finetuned_model: bool = True
    max_upload_size_mb: int = 10
    max_image_pixels: int = 60_000_000
    max_document_pages: int = 100
    pdf_render_dpi: int = 160
    recognizer_backend: str = "trocr"
    generation_max_new_tokens: int = 128
    qwen_model_preset: str = "light"
    qwen_model_name: str | None = None
    qwen_lora_adapter_dir: Path | None = None
    qwen_device_map: str = "auto"
    qwen_torch_dtype: str = "auto"

    @property
    def qwen_effective_model_name(self) -> str:
        if self.qwen_model_name:
            return self.qwen_model_name
        if self.qwen_model_preset not in QWEN_MODEL_PRESETS:
            choices = ", ".join(sorted(QWEN_MODEL_PRESETS))
            raise ValueError(f"Unknown Qwen model preset: {self.qwen_model_preset}. Available presets: {choices}.")
        return QWEN_MODEL_PRESETS[self.qwen_model_preset]


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    settings.feedback_dir.mkdir(parents=True, exist_ok=True)
    settings.jobs_dir.mkdir(parents=True, exist_ok=True)
    settings.exports_dir.mkdir(parents=True, exist_ok=True)
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    settings.user_dictionary_path.parent.mkdir(parents=True, exist_ok=True)
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    settings.model_registry_path.parent.mkdir(parents=True, exist_ok=True)
    settings.model_download_state_path.parent.mkdir(parents=True, exist_ok=True)
    settings.hf_home.mkdir(parents=True, exist_ok=True)
    return settings
