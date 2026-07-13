from pathlib import Path

from app.core.config import get_settings
from app.ocr.base import BaseRecognizer
from app.ocr.trocr_recognizer import TrOCRRecognizer
from app.ocr.qwen_vl_recognizer import QwenVLRecognizer


def build_recognizer(backend: str | None = None, model_name: str | None = None) -> BaseRecognizer:
    settings = get_settings()
    selected_backend = backend or settings.recognizer_backend
    production_model = None
    try:
        from app.services.model_registry_service import ModelRegistryService

        production_model = ModelRegistryService().production_model(selected_backend)
    except (OSError, ValueError):
        production_model = None
    if selected_backend == "trocr":
        if model_name:
            return TrOCRRecognizer(model_name=model_name)
        if production_model and production_model.get("local_path"):
            return TrOCRRecognizer(model_name=production_model["local_path"])
        if settings.prefer_local_finetuned_model:
            for model_dir in (
                settings.local_mixed_epoch_2_model_dir,
                settings.local_mixed_model_dir,
                settings.local_finetuned_model_dir,
            ):
                if model_dir.exists():
                    return TrOCRRecognizer(model_name=str(model_dir))
        return TrOCRRecognizer()
    if selected_backend == "qwen":
        if production_model:
            local_path = production_model.get("local_path")
            is_adapter = bool(production_model.get("adapter"))
            return QwenVLRecognizer(
                model_name=model_name
                or (production_model.get("base_model") if is_adapter else local_path)
                or production_model.get("base_model"),
                adapter_dir=Path(local_path) if local_path and is_adapter else None,
            )
        return QwenVLRecognizer(model_name=model_name)
    raise ValueError(f"Unsupported recognizer backend: {selected_backend}")
