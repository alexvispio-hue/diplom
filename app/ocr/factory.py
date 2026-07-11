from app.core.config import get_settings
from app.ocr.base import BaseRecognizer
from app.ocr.trocr_recognizer import TrOCRRecognizer
from app.ocr.qwen_vl_recognizer import QwenVLRecognizer


def build_recognizer(backend: str | None = None, model_name: str | None = None) -> BaseRecognizer:
    settings = get_settings()
    selected_backend = backend or settings.recognizer_backend
    if selected_backend == "trocr":
        if model_name:
            return TrOCRRecognizer(model_name=model_name)
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
        return QwenVLRecognizer(model_name=model_name)
    raise ValueError(f"Unsupported recognizer backend: {selected_backend}")
