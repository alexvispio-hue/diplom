from app.core.config import get_settings
from app.ocr.base import BaseRecognizer
from app.ocr.trocr_recognizer import TrOCRRecognizer


def build_recognizer() -> BaseRecognizer:
    settings = get_settings()
    if settings.recognizer_backend == "trocr":
        if settings.prefer_local_finetuned_model:
            for model_dir in (settings.local_mixed_model_dir, settings.local_finetuned_model_dir):
                if model_dir.exists():
                    return TrOCRRecognizer(model_name=str(model_dir))
        return TrOCRRecognizer()
    raise ValueError(f"Unsupported recognizer backend: {settings.recognizer_backend}")
