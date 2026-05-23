from app.core.config import get_settings
from app.ocr.base import BaseRecognizer
from app.ocr.trocr_recognizer import TrOCRRecognizer


def build_recognizer() -> BaseRecognizer:
    settings = get_settings()
    if settings.recognizer_backend == "trocr":
        return TrOCRRecognizer()
    raise ValueError(f"Unsupported recognizer backend: {settings.recognizer_backend}")
