from io import BytesIO
from pathlib import Path
import asyncio

from fastapi import UploadFile
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.ocr.base import BaseRecognizer, RecognitionResult
from app.services.recognition_service import RecognitionService
from app.storage.database import Base


class StubRecognizer(BaseRecognizer):
    def recognize(self, image_path: Path) -> RecognitionResult:
        return RecognitionResult(text="текст", confidence=None, model_name="stub")


def build_image_upload() -> UploadFile:
    image_bytes = BytesIO()
    Image.new("RGB", (40, 20), "white").save(image_bytes, format="PNG")
    image_bytes.seek(0)
    return UploadFile(filename="sample.png", file=image_bytes)


def test_recognition_can_skip_preprocessing(tmp_path: Path) -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(bind=engine)

    with Session(engine) as db:
        service = RecognitionService(StubRecognizer())
        service.settings.upload_dir = tmp_path / "uploads"
        service.settings.processed_dir = tmp_path / "processed"
        service.settings.upload_dir.mkdir()
        service.settings.processed_dir.mkdir()

        record = asyncio.run(
            service.recognize_upload(build_image_upload(), db, preprocessing_applied=False)
        )

    assert record.recognized_text == "текст"
    assert record.preprocessing_applied is False
    assert record.processed_image_path is None
