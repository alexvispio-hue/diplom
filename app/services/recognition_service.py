import shutil
import time
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.image_processing.preprocessing import ImagePreprocessor
from app.ocr.base import BaseRecognizer
from app.storage.models import RecognitionRecord


class RecognitionService:
    def __init__(self, recognizer: BaseRecognizer, preprocessor: ImagePreprocessor | None = None) -> None:
        self.settings = get_settings()
        self.recognizer = recognizer
        self.preprocessor = preprocessor or ImagePreprocessor()

    async def recognize_upload(
        self,
        upload: UploadFile,
        db: Session,
        preprocessing_applied: bool = True,
    ) -> RecognitionRecord:
        source_path = self._build_upload_path(upload.filename or "image.png")
        processed_path = self.settings.processed_dir / f"{source_path.stem}.png"

        file_size_bytes = 0
        with source_path.open("wb") as target:
            shutil.copyfileobj(upload.file, target)
            file_size_bytes = target.tell()

        max_size_bytes = self.settings.max_upload_size_mb * 1024 * 1024
        if file_size_bytes > max_size_bytes:
            source_path.unlink(missing_ok=True)
            raise ValueError(f"Размер файла превышает {self.settings.max_upload_size_mb} МБ.")

        start = time.perf_counter()
        recognition_path = source_path
        saved_processed_path = None
        if preprocessing_applied:
            self.preprocessor.preprocess(source_path, processed_path)
            recognition_path = processed_path
            saved_processed_path = str(processed_path)

        result = self.recognizer.recognize(recognition_path)
        elapsed_ms = int((time.perf_counter() - start) * 1000)

        record = RecognitionRecord(
            original_filename=upload.filename or source_path.name,
            file_size_bytes=file_size_bytes,
            stored_image_path=str(source_path),
            processed_image_path=saved_processed_path,
            preprocessing_applied=preprocessing_applied,
            recognized_text=result.text,
            confidence=result.confidence,
            model_name=result.model_name,
            processing_time_ms=elapsed_ms,
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record

    def _build_upload_path(self, filename: str) -> Path:
        suffix = Path(filename).suffix.lower() or ".png"
        safe_name = f"{uuid4().hex}{suffix}"
        return self.settings.upload_dir / safe_name
