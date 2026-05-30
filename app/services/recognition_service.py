import json
import shutil
import time
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.image_processing.page_segmentation import PageLineSegmenter
from app.image_processing.preprocessing import ImagePreprocessor
from app.ocr.base import BaseRecognizer
from app.storage.models import RecognitionRecord


class RecognitionService:
    def __init__(
        self,
        recognizer: BaseRecognizer,
        preprocessor: ImagePreprocessor | None = None,
        page_segmenter: PageLineSegmenter | None = None,
    ) -> None:
        self.settings = get_settings()
        self.recognizer = recognizer
        self.preprocessor = preprocessor or ImagePreprocessor()
        self.page_segmenter = page_segmenter or PageLineSegmenter()

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

    async def recognize_page_upload(self, upload: UploadFile, db: Session) -> RecognitionRecord:
        source_path = self._build_upload_path(upload.filename or "image.png")
        line_dir = self.settings.processed_dir / f"{source_path.stem}_lines"
        annotated_path = self.settings.processed_dir / f"{source_path.stem}_annotated.png"

        with source_path.open("wb") as target:
            shutil.copyfileobj(upload.file, target)
            file_size_bytes = target.tell()

        max_size_bytes = self.settings.max_upload_size_mb * 1024 * 1024
        if file_size_bytes > max_size_bytes:
            source_path.unlink(missing_ok=True)
            raise ValueError(f"Размер файла превышает {self.settings.max_upload_size_mb} МБ.")

        start = time.perf_counter()
        segmentation = self.page_segmenter.segment(source_path, line_dir, annotated_path)
        line_results = [self.recognizer.recognize(line.image_path) for line in segmentation.lines]
        elapsed_ms = int((time.perf_counter() - start) * 1000)

        record = RecognitionRecord(
            original_filename=upload.filename or source_path.name,
            file_size_bytes=file_size_bytes,
            stored_image_path=str(source_path),
            processed_image_path=str(segmentation.annotated_image_path),
            preprocessing_applied=True,
            recognition_mode="page",
            line_count=len(segmentation.lines),
            line_image_paths_json=json.dumps([str(line.image_path) for line in segmentation.lines]),
            recognized_text="\n".join(result.text.strip() for result in line_results if result.text.strip()),
            confidence=self._average_confidence([result.confidence for result in line_results]),
            model_name=line_results[0].model_name,
            processing_time_ms=elapsed_ms,
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record

    @staticmethod
    def _average_confidence(confidences: list[float | None]) -> float | None:
        known_confidences = [confidence for confidence in confidences if confidence is not None]
        if not known_confidences:
            return None
        return sum(known_confidences) / len(known_confidences)

    def _build_upload_path(self, filename: str) -> Path:
        suffix = Path(filename).suffix.lower() or ".png"
        safe_name = f"{uuid4().hex}{suffix}"
        return self.settings.upload_dir / safe_name
