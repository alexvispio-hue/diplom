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
from app.ocr.factory import build_recognizer
from app.services.feedback_service import FeedbackService
from app.storage.models import RecognitionRecord


PAGE_MODE_TROCR_LINES = "trocr_lines"
PAGE_MODE_QWEN_LINES = "qwen_lines"
PAGE_MODE_QWEN_PAGE = "qwen_page"
PAGE_MODE_QWEN_LAYOUT = "qwen_layout"
PAGE_MODE_COMPARE = "compare"
PAGE_MODE_HYBRID = "hybrid"
SUPPORTED_PAGE_MODES = {
    PAGE_MODE_TROCR_LINES,
    PAGE_MODE_QWEN_LINES,
    PAGE_MODE_QWEN_PAGE,
    PAGE_MODE_QWEN_LAYOUT,
    PAGE_MODE_COMPARE,
    PAGE_MODE_HYBRID,
}


class RecognitionService:
    def __init__(
        self,
        recognizer: BaseRecognizer,
        preprocessor: ImagePreprocessor | None = None,
        page_segmenter: PageLineSegmenter | None = None,
    ) -> None:
        self.settings = get_settings()
        self.recognizer = recognizer
        self._recognizers: dict[str, BaseRecognizer] = {"trocr": recognizer}
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

    async def recognize_page_upload(
        self,
        upload: UploadFile,
        db: Session,
        mode: str = PAGE_MODE_TROCR_LINES,
        preprocessing_applied: bool = True,
        model_name: str | None = None,
        max_tokens: int | None = None,
        save_intermediate_images: bool = True,
        return_coordinates: bool = True,
    ) -> RecognitionRecord:
        if mode not in SUPPORTED_PAGE_MODES:
            raise ValueError(f"Неподдерживаемый режим распознавания страницы: {mode}.")

        source_path = self._build_upload_path(upload.filename or "image.png")
        line_dir = self.settings.processed_dir / f"{source_path.stem}_lines"
        annotated_path = self.settings.processed_dir / f"{source_path.stem}_annotated.png"
        preprocessed_path = self.settings.processed_dir / f"{source_path.stem}_page.png"

        with source_path.open("wb") as target:
            shutil.copyfileobj(upload.file, target)
            file_size_bytes = target.tell()

        max_size_bytes = self.settings.max_upload_size_mb * 1024 * 1024
        if file_size_bytes > max_size_bytes:
            source_path.unlink(missing_ok=True)
            raise ValueError(f"Размер файла превышает {self.settings.max_upload_size_mb} МБ.")

        start = time.perf_counter()
        page_path = source_path
        if preprocessing_applied:
            self.preprocessor.preprocess(source_path, preprocessed_path)
            page_path = preprocessed_path

        segmentation = None
        line_results = []
        recognized_text = ""
        confidence = None
        model_name = ""
        review_notes: list[dict] = []

        if mode == PAGE_MODE_QWEN_PAGE:
            qwen = self._get_recognizer("qwen", model_name=model_name)
            page_result = self._recognize_full_page(qwen, page_path, max_tokens=max_tokens)
            recognized_text = page_result.text
            confidence = page_result.confidence
            model_name = page_result.model_name
        elif mode == PAGE_MODE_QWEN_LAYOUT:
            qwen = self._get_recognizer("qwen", model_name=model_name)
            layout_result = self._recognize_layout(qwen, page_path, max_tokens=max_tokens)
            recognized_text = layout_result.text
            confidence = layout_result.confidence
            model_name = layout_result.model_name
        else:
            segmentation = self.page_segmenter.segment(page_path, line_dir, annotated_path)
            if mode == PAGE_MODE_TROCR_LINES:
                line_results = [self.recognizer.recognize(line.image_path) for line in segmentation.lines]
                recognized_text = "\n".join(result.text.strip() for result in line_results if result.text.strip())
                confidence = self._average_confidence([result.confidence for result in line_results])
                model_name = line_results[0].model_name if line_results else self.recognizer.model_name
            elif mode == PAGE_MODE_QWEN_LINES:
                qwen = self._get_recognizer("qwen", model_name=model_name)
                line_results = [self._recognize_line(qwen, line.image_path, max_tokens=max_tokens) for line in segmentation.lines]
                recognized_text = "\n".join(result.text.strip() for result in line_results if result.text.strip())
                confidence = self._average_confidence([result.confidence for result in line_results])
                model_name = line_results[0].model_name if line_results else qwen.model_name
            elif mode in {PAGE_MODE_COMPARE, PAGE_MODE_HYBRID}:
                qwen = self._get_recognizer("qwen", model_name=model_name)
                trocr_results = [self.recognizer.recognize(line.image_path) for line in segmentation.lines]
                qwen_results = [self._recognize_line(qwen, line.image_path, max_tokens=max_tokens) for line in segmentation.lines]
                if mode == PAGE_MODE_COMPARE:
                    recognized_text = self._format_comparison(trocr_results, qwen_results)
                else:
                    recognized_text, review_notes = self._build_hybrid_result(trocr_results, qwen_results)
                confidence = self._average_confidence([result.confidence for result in trocr_results + qwen_results])
                trocr_model_name = trocr_results[0].model_name if trocr_results else "trocr"
                qwen_model_name = qwen_results[0].model_name if qwen_results else getattr(qwen, "model_name", "qwen")
                model_name = f"{trocr_model_name} + {qwen_model_name}"

        elapsed_ms = int((time.perf_counter() - start) * 1000)

        record = RecognitionRecord(
            original_filename=upload.filename or source_path.name,
            file_size_bytes=file_size_bytes,
            stored_image_path=str(source_path),
            processed_image_path=self._select_processed_image_path(
                segmentation=segmentation,
                preprocessed_path=preprocessed_path,
                source_path=source_path,
                preprocessing_applied=preprocessing_applied,
                save_intermediate_images=save_intermediate_images,
            ),
            preprocessing_applied=preprocessing_applied,
            recognition_mode=mode,
            line_count=len(segmentation.lines) if segmentation else 0,
            line_image_paths_json=(
                json.dumps([str(line.image_path) for line in segmentation.lines])
                if segmentation and save_intermediate_images
                else None
            ),
            line_bboxes_json=json.dumps([line.bbox for line in segmentation.lines]) if segmentation and return_coordinates else None,
            recognized_text=recognized_text,
            confidence=confidence,
            model_name=model_name,
            needs_review=bool(review_notes),
            review_notes_json=json.dumps(review_notes, ensure_ascii=False) if review_notes else None,
            processing_time_ms=elapsed_ms,
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        if review_notes:
            record = FeedbackService().save(record, "partial", db, corrected_text=None)
        if not save_intermediate_images:
            self._remove_intermediate_paths(line_dir, annotated_path, preprocessed_path, preprocessing_applied)
        return record

    @staticmethod
    def _average_confidence(confidences: list[float | None]) -> float | None:
        known_confidences = [confidence for confidence in confidences if confidence is not None]
        if not known_confidences:
            return None
        return sum(known_confidences) / len(known_confidences)

    def _get_recognizer(self, backend: str, model_name: str | None = None) -> BaseRecognizer:
        cache_key = f"{backend}:{model_name}" if model_name else backend
        if cache_key not in self._recognizers:
            self._recognizers[cache_key] = build_recognizer(backend, model_name=model_name)
        return self._recognizers[cache_key]

    @staticmethod
    def _recognize_line(recognizer: BaseRecognizer, image_path: Path, max_tokens: int | None = None):
        try:
            return recognizer.recognize(image_path, max_new_tokens=max_tokens)
        except TypeError:
            return recognizer.recognize(image_path)

    @staticmethod
    def _recognize_full_page(recognizer: BaseRecognizer, image_path: Path, max_tokens: int | None = None):
        if hasattr(recognizer, "recognize_page"):
            try:
                return recognizer.recognize_page(image_path, max_new_tokens=max_tokens)
            except TypeError:
                return recognizer.recognize_page(image_path)
        return recognizer.recognize(image_path)

    @staticmethod
    def _recognize_layout(recognizer: BaseRecognizer, image_path: Path, max_tokens: int | None = None):
        if hasattr(recognizer, "recognize_layout"):
            try:
                return recognizer.recognize_layout(image_path, max_new_tokens=max_tokens)
            except TypeError:
                return recognizer.recognize_layout(image_path)
        return recognizer.recognize(image_path)

    @staticmethod
    def _select_processed_image_path(
        segmentation,
        preprocessed_path: Path,
        source_path: Path,
        preprocessing_applied: bool,
        save_intermediate_images: bool,
    ) -> str | None:
        if not save_intermediate_images:
            return None
        if segmentation:
            return str(segmentation.annotated_image_path)
        return str(preprocessed_path if preprocessing_applied else source_path)

    @staticmethod
    def _remove_intermediate_paths(
        line_dir: Path,
        annotated_path: Path,
        preprocessed_path: Path,
        preprocessing_applied: bool,
    ) -> None:
        shutil.rmtree(line_dir, ignore_errors=True)
        annotated_path.unlink(missing_ok=True)
        if preprocessing_applied:
            preprocessed_path.unlink(missing_ok=True)

    @staticmethod
    def _format_comparison(trocr_results, qwen_results) -> str:
        blocks = []
        for index, (trocr_result, qwen_result) in enumerate(zip(trocr_results, qwen_results), start=1):
            blocks.append(
                "\n".join(
                    [
                        f"Строка {index}",
                        f"TrOCR: {trocr_result.text.strip()}",
                        f"Qwen: {qwen_result.text.strip()}",
                    ]
                )
            )
        return "\n\n".join(blocks)

    @classmethod
    def _format_hybrid(cls, trocr_results, qwen_results) -> str:
        text, _ = cls._build_hybrid_result(trocr_results, qwen_results)
        return text

    @classmethod
    def _build_hybrid_result(cls, trocr_results, qwen_results) -> tuple[str, list[dict]]:
        lines = []
        review_notes = []
        for index, (trocr_result, qwen_result) in enumerate(zip(trocr_results, qwen_results), start=1):
            trocr_text = trocr_result.text.strip()
            qwen_text = qwen_result.text.strip()
            selected_text = cls._select_hybrid_text(trocr_text, qwen_text, trocr_result.confidence)
            disagreement = cls._relative_distance(trocr_text, qwen_text)
            if disagreement >= 0.35 and trocr_text and qwen_text:
                selected_text = f"{selected_text} [review line {index}: TrOCR != Qwen]"
                review_notes.append(
                    {
                        "line_number": index,
                        "trocr_text": trocr_text,
                        "qwen_text": qwen_text,
                        "disagreement": disagreement,
                        "reason": "TrOCR and Qwen line outputs differ significantly.",
                    }
                )
            if selected_text:
                lines.append(selected_text)
        return "\n".join(lines), review_notes

    @staticmethod
    def _select_hybrid_text(trocr_text: str, qwen_text: str, trocr_confidence: float | None) -> str:
        if not trocr_text:
            return qwen_text
        if not qwen_text:
            return trocr_text
        if trocr_text == qwen_text:
            return trocr_text
        if trocr_confidence is not None and trocr_confidence >= 0.9:
            return trocr_text
        if len(qwen_text) >= len(trocr_text) * 1.15:
            return qwen_text
        return trocr_text

    @staticmethod
    def _relative_distance(left: str, right: str) -> float:
        if left == right:
            return 0.0
        if not left or not right:
            return 1.0

        previous = list(range(len(right) + 1))
        for left_index, left_char in enumerate(left, start=1):
            current = [left_index]
            for right_index, right_char in enumerate(right, start=1):
                substitution_cost = 0 if left_char == right_char else 1
                current.append(
                    min(
                        previous[right_index] + 1,
                        current[right_index - 1] + 1,
                        previous[right_index - 1] + substitution_cost,
                    )
                )
            previous = current
        return previous[-1] / max(len(left), len(right))

    def _build_upload_path(self, filename: str) -> Path:
        suffix = Path(filename).suffix.lower() or ".png"
        safe_name = f"{uuid4().hex}{suffix}"
        return self.settings.upload_dir / safe_name
