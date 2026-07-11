from io import BytesIO
from pathlib import Path
import asyncio

import cv2
from fastapi import UploadFile
import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.image_processing.page_segmentation import PageLineSegmenter
from app.ocr.base import BaseRecognizer, RecognitionResult
from app.services.recognition_service import RecognitionService
from app.storage.database import Base


class StubRecognizer(BaseRecognizer):
    def recognize(self, image_path: Path) -> RecognitionResult:
        return RecognitionResult(text=image_path.stem, confidence=0.8, model_name="stub")


class PageStubRecognizer(BaseRecognizer):
    model_name = "page-stub"

    def recognize(self, image_path: Path) -> RecognitionResult:
        return RecognitionResult(text=f"line:{image_path.stem}", confidence=None, model_name=self.model_name)

    def recognize_page(self, image_path: Path) -> RecognitionResult:
        return RecognitionResult(text="страница целиком", confidence=None, model_name=self.model_name)

    def recognize_layout(self, image_path: Path) -> RecognitionResult:
        return RecognitionResult(
            text='[{"line_number": 1, "text": "first line", "box": [40, 40, 250, 80]}]',
            confidence=None,
            model_name=self.model_name,
        )


def build_page_bytes() -> bytes:
    image = np.full((300, 800, 3), 255, dtype=np.uint8)
    for index, text in enumerate(("first line", "second line", "third line")):
        cv2.putText(image, text, (40, 70 + index * 85), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    success, encoded = cv2.imencode(".png", image)
    assert success
    return encoded.tobytes()


def test_page_segmenter_extracts_text_lines(tmp_path: Path) -> None:
    source_path = tmp_path / "page.png"
    source_path.write_bytes(build_page_bytes())

    result = PageLineSegmenter().segment(
        source_path,
        tmp_path / "lines",
        tmp_path / "annotated.png",
    )

    assert len(result.lines) == 3
    assert result.annotated_image_path.exists()
    assert all(line.image_path.exists() for line in result.lines)


def test_page_recognition_collects_line_results(tmp_path: Path) -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(bind=engine)
    upload = UploadFile(filename="page.png", file=BytesIO(build_page_bytes()))

    with Session(engine) as db:
        service = RecognitionService(StubRecognizer())
        service.settings.upload_dir = tmp_path / "uploads"
        service.settings.processed_dir = tmp_path / "processed"
        service.settings.upload_dir.mkdir()
        service.settings.processed_dir.mkdir()

        record = asyncio.run(service.recognize_page_upload(upload, db))

    assert record.recognition_mode == "trocr_lines"
    assert record.line_count == 3
    assert record.recognized_text == "line_000\nline_001\nline_002"
    assert record.confidence == pytest.approx(0.8)
    assert len(record.line_image_urls) == 3


def test_page_recognition_can_use_qwen_full_page_mode(tmp_path: Path) -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(bind=engine)
    upload = UploadFile(filename="page.png", file=BytesIO(build_page_bytes()))

    with Session(engine) as db:
        service = RecognitionService(StubRecognizer())
        service._recognizers["qwen"] = PageStubRecognizer()
        service.settings.upload_dir = tmp_path / "uploads"
        service.settings.processed_dir = tmp_path / "processed"
        service.settings.upload_dir.mkdir()
        service.settings.processed_dir.mkdir()

        record = asyncio.run(
            service.recognize_page_upload(upload, db, mode="qwen_page", preprocessing_applied=False)
        )

    assert record.recognition_mode == "qwen_page"
    assert record.line_count == 0
    assert record.recognized_text == "страница целиком"
    assert record.model_name == "page-stub"


def test_page_recognition_can_use_qwen_layout_mode(tmp_path: Path) -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(bind=engine)
    upload = UploadFile(filename="page.png", file=BytesIO(build_page_bytes()))

    with Session(engine) as db:
        service = RecognitionService(StubRecognizer())
        service._recognizers["qwen"] = PageStubRecognizer()
        service.settings.upload_dir = tmp_path / "uploads"
        service.settings.processed_dir = tmp_path / "processed"
        service.settings.upload_dir.mkdir()
        service.settings.processed_dir.mkdir()

        record = asyncio.run(
            service.recognize_page_upload(upload, db, mode="qwen_layout", preprocessing_applied=False)
        )

    assert record.recognition_mode == "qwen_layout"
    assert record.line_count == 0
    assert '"line_number": 1' in record.recognized_text
    assert '"box": [40, 40, 250, 80]' in record.recognized_text
    assert record.model_name == "page-stub"


def test_page_recognition_can_compare_trocr_and_qwen_lines(tmp_path: Path) -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(bind=engine)
    upload = UploadFile(filename="page.png", file=BytesIO(build_page_bytes()))

    with Session(engine) as db:
        service = RecognitionService(StubRecognizer())
        service._recognizers["qwen"] = PageStubRecognizer()
        service.settings.upload_dir = tmp_path / "uploads"
        service.settings.processed_dir = tmp_path / "processed"
        service.settings.upload_dir.mkdir()
        service.settings.processed_dir.mkdir()

        record = asyncio.run(service.recognize_page_upload(upload, db, mode="compare"))

    assert record.recognition_mode == "compare"
    assert record.line_count == 3
    assert "Строка 1" in record.recognized_text
    assert "TrOCR: line_000" in record.recognized_text
    assert "Qwen: line:line_000" in record.recognized_text


def test_page_recognition_can_use_hybrid_mode(tmp_path: Path) -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(bind=engine)
    upload = UploadFile(filename="page.png", file=BytesIO(build_page_bytes()))

    with Session(engine) as db:
        service = RecognitionService(StubRecognizer())
        service._recognizers["qwen"] = PageStubRecognizer()
        service.settings.upload_dir = tmp_path / "uploads"
        service.settings.processed_dir = tmp_path / "processed"
        service.settings.feedback_dir = tmp_path / "feedback"
        service.settings.upload_dir.mkdir()
        service.settings.processed_dir.mkdir()

        record = asyncio.run(service.recognize_page_upload(upload, db, mode="hybrid"))

    assert record.recognition_mode == "hybrid"
    assert record.line_count == 3
    assert "line:line_000" in record.recognized_text
    assert "[review line 1: TrOCR != Qwen]" in record.recognized_text
    assert record.model_name == "stub + page-stub"
    assert record.needs_review is True
    assert record.feedback_rating == "partial"
    assert record.feedback_saved_for_training is True
    assert record.review_notes[0]["line_number"] == 1
    assert record.review_notes[0]["trocr_text"] == "line_000"
    assert record.review_notes[0]["qwen_text"] == "line:line_000"

    metadata_path = tmp_path / "feedback" / "problems" / f"recognition_{record.id}.json"
    assert metadata_path.exists()
