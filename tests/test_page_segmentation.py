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

    assert record.recognition_mode == "page"
    assert record.line_count == 3
    assert record.recognized_text == "line_000\nline_001\nline_002"
    assert record.confidence == pytest.approx(0.8)
    assert len(record.line_image_urls) == 3
