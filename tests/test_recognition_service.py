from io import BytesIO
from pathlib import Path
import asyncio

from fastapi import UploadFile
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.image_processing.page_segmentation import PageSegmentationResult, TextLine
from app.ocr.base import BaseRecognizer, RecognitionResult
from app.services.recognition_service import RecognitionService
from app.storage.database import Base


class StubRecognizer(BaseRecognizer):
    def recognize(self, image_path: Path) -> RecognitionResult:
        return RecognitionResult(text="текст", confidence=None, model_name="stub")


class StubPageSegmenter:
    def segment(self, source_path: Path, output_dir: Path, annotated_image_path: Path) -> PageSegmentationResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        line_path = output_dir / "line_000.png"
        Image.new("L", (30, 12), "white").save(line_path)
        Image.new("RGB", (40, 20), "white").save(annotated_image_path)
        return PageSegmentationResult(
            annotated_image_path=annotated_image_path,
            lines=[TextLine(index=0, image_path=line_path, bbox=(2, 3, 32, 15))],
        )


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


def test_page_recognition_can_skip_intermediate_images_and_coordinates(tmp_path: Path) -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(bind=engine)

    with Session(engine) as db:
        service = RecognitionService(StubRecognizer(), page_segmenter=StubPageSegmenter())
        service.settings.upload_dir = tmp_path / "uploads"
        service.settings.processed_dir = tmp_path / "processed"
        service.settings.upload_dir.mkdir()
        service.settings.processed_dir.mkdir()

        record = asyncio.run(
            service.recognize_page_upload(
                build_image_upload(),
                db,
                preprocessing_applied=False,
                save_intermediate_images=False,
                return_coordinates=False,
            )
        )

    assert record.recognized_text == "текст"
    assert record.line_count == 1
    assert record.processed_image_path is None
    assert record.line_image_paths_json is None
    assert record.line_bboxes_json is None
    assert record.line_image_urls == []
    assert record.line_coordinates == []
    assert not any((tmp_path / "processed").iterdir())


def test_page_recognition_stores_coordinates_when_requested(tmp_path: Path) -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(bind=engine)

    with Session(engine) as db:
        service = RecognitionService(StubRecognizer(), page_segmenter=StubPageSegmenter())
        service.settings.upload_dir = tmp_path / "uploads"
        service.settings.processed_dir = tmp_path / "processed"
        service.settings.upload_dir.mkdir()
        service.settings.processed_dir.mkdir()

        record = asyncio.run(
            service.recognize_page_upload(
                build_image_upload(),
                db,
                preprocessing_applied=False,
                save_intermediate_images=True,
                return_coordinates=True,
            )
        )

    assert record.processed_image_path is not None
    assert record.line_image_paths_json is not None
    assert record.line_coordinates == [{"line_number": 1, "x1": 2, "y1": 3, "x2": 32, "y2": 15}]
