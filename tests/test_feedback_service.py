import json
from pathlib import Path

from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.services.feedback_service import FeedbackService
from app.storage.database import Base
from app.storage.models import RecognitionRecord


def build_record(image_path: Path) -> RecognitionRecord:
    return RecognitionRecord(
        original_filename=image_path.name,
        stored_image_path=str(image_path),
        recognized_text="пример",
        model_name="stub",
        processing_time_ms=10,
    )


def test_partial_feedback_saves_problem_sample(tmp_path: Path) -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(bind=engine)
    image_path = tmp_path / "word.png"
    Image.new("RGB", (40, 20), "white").save(image_path)

    with Session(engine) as db:
        record = build_record(image_path)
        db.add(record)
        db.commit()
        db.refresh(record)

        service = FeedbackService()
        service.settings.feedback_dir = tmp_path / "feedback"
        service.save(record, "partial", db)

    saved_image = tmp_path / "feedback" / "problems" / f"recognition_{record.id}.png"
    metadata_path = tmp_path / "feedback" / "problems" / f"recognition_{record.id}.json"
    assert saved_image.exists()
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["recognized_text"] == "пример"
    assert record.feedback_saved_for_training is True

    with Session(engine) as db:
        record = db.get(RecognitionRecord, record.id)
        service = FeedbackService()
        service.settings.feedback_dir = tmp_path / "feedback"
        service.save(record, "exact", db)

    assert saved_image.exists() is False
    assert metadata_path.exists() is False
    assert record.feedback_saved_for_training is False


def test_exact_feedback_does_not_copy_image(tmp_path: Path) -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(bind=engine)
    image_path = tmp_path / "word.png"
    Image.new("RGB", (40, 20), "white").save(image_path)

    with Session(engine) as db:
        record = build_record(image_path)
        db.add(record)
        db.commit()
        db.refresh(record)

        service = FeedbackService()
        service.settings.feedback_dir = tmp_path / "feedback"
        service.save(record, "exact", db)

    assert record.feedback_saved_for_training is False
