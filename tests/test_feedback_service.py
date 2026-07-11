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
        recognized_text="recognized text",
        model_name="stub",
        processing_time_ms=10,
    )


def test_partial_feedback_saves_problem_sample_with_page_artifacts(tmp_path: Path) -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(bind=engine)
    image_path = tmp_path / "word.png"
    processed_path = tmp_path / "word_processed.png"
    line_1_path = tmp_path / "line_1.png"
    line_2_path = tmp_path / "line_2.png"
    Image.new("RGB", (40, 20), "white").save(image_path)
    Image.new("RGB", (40, 20), "gray").save(processed_path)
    Image.new("RGB", (40, 8), "white").save(line_1_path)
    Image.new("RGB", (40, 8), "white").save(line_2_path)

    with Session(engine) as db:
        record = build_record(image_path)
        record.processed_image_path = str(processed_path)
        record.line_count = 2
        record.line_image_paths_json = json.dumps([str(line_1_path), str(line_2_path)])
        record.needs_review = True
        record.review_notes_json = json.dumps(
            [
                {
                    "line_number": 1,
                    "trocr_text": "recognized",
                    "qwen_text": "other",
                    "disagreement": 0.5,
                    "reason": "test",
                }
            ]
        )
        db.add(record)
        db.commit()
        db.refresh(record)

        service = FeedbackService()
        service.settings.feedback_dir = tmp_path / "feedback"
        service.save(record, "partial", db, corrected_text="corrected text")

    problems_dir = tmp_path / "feedback" / "problems"
    saved_image = problems_dir / f"recognition_{record.id}.png"
    saved_processed_image = problems_dir / f"recognition_{record.id}_processed.png"
    saved_line_1 = problems_dir / f"recognition_{record.id}_line_001.png"
    saved_line_2 = problems_dir / f"recognition_{record.id}_line_002.png"
    metadata_path = problems_dir / f"recognition_{record.id}.json"
    corrected_text_path = problems_dir / f"recognition_{record.id}.txt"

    assert saved_image.exists()
    assert saved_processed_image.exists()
    assert saved_line_1.exists()
    assert saved_line_2.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["recognized_text"] == "recognized text"
    assert metadata["corrected_text"] == "corrected text"
    assert metadata["recognition_mode"] == "fragment"
    assert metadata["needs_review"] is True
    assert metadata["review_notes"][0]["reason"] == "test"
    assert metadata["processed_image_filename"] == f"recognition_{record.id}_processed.png"
    assert metadata["line_image_filenames"] == [
        f"recognition_{record.id}_line_001.png",
        f"recognition_{record.id}_line_002.png",
    ]
    assert corrected_text_path.read_text(encoding="utf-8") == "corrected text"
    assert record.feedback_saved_for_training is True

    with Session(engine) as db:
        record = db.get(RecognitionRecord, record.id)
        service = FeedbackService()
        service.settings.feedback_dir = tmp_path / "feedback"
        service.save(record, "exact", db)

    assert saved_image.exists() is False
    assert saved_processed_image.exists() is False
    assert saved_line_1.exists() is False
    assert saved_line_2.exists() is False
    assert metadata_path.exists() is False
    assert corrected_text_path.exists() is False
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
        service.save(record, "exact", db, corrected_text="recognized text")

    assert record.feedback_saved_for_training is False
    assert record.corrected_text == "recognized text"


def test_list_problem_samples_builds_review_queue(tmp_path: Path) -> None:
    problems_dir = tmp_path / "feedback" / "problems"
    problems_dir.mkdir(parents=True)
    Image.new("RGB", (40, 20), "white").save(problems_dir / "recognition_7.png")
    Image.new("RGB", (40, 20), "gray").save(problems_dir / "recognition_7_processed.png")
    metadata = {
        "recognition_id": 7,
        "original_filename": "sample.png",
        "image_filename": "recognition_7.png",
        "processed_image_filename": "recognition_7_processed.png",
        "line_image_filenames": [],
        "corrected_text_filename": None,
        "recognized_text": "wrong",
        "corrected_text": "right",
        "rating": "partial",
        "model_name": "stub",
        "recognition_mode": "trocr_lines",
        "line_count": 0,
        "line_coordinates": [],
        "preprocessing_applied": True,
        "created_at": "2026-01-01T00:00:00",
        "feedback_created_at": "2026-01-01T00:00:01",
    }
    (problems_dir / "recognition_7.json").write_text(json.dumps(metadata), encoding="utf-8")

    service = FeedbackService()
    service.settings.feedback_dir = tmp_path / "feedback"

    samples = service.list_problem_samples()

    assert len(samples) == 1
    assert samples[0]["recognition_id"] == 7
    assert samples[0]["image_url"] == "/api/feedback/problems/7/image/original"
    assert samples[0]["processed_image_url"] == "/api/feedback/problems/7/image/processed"
    assert service.get_problem_image_path(7, "original").name == "recognition_7.png"


def test_list_problem_samples_supports_legacy_metadata(tmp_path: Path) -> None:
    problems_dir = tmp_path / "feedback" / "problems"
    problems_dir.mkdir(parents=True)
    legacy_metadata = {
        "recognition_id": 3,
        "original_filename": "legacy.jpg",
        "image_filename": "recognition_3.jpg",
        "recognized_text": "текст",
        "rating": "mismatch",
        "model_name": "legacy-model",
        "preprocessing_applied": True,
        "created_at": "2026-01-01T00:00:00",
        "feedback_created_at": "2026-01-01T00:00:01",
    }
    (problems_dir / "recognition_3.json").write_text(
        json.dumps(legacy_metadata),
        encoding="utf-8",
    )
    service = FeedbackService()
    service.settings.feedback_dir = tmp_path / "feedback"

    samples = service.list_problem_samples()

    assert samples[0]["recognition_mode"] == "fragment"
    assert samples[0]["line_count"] == 1
