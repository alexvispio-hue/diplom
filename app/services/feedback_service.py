from datetime import datetime
import json
from pathlib import Path
import shutil

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.storage.models import RecognitionRecord


PROBLEM_RATINGS = {"partial", "mismatch"}
VALID_RATINGS = {"exact", *PROBLEM_RATINGS}


class FeedbackService:
    def __init__(self) -> None:
        self.settings = get_settings()

    def save(self, record: RecognitionRecord, rating: str, db: Session) -> RecognitionRecord:
        if rating not in VALID_RATINGS:
            raise ValueError("Неизвестная оценка распознавания.")

        record.feedback_rating = rating
        record.feedback_created_at = datetime.utcnow()
        if rating in PROBLEM_RATINGS:
            self._save_problem_sample(record)
        elif record.feedback_sample_path:
            self._remove_problem_sample(record)

        db.add(record)
        db.commit()
        db.refresh(record)
        return record

    def _save_problem_sample(self, record: RecognitionRecord) -> None:
        source_path = Path(record.stored_image_path)
        if not source_path.exists():
            raise ValueError("Исходное изображение для обратной связи отсутствует.")

        problems_dir = self.settings.feedback_dir / "problems"
        problems_dir.mkdir(parents=True, exist_ok=True)
        suffix = source_path.suffix.lower() or ".png"
        sample_path = problems_dir / f"recognition_{record.id}{suffix}"
        shutil.copy2(source_path, sample_path)

        metadata = {
            "recognition_id": record.id,
            "original_filename": record.original_filename,
            "image_filename": sample_path.name,
            "recognized_text": record.recognized_text,
            "rating": record.feedback_rating,
            "model_name": record.model_name,
            "preprocessing_applied": record.preprocessing_applied,
            "created_at": record.created_at.isoformat(),
            "feedback_created_at": record.feedback_created_at.isoformat(),
        }
        metadata_path = problems_dir / f"recognition_{record.id}.json"
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        record.feedback_sample_path = str(sample_path)

    @staticmethod
    def _remove_problem_sample(record: RecognitionRecord) -> None:
        sample_path = Path(record.feedback_sample_path)
        metadata_path = sample_path.with_suffix(".json")
        sample_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
        record.feedback_sample_path = None
