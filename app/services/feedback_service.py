from datetime import UTC, datetime
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

    def save(
        self,
        record: RecognitionRecord,
        rating: str,
        db: Session,
        corrected_text: str | None = None,
        include_for_training: bool = False,
    ) -> RecognitionRecord:
        if rating not in VALID_RATINGS:
            raise ValueError("Неизвестная оценка распознавания.")

        record.feedback_rating = rating
        record.corrected_text = self._normalize_corrected_text(corrected_text)
        record.feedback_created_at = datetime.now(UTC)
        record.feedback_training_approved = bool(include_for_training)
        record.feedback_training_excluded = False
        if rating in PROBLEM_RATINGS:
            self._save_problem_sample(record)
        elif record.feedback_sample_path:
            self._remove_problem_sample(record)

        db.add(record)
        db.commit()
        db.refresh(record)
        return record

    def exclude_from_training(self, record: RecognitionRecord, db: Session) -> RecognitionRecord:
        record.feedback_training_approved = False
        record.feedback_training_excluded = True
        if record.feedback_sample_path:
            metadata_path = self.settings.feedback_dir / "problems" / f"recognition_{record.id}.json"
            if metadata_path.is_file():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                metadata["training_approved"] = False
                metadata["training_excluded"] = True
                metadata_path.write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record

    def list_problem_samples(self) -> list[dict]:
        problems_dir = self.settings.feedback_dir / "problems"
        if not problems_dir.exists():
            return []

        samples = []
        for metadata_path in sorted(problems_dir.glob("recognition_*.json"), reverse=True):
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            recognition_id = int(metadata["recognition_id"])
            image_filename = metadata["image_filename"]
            processed_image_filename = metadata.get("processed_image_filename")
            line_image_filenames = metadata.get("line_image_filenames") or []
            metadata["image_url"] = f"/api/feedback/problems/{recognition_id}/image/original"
            metadata["processed_image_url"] = (
                f"/api/feedback/problems/{recognition_id}/image/processed"
                if processed_image_filename
                else None
            )
            metadata["line_image_urls"] = [
                f"/api/feedback/problems/{recognition_id}/lines/{index}"
                for index, _ in enumerate(line_image_filenames)
            ]
            metadata.setdefault("line_coordinates", [])
            metadata.setdefault("recognition_mode", "fragment")
            metadata.setdefault("line_count", 1)
            metadata.setdefault("needs_review", False)
            metadata.setdefault("review_notes", [])
            metadata.setdefault("corrected_text", None)
            metadata.setdefault("corrected_text_filename", None)
            metadata.setdefault("training_approved", False)
            metadata.setdefault("training_excluded", False)
            samples.append(metadata)
        return samples

    def get_problem_image_path(
        self,
        recognition_id: int,
        image_kind: str,
        line_index: int | None = None,
    ) -> Path:
        metadata = self._load_problem_metadata(recognition_id)
        if image_kind == "original":
            filename = metadata["image_filename"]
        elif image_kind == "processed":
            filename = metadata.get("processed_image_filename")
        elif image_kind == "line":
            line_filenames = metadata.get("line_image_filenames") or []
            if line_index is None or line_index < 0 or line_index >= len(line_filenames):
                raise ValueError("Строка проблемного примера не найдена.")
            filename = line_filenames[line_index]
        else:
            raise ValueError("Изображение проблемного примера не найдено.")

        if not filename:
            raise ValueError("Изображение проблемного примера не найдено.")
        return self._resolve_problem_file(filename)

    def _save_problem_sample(self, record: RecognitionRecord) -> None:
        source_path = Path(record.stored_image_path)
        if not source_path.exists():
            raise ValueError("Исходное изображение для обратной связи отсутствует.")

        problems_dir = self.settings.feedback_dir / "problems"
        problems_dir.mkdir(parents=True, exist_ok=True)
        suffix = source_path.suffix.lower() or ".png"
        sample_path = problems_dir / f"recognition_{record.id}{suffix}"
        shutil.copy2(source_path, sample_path)
        processed_image_filename = self._copy_optional_image(
            record.processed_image_path,
            problems_dir / f"recognition_{record.id}_processed",
        )
        line_image_filenames = self._copy_line_images(record, problems_dir)
        corrected_text_path = problems_dir / f"recognition_{record.id}.txt"
        if record.corrected_text:
            corrected_text_path.write_text(record.corrected_text, encoding="utf-8")
        else:
            corrected_text_path.unlink(missing_ok=True)

        metadata = {
            "recognition_id": record.id,
            "original_filename": record.original_filename,
            "image_filename": sample_path.name,
            "processed_image_filename": processed_image_filename,
            "line_image_filenames": line_image_filenames,
            "corrected_text_filename": corrected_text_path.name if record.corrected_text else None,
            "recognized_text": record.recognized_text,
            "corrected_text": record.corrected_text,
            "rating": record.feedback_rating,
            "model_name": record.model_name,
            "recognition_mode": record.recognition_mode,
            "needs_review": record.needs_review,
            "review_notes": record.review_notes,
            "line_count": record.line_count,
            "line_image_paths": json.loads(record.line_image_paths_json or "[]"),
            "line_coordinates": record.line_coordinates,
            "preprocessing_applied": record.preprocessing_applied,
            "created_at": record.created_at.isoformat(),
            "feedback_created_at": record.feedback_created_at.isoformat(),
            "training_approved": bool(record.feedback_training_approved),
            "training_excluded": bool(record.feedback_training_excluded),
        }
        metadata_path = problems_dir / f"recognition_{record.id}.json"
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        record.feedback_sample_path = str(sample_path)

    def _load_problem_metadata(self, recognition_id: int) -> dict:
        metadata_path = self._resolve_problem_file(f"recognition_{recognition_id}.json")
        if not metadata_path.exists():
            raise ValueError("Проблемный пример не найден.")
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    def _resolve_problem_file(self, filename: str) -> Path:
        problems_dir = (self.settings.feedback_dir / "problems").resolve()
        target_path = (problems_dir / filename).resolve()
        try:
            target_path.relative_to(problems_dir)
        except ValueError as exc:
            raise ValueError("Некорректный путь к проблемному примеру.") from exc
        return target_path

    @staticmethod
    def _copy_optional_image(source: str | None, target_without_suffix: Path) -> str | None:
        if not source:
            return None
        source_path = Path(source)
        if not source_path.exists():
            return None

        suffix = source_path.suffix.lower() or ".png"
        target_path = target_without_suffix.with_suffix(suffix)
        shutil.copy2(source_path, target_path)
        return target_path.name

    @classmethod
    def _copy_line_images(cls, record: RecognitionRecord, problems_dir: Path) -> list[str]:
        copied: list[str] = []
        for index, line_path_value in enumerate(json.loads(record.line_image_paths_json or "[]"), start=1):
            copied_filename = cls._copy_optional_image(
                line_path_value,
                problems_dir / f"recognition_{record.id}_line_{index:03d}",
            )
            if copied_filename:
                copied.append(copied_filename)
        return copied

    @staticmethod
    def _remove_problem_sample(record: RecognitionRecord) -> None:
        sample_path = Path(record.feedback_sample_path)
        problems_dir = sample_path.parent
        for related_path in problems_dir.glob(f"recognition_{record.id}*"):
            if related_path.is_file():
                related_path.unlink(missing_ok=True)
        record.feedback_sample_path = None

    @staticmethod
    def _normalize_corrected_text(corrected_text: str | None) -> str | None:
        if corrected_text is None:
            return None
        normalized = corrected_text.strip()
        return normalized or None
