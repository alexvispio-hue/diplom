from datetime import UTC, datetime

import json

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.storage.database import Base


class RecognitionRecord(Base):
    __tablename__ = "recognition_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    stored_image_path: Mapped[str] = mapped_column(String(512), nullable=False)
    processed_image_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    preprocessing_applied: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    recognition_mode: Mapped[str] = mapped_column(String(32), nullable=False, default="fragment")
    line_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    line_image_paths_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    line_bboxes_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    recognized_text: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    processing_time_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    needs_review: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    review_notes_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    feedback_rating: Mapped[str | None] = mapped_column(String(32), nullable=True)
    corrected_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    feedback_sample_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    feedback_created_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC), nullable=False)

    @property
    def original_image_url(self) -> str:
        return f"/api/history/{self.id}/image/original"

    @property
    def processed_image_url(self) -> str | None:
        if not self.processed_image_path:
            return None
        return f"/api/history/{self.id}/image/processed"

    @property
    def line_image_urls(self) -> list[str]:
        if not self.line_image_paths_json:
            return []
        return [f"/api/history/{self.id}/lines/{index}" for index, _ in enumerate(json.loads(self.line_image_paths_json))]

    @property
    def line_coordinates(self) -> list[dict[str, int]]:
        if not self.line_bboxes_json:
            return []
        coordinates = []
        for index, bbox in enumerate(json.loads(self.line_bboxes_json), start=1):
            x1, y1, x2, y2 = bbox
            coordinates.append(
                {
                    "line_number": index,
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                }
            )
        return coordinates

    @property
    def feedback_saved_for_training(self) -> bool:
        return self.feedback_sample_path is not None

    @property
    def review_notes(self) -> list[dict]:
        if not self.review_notes_json:
            return []
        return json.loads(self.review_notes_json)
