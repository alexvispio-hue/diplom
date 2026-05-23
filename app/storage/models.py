from datetime import datetime

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
    recognized_text: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    processing_time_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    @property
    def original_image_url(self) -> str:
        return f"/api/history/{self.id}/image/original"

    @property
    def processed_image_url(self) -> str | None:
        if not self.processed_image_path:
            return None
        return f"/api/history/{self.id}/image/processed"
