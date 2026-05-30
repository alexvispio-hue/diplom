from datetime import datetime

from typing import Literal

from pydantic import BaseModel, ConfigDict


class RecognitionResponse(BaseModel):
    id: int
    filename: str
    file_size_bytes: int
    recognized_text: str
    confidence: float | None
    model_name: str
    processing_time_ms: int
    created_at: datetime
    preprocessing_applied: bool
    recognition_mode: str
    line_count: int
    line_image_urls: list[str]
    feedback_rating: str | None
    feedback_saved_for_training: bool
    original_image_url: str
    processed_image_url: str | None


class RecognitionHistoryItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    original_filename: str
    file_size_bytes: int
    recognized_text: str
    confidence: float | None
    model_name: str
    processing_time_ms: int
    created_at: datetime
    preprocessing_applied: bool
    recognition_mode: str
    line_count: int
    line_image_urls: list[str]
    feedback_rating: str | None
    feedback_saved_for_training: bool
    original_image_url: str
    processed_image_url: str | None


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str


class ModelInfo(BaseModel):
    backend: str
    model_name: str
    local_inference: bool


class FeedbackRequest(BaseModel):
    rating: Literal["exact", "partial", "mismatch"]
