from datetime import datetime

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class LineCoordinate(BaseModel):
    line_number: int
    x1: int
    y1: int
    x2: int
    y2: int


class ReviewNote(BaseModel):
    line_number: int
    trocr_text: str
    qwen_text: str
    disagreement: float
    reason: str


class RecognitionResponse(BaseModel):
    id: int
    filename: str
    file_size_bytes: int
    recognized_text: str
    confidence: float | None
    model_name: str
    processing_time_ms: int
    created_at: datetime
    needs_review: bool = False
    review_notes: list[ReviewNote] = Field(default_factory=list)
    preprocessing_applied: bool
    recognition_mode: str
    line_count: int
    line_image_urls: list[str]
    line_coordinates: list[LineCoordinate] = Field(default_factory=list)
    feedback_rating: str | None
    corrected_text: str | None
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
    needs_review: bool = False
    review_notes: list[ReviewNote] = Field(default_factory=list)
    preprocessing_applied: bool
    recognition_mode: str
    line_count: int
    line_image_urls: list[str]
    line_coordinates: list[LineCoordinate] = Field(default_factory=list)
    feedback_rating: str | None
    corrected_text: str | None
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
    supports_page: bool = False
    supports_training: bool = False


class MLPackageStatus(BaseModel):
    package: str
    import_name: str
    installed: bool
    version: str | None


class MLEnvironmentStatus(BaseModel):
    packages: list[MLPackageStatus]
    ready_for_trocr: bool
    ready_for_qwen_inference: bool
    ready_for_qwen_lora: bool
    torch_cuda_available: bool | None
    torch_device_count: int | None


class QwenSftDatasetStatus(BaseModel):
    path: str
    examples: int


class QwenAcceptanceCounts(BaseModel):
    passed: int
    partial: int
    failed: int


class QwenProjectStatusResponse(BaseModel):
    generated_at: str
    model: str | None
    install_profile: str | None
    runtime_status: str
    missing_packages: list[str]
    recommended_next_command: str
    sft_dataset: QwenSftDatasetStatus
    steps: dict[str, str]
    lora_dry_run_passed: bool
    acceptance: QwenAcceptanceCounts


class AcceptanceAuditItem(BaseModel):
    id: int
    requirement: str
    status: Literal["passed", "partial", "failed"]
    evidence: list[str]
    notes: str = ""


class FeedbackStatsResponse(BaseModel):
    total_results: int
    rated_results: int
    unrated_results: int
    exact_count: int
    partial_count: int
    mismatch_count: int
    exact_rate: float | None
    exact_rate_all: float | None


class FeedbackProblemItem(BaseModel):
    recognition_id: int
    original_filename: str
    image_filename: str
    image_url: str
    processed_image_filename: str | None = None
    processed_image_url: str | None = None
    line_image_filenames: list[str] = Field(default_factory=list)
    line_image_urls: list[str] = Field(default_factory=list)
    corrected_text_filename: str | None = None
    recognized_text: str
    corrected_text: str | None = None
    rating: str
    model_name: str
    recognition_mode: str
    needs_review: bool = False
    review_notes: list[ReviewNote] = Field(default_factory=list)
    line_count: int
    line_coordinates: list[LineCoordinate] = Field(default_factory=list)
    preprocessing_applied: bool
    created_at: str
    feedback_created_at: str


class ExperimentSummaryItem(BaseModel):
    summary_path: str
    model: str
    mode: str
    dataset: str
    cer: float | None
    wer: float | None
    page_time: float | None
    line_time: float | None
    notes: str


PageRecognitionMode = Literal["trocr_lines", "qwen_lines", "qwen_page", "qwen_layout", "compare", "hybrid"]


class FeedbackRequest(BaseModel):
    rating: Literal["exact", "partial", "mismatch"]
    corrected_text: str | None = None


class FeedbackByIdRequest(FeedbackRequest):
    recognition_id: int
