from datetime import datetime

from typing import Any, Literal

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
    reliability_score: float | None = None
    model_name: str
    processing_time_ms: int
    peak_vram_bytes: int | None = None
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
    feedback_training_approved: bool = False
    feedback_training_excluded: bool = False
    original_image_url: str
    processed_image_url: str | None
    model_registry_id: str | None = None
    routing_reason: str | None = None
    image_quality: dict | None = None


class RecognitionHistoryItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    original_filename: str
    file_size_bytes: int
    recognized_text: str
    confidence: float | None
    reliability_score: float | None = None
    model_name: str
    processing_time_ms: int
    peak_vram_bytes: int | None = None
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
    feedback_training_approved: bool = False
    feedback_training_excluded: bool = False
    original_image_url: str
    processed_image_url: str | None
    model_registry_id: str | None = None
    routing_reason: str | None = None
    image_quality: dict | None = None


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
    cuda_version: str | None = None
    gpu_devices: list[dict[str, Any]] = Field(default_factory=list)
    python_version: str | None = None
    platform: str | None = None


class DiagnosticsResponse(BaseModel):
    generated_at: str
    application: dict[str, Any]
    runtime: dict[str, Any]
    storage: dict[str, Any]
    ports: dict[str, str]
    models: list[dict[str, Any]]
    recent_errors: list[str]


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
    pipeline_maturity: str = "not_ready"
    production_ready: bool = False
    blocking_reasons: list[str] = Field(default_factory=list)
    full_dataset: dict = Field(default_factory=dict)


class RegisteredModel(BaseModel):
    id: str
    backend: Literal["trocr", "qwen"]
    base_model: str
    revision: str | None = None
    adapter: str | None = None
    local_path: str | None = None
    stage: str
    quality_report: str | None = None
    created_at: str
    notes: str | None = None
    is_production: bool = False
    installed: bool = True
    source: str | None = None
    size_bytes: int | None = None
    checksum_sha256: str | None = None


class RegisterModelRequest(BaseModel):
    id: str
    backend: Literal["trocr", "qwen"]
    base_model: str
    revision: str | None = None
    adapter: str | None = None
    local_path: str | None = None
    stage: Literal["baseline", "experimental", "candidate", "rejected"] = "experimental"
    quality_report: str | None = None
    notes: str | None = None
    source: str | None = None
    size_bytes: int | None = None
    checksum_sha256: str | None = None


class ModelInstallRequest(BaseModel):
    id: str
    repo_id: str
    backend: Literal["trocr", "qwen"]
    revision: str | None = None
    expected_sha256: str | None = None
    stage: Literal["baseline", "experimental", "candidate", "rejected"] = "experimental"


class ModelDownloadResponse(BaseModel):
    id: str
    model_id: str
    repo_id: str
    backend: str
    revision: str | None = None
    expected_sha256: str | None = None
    stage: str
    target_path: str
    status: str
    downloaded_bytes: int
    total_bytes: int | None = None
    completed_files: int
    total_files: int | None = None
    checksum_sha256: str | None = None
    error: str | None = None
    created_at: str
    updated_at: str


class ModelActionResponse(BaseModel):
    status: str
    model: RegisteredModel | None = None


class ModelBackendRequest(BaseModel):
    backend: Literal["trocr", "qwen"]


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
    training_approved: bool = False
    training_excluded: bool = False


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


PageRecognitionMode = Literal["auto", "trocr_lines", "qwen_lines", "qwen_page", "qwen_layout", "compare", "hybrid"]


class FeedbackRequest(BaseModel):
    rating: Literal["exact", "partial", "mismatch"]
    corrected_text: str | None = None
    include_for_training: bool = False


class FeedbackByIdRequest(FeedbackRequest):
    recognition_id: int


class OcrJobPageResponse(BaseModel):
    id: int
    page_number: int
    original_filename: str
    status: str
    recognition_id: int | None = None
    error: str | None = None
    recognized_text: str | None = None
    corrected_text: str | None = None
    reliability_score: float | None = None
    needs_review: bool = False
    model_name: str | None = None
    recognition_mode: str | None = None
    routing_reason: str | None = None
    image_quality: dict[str, Any] | None = None
    peak_vram_bytes: int | None = None


class OcrJobResponse(BaseModel):
    id: str
    title: str
    status: str
    document_target: str
    recognition_mode: str
    preprocessing_applied: bool
    model_override: str | None = None
    max_tokens: int
    total_pages: int
    completed_pages: int
    failed_pages: int
    cancel_requested: bool
    pause_requested: bool = False
    error: str | None = None
    created_at: datetime
    updated_at: datetime
    pages: list[OcrJobPageResponse]


class FolderJobRequest(BaseModel):
    folder: str
    title: str = "Новый документ"
    recursive: bool = False
    sort_by: Literal["name", "modified", "created"] = "name"
    document_target: Literal["fragment", "page"] = "page"
    mode: PageRecognitionMode = "auto"
    preprocess: bool = True
    model: str | None = None
    max_tokens: int = 1024


class DocumentDraftCreate(BaseModel):
    title: str = "Новый документ"
    source_job_id: str | None = None
    pages: list[dict[str, Any]] = Field(default_factory=list)
    document_text: str = ""


class DocumentFromJobRequest(BaseModel):
    title: str | None = None


class DocumentDraftUpdate(BaseModel):
    expected_revision: int
    title: str | None = None
    pages: list[dict[str, Any]] | None = None
    document_text: str | None = None


class DocumentRestoreRequest(BaseModel):
    expected_revision: int


class DocumentDraftResponse(BaseModel):
    id: str
    title: str
    source_job_id: str | None = None
    pages: list[dict[str, Any]]
    document_text: str
    revision: int
    created_at: datetime
    updated_at: datetime


class DocumentRevisionResponse(BaseModel):
    revision: int
    title: str
    pages: list[dict[str, Any]]
    document_text: str
    created_at: datetime


class DictionaryEntryRequest(BaseModel):
    word: str


class DictionaryResponse(BaseModel):
    words: list[str]


class GoldReviewUpdate(BaseModel):
    review_status: Literal["pending_human_review", "verified", "rejected"]
    reference_text: str
    reviewer: str | None = None
    review_notes: str | None = None


class GoldReviewSummary(BaseModel):
    total: int
    pending: int
    verified: int
    rejected: int
    sources: dict[str, int]
    ready_to_finalize: bool
