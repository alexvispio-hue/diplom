import json
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import desc
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.api.schemas import (
    AcceptanceAuditItem,
    ExperimentSummaryItem,
    FeedbackByIdRequest,
    FeedbackProblemItem,
    FeedbackStatsResponse,
    FeedbackRequest,
    HealthResponse,
    MLEnvironmentStatus,
    ModelInfo,
    PageRecognitionMode,
    QwenProjectStatusResponse,
    RecognitionHistoryItem,
    RecognitionResponse,
)
from app.core.config import get_settings
from app.ocr.factory import build_recognizer
from app.services.experiment_results_service import ExperimentResultsService
from app.services.ml_environment_service import QWEN_RUNTIME_PACKAGES, collect_status, missing_packages
from app.services.qwen_project_status_service import QwenProjectStatusService
from app.services.recognition_service import (
    PAGE_MODE_COMPARE,
    PAGE_MODE_HYBRID,
    PAGE_MODE_QWEN_LAYOUT,
    PAGE_MODE_QWEN_LINES,
    PAGE_MODE_QWEN_PAGE,
    PAGE_MODE_TROCR_LINES,
    RecognitionService,
)
from app.services.feedback_service import FeedbackService
from app.storage.database import get_db
from app.storage.models import RecognitionRecord
from scripts.audit_acceptance import check_results


router = APIRouter()
settings = get_settings()
recognition_service = RecognitionService(build_recognizer())
feedback_service = FeedbackService()
experiment_results_service = ExperimentResultsService()
qwen_project_status_service = QwenProjectStatusService()
QWEN_PAGE_MODES = {
    PAGE_MODE_QWEN_LINES,
    PAGE_MODE_QWEN_PAGE,
    PAGE_MODE_QWEN_LAYOUT,
    PAGE_MODE_COMPARE,
    PAGE_MODE_HYBRID,
}


def _build_recognition_response(record: RecognitionRecord) -> RecognitionResponse:
    return RecognitionResponse(
        id=record.id,
        filename=record.original_filename,
        file_size_bytes=record.file_size_bytes,
        recognized_text=record.recognized_text,
        confidence=record.confidence,
        model_name=record.model_name,
        processing_time_ms=record.processing_time_ms,
        created_at=record.created_at,
        needs_review=bool(record.needs_review),
        review_notes=record.review_notes,
        preprocessing_applied=record.preprocessing_applied,
        recognition_mode=record.recognition_mode,
        line_count=record.line_count,
        line_image_urls=record.line_image_urls,
        line_coordinates=record.line_coordinates,
        feedback_rating=record.feedback_rating,
        corrected_text=record.corrected_text,
        feedback_saved_for_training=record.feedback_saved_for_training,
        original_image_url=f"/api/history/{record.id}/image/original",
        processed_image_url=f"/api/history/{record.id}/image/processed" if record.processed_image_path else None,
    )


def _qwen_model_display_name() -> str:
    model_name = getattr(settings, "qwen_effective_model_name", None) or settings.qwen_model_name
    if settings.qwen_lora_adapter_dir is None:
        return model_name
    return f"{model_name}+lora:{Path(settings.qwen_lora_adapter_dir).name}"


def _ensure_qwen_environment_ready(mode: str) -> None:
    if mode not in QWEN_PAGE_MODES:
        return

    status = collect_status()
    if status["ready_for_qwen_inference"]:
        return

    missing = missing_packages(status, QWEN_RUNTIME_PACKAGES)
    missing_text = ", ".join(missing) if missing else "unknown"
    raise HTTPException(
        status_code=503,
        detail=(
            "Qwen mode requires local ML dependencies before inference. "
            f"Missing packages: {missing_text}. "
            "Run `python scripts/setup_ml_environment.py --install` after confirming the download size."
        ),
    )


def _validate_max_tokens(max_tokens: int | None) -> None:
    if max_tokens is not None and max_tokens <= 0:
        raise HTTPException(status_code=400, detail="max_tokens должен быть положительным числом.")


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", app_name=settings.app_name, version=settings.app_version)


@router.get("/models", response_model=list[ModelInfo])
def models() -> list[ModelInfo]:
    return [
        ModelInfo(
            backend="trocr",
            model_name=recognition_service.recognizer.model_name,
            local_inference=True,
            supports_page=True,
            supports_training=True,
        ),
        ModelInfo(
            backend="qwen",
            model_name=_qwen_model_display_name(),
            local_inference=True,
            supports_page=True,
            supports_training=True,
        ),
    ]


@router.get("/ml-status", response_model=MLEnvironmentStatus)
def ml_status() -> MLEnvironmentStatus:
    return MLEnvironmentStatus(**collect_status())


@router.get("/qwen-status", response_model=QwenProjectStatusResponse)
def qwen_status() -> QwenProjectStatusResponse:
    return QwenProjectStatusResponse(**qwen_project_status_service.get_status())


@router.get("/acceptance-audit", response_model=list[AcceptanceAuditItem])
def acceptance_audit() -> list[AcceptanceAuditItem]:
    return [AcceptanceAuditItem(**check.__dict__) for check in check_results()]


@router.get("/experiments/page-summary", response_model=list[ExperimentSummaryItem])
def page_experiment_summary() -> list:
    return experiment_results_service.list_page_summaries()


@router.post("/recognize", response_model=RecognitionResponse)
async def recognize_image(
    file: UploadFile = File(...),
    preprocess: bool = Form(default=True),
    db: Session = Depends(get_db),
) -> RecognitionResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Загрузите файл изображения.")

    try:
        record = await recognition_service.recognize_upload(file, db, preprocessing_applied=preprocess)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return _build_recognition_response(record)


async def _recognize_page_impl(
    file: UploadFile,
    db: Session,
    mode: PageRecognitionMode = PAGE_MODE_TROCR_LINES,
    preprocess: bool = True,
    model: str | None = None,
    max_tokens: int | None = None,
    save_intermediate_images: bool = True,
    return_coordinates: bool = True,
) -> RecognitionResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Загрузите файл изображения.")
    _validate_max_tokens(max_tokens)
    _ensure_qwen_environment_ready(mode)

    try:
        record = await recognition_service.recognize_page_upload(
            file,
            db,
            mode=mode,
            preprocessing_applied=preprocess,
            model_name=model,
            max_tokens=max_tokens,
            save_intermediate_images=save_intermediate_images,
            return_coordinates=return_coordinates,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return _build_recognition_response(record)


@router.post("/recognize-page", response_model=RecognitionResponse)
async def recognize_page_legacy(
    file: UploadFile = File(...),
    mode: PageRecognitionMode = Form(default=PAGE_MODE_TROCR_LINES),
    preprocess: bool = Form(default=True),
    model: str | None = Form(default=None),
    max_tokens: int | None = Form(default=None),
    save_intermediate_images: bool = Form(default=True),
    return_coordinates: bool = Form(default=True),
    db: Session = Depends(get_db),
) -> RecognitionResponse:
    return await _recognize_page_impl(
        file=file,
        db=db,
        mode=mode,
        preprocess=preprocess,
        model=model,
        max_tokens=max_tokens,
        save_intermediate_images=save_intermediate_images,
        return_coordinates=return_coordinates,
    )


@router.post("/recognize/page", response_model=RecognitionResponse)
async def recognize_page(
    file: UploadFile = File(...),
    mode: PageRecognitionMode = Form(default=PAGE_MODE_TROCR_LINES),
    preprocess: bool = Form(default=True),
    model: str | None = Form(default=None),
    max_tokens: int | None = Form(default=None),
    save_intermediate_images: bool = Form(default=True),
    return_coordinates: bool = Form(default=True),
    db: Session = Depends(get_db),
) -> RecognitionResponse:
    return await _recognize_page_impl(
        file=file,
        db=db,
        mode=mode,
        preprocess=preprocess,
        model=model,
        max_tokens=max_tokens,
        save_intermediate_images=save_intermediate_images,
        return_coordinates=return_coordinates,
    )


@router.post("/recognize/page/qwen", response_model=RecognitionResponse)
async def recognize_page_qwen(
    file: UploadFile = File(...),
    preprocess: bool = Form(default=True),
    max_tokens: int | None = Form(default=None),
    db: Session = Depends(get_db),
) -> RecognitionResponse:
    return await _recognize_page_impl(
        file=file,
        db=db,
        mode="qwen_page",
        preprocess=preprocess,
        max_tokens=max_tokens,
    )


@router.post("/recognize/page/compare", response_model=RecognitionResponse)
async def recognize_page_compare(
    file: UploadFile = File(...),
    preprocess: bool = Form(default=True),
    max_tokens: int | None = Form(default=None),
    db: Session = Depends(get_db),
) -> RecognitionResponse:
    return await _recognize_page_impl(
        file=file,
        db=db,
        mode="compare",
        preprocess=preprocess,
        max_tokens=max_tokens,
    )


@router.get("/history", response_model=list[RecognitionHistoryItem])
def history(db: Session = Depends(get_db), limit: int = 20) -> list[RecognitionRecord]:
    return (
        db.query(RecognitionRecord)
        .order_by(desc(RecognitionRecord.created_at))
        .limit(min(limit, 100))
        .all()
    )


@router.get("/history/{record_id}", response_model=RecognitionHistoryItem)
def history_item(record_id: int, db: Session = Depends(get_db)) -> RecognitionRecord:
    record = db.get(RecognitionRecord, record_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Запись не найдена.")
    return record


@router.post("/history/{record_id}/feedback", response_model=RecognitionHistoryItem)
def save_feedback(record_id: int, feedback: FeedbackRequest, db: Session = Depends(get_db)) -> RecognitionRecord:
    return _save_feedback_for_record(record_id, feedback, db)


@router.post("/feedback", response_model=RecognitionHistoryItem)
def save_feedback_by_id(feedback: FeedbackByIdRequest, db: Session = Depends(get_db)) -> RecognitionRecord:
    return _save_feedback_for_record(feedback.recognition_id, feedback, db)


@router.get("/feedback/stats", response_model=FeedbackStatsResponse)
def feedback_stats(db: Session = Depends(get_db)) -> FeedbackStatsResponse:
    total_results = db.query(func.count(RecognitionRecord.id)).scalar() or 0
    rating_rows = (
        db.query(RecognitionRecord.feedback_rating, func.count(RecognitionRecord.id))
        .filter(RecognitionRecord.feedback_rating.is_not(None))
        .group_by(RecognitionRecord.feedback_rating)
        .all()
    )
    rating_counts = {rating: count for rating, count in rating_rows}
    exact_count = int(rating_counts.get("exact", 0))
    partial_count = int(rating_counts.get("partial", 0))
    mismatch_count = int(rating_counts.get("mismatch", 0))
    rated_results = exact_count + partial_count + mismatch_count
    return FeedbackStatsResponse(
        total_results=int(total_results),
        rated_results=rated_results,
        unrated_results=max(0, int(total_results) - rated_results),
        exact_count=exact_count,
        partial_count=partial_count,
        mismatch_count=mismatch_count,
        exact_rate=exact_count / rated_results if rated_results else None,
        exact_rate_all=exact_count / total_results if total_results else None,
    )


@router.get("/feedback/problems", response_model=list[FeedbackProblemItem])
def feedback_problems() -> list[dict]:
    return feedback_service.list_problem_samples()


@router.get("/feedback/problems/{recognition_id}/image/{image_kind}")
def feedback_problem_image(recognition_id: int, image_kind: str) -> FileResponse:
    try:
        path = feedback_service.get_problem_image_path(recognition_id, image_kind)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not path.exists():
        raise HTTPException(status_code=404, detail="Файл проблемного примера отсутствует.")
    return FileResponse(path)


@router.get("/feedback/problems/{recognition_id}/lines/{line_index}")
def feedback_problem_line_image(recognition_id: int, line_index: int) -> FileResponse:
    try:
        path = feedback_service.get_problem_image_path(recognition_id, "line", line_index=line_index)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not path.exists():
        raise HTTPException(status_code=404, detail="Файл строки проблемного примера отсутствует.")
    return FileResponse(path)


def _save_feedback_for_record(record_id: int, feedback: FeedbackRequest, db: Session) -> RecognitionRecord:
    record = db.get(RecognitionRecord, record_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Запись не найдена.")
    try:
        return feedback_service.save(record, feedback.rating, db, corrected_text=feedback.corrected_text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/history/{record_id}/image/{image_kind}")
def history_image(record_id: int, image_kind: str, db: Session = Depends(get_db)) -> FileResponse:
    record = db.get(RecognitionRecord, record_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Запись не найдена.")

    if image_kind == "original":
        path = Path(record.stored_image_path)
    elif image_kind == "processed" and record.processed_image_path:
        path = Path(record.processed_image_path)
    else:
        raise HTTPException(status_code=404, detail="Изображение не найдено.")

    if not path.exists():
        raise HTTPException(status_code=404, detail="Файл изображения отсутствует.")
    return FileResponse(path)


@router.get("/history/{record_id}/lines/{line_index}")
def history_line_image(record_id: int, line_index: int, db: Session = Depends(get_db)) -> FileResponse:
    record = db.get(RecognitionRecord, record_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Запись не найдена.")

    paths = json.loads(record.line_image_paths_json or "[]")
    if line_index < 0 or line_index >= len(paths):
        raise HTTPException(status_code=404, detail="Строка не найдена.")

    path = Path(paths[line_index])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Файл строки отсутствует.")
    return FileResponse(path)
