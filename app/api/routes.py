import json
from pathlib import Path
from typing import Literal
from urllib.parse import quote

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, Response
from sqlalchemy import desc
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.api.schemas import (
    AcceptanceAuditItem,
    DocumentDraftCreate,
    DocumentDraftResponse,
    DocumentDraftUpdate,
    DocumentFromJobRequest,
    DocumentRestoreRequest,
    DocumentRevisionResponse,
    DictionaryEntryRequest,
    DictionaryResponse,
    DiagnosticsResponse,
    ExperimentSummaryItem,
    FeedbackByIdRequest,
    FeedbackProblemItem,
    FeedbackStatsResponse,
    FolderJobRequest,
    GoldReviewSummary,
    GoldReviewUpdate,
    FeedbackRequest,
    HealthResponse,
    MLEnvironmentStatus,
    ModelActionResponse,
    ModelBackendRequest,
    ModelDownloadResponse,
    ModelInstallRequest,
    ModelInfo,
    OcrJobResponse,
    RegisterModelRequest,
    RegisteredModel,
    PageRecognitionMode,
    QwenProjectStatusResponse,
    RecognitionHistoryItem,
    RecognitionResponse,
)
from app.core.config import get_settings
from app.ocr.factory import build_recognizer
from app.services.experiment_results_service import ExperimentResultsService
from app.services.document_draft_service import DocumentDraftService
from app.services.diagnostics_service import DiagnosticsService
from app.services.document_service import (
    DocumentImportError,
    build_export_pages,
    expand_document,
    export_docx,
    export_json,
    export_searchable_pdf,
    export_text,
    load_image_folder,
)
from app.services.job_service import OcrJobService
from app.services.gold_review_service import GoldReviewService
from app.services.ml_environment_service import QWEN_RUNTIME_PACKAGES, collect_status, missing_packages
from app.services.model_registry_service import ModelRegistryService
from app.services.model_download_service import ModelDownloadService
from app.services.qwen_project_status_service import QwenProjectStatusService
from app.services.user_dictionary_service import UserDictionaryService
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
model_registry_service = ModelRegistryService()
model_download_service = ModelDownloadService()
job_service = OcrJobService()
document_draft_service = DocumentDraftService()
user_dictionary_service = UserDictionaryService()
diagnostics_service = DiagnosticsService()
gold_review_service = GoldReviewService()
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
        reliability_score=record.reliability_score,
        model_name=record.model_name,
        processing_time_ms=record.processing_time_ms,
        peak_vram_bytes=record.peak_vram_bytes,
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
        feedback_training_approved=bool(record.feedback_training_approved),
        feedback_training_excluded=bool(record.feedback_training_excluded),
        original_image_url=f"/api/history/{record.id}/image/original",
        processed_image_url=f"/api/history/{record.id}/image/processed" if record.processed_image_path else None,
        model_registry_id=record.model_registry_id,
        routing_reason=record.routing_reason,
        image_quality=record.image_quality,
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


def _registered_model(model_id: str) -> dict:
    for item in model_registry_service.list_models():
        if item["id"] == model_id:
            return item
    raise HTTPException(status_code=404, detail="Модель не найдена в registry.")


@router.get("/models/installed", response_model=list[RegisteredModel])
def installed_models() -> list[dict]:
    return model_registry_service.list_models()


@router.post("/models/register", response_model=RegisteredModel)
def register_model(payload: RegisterModelRequest) -> dict:
    try:
        model = model_registry_service.register(payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _registered_model(model["id"])


@router.post("/models/install", response_model=ModelDownloadResponse, status_code=status.HTTP_202_ACCEPTED)
def install_model(
    payload: ModelInstallRequest,
    background_tasks: BackgroundTasks,
) -> dict:
    try:
        download = model_download_service.create(
            model_id=payload.id,
            repo_id=payload.repo_id,
            backend=payload.backend,
            revision=payload.revision,
            expected_sha256=payload.expected_sha256,
            stage=payload.stage,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    background_tasks.add_task(model_download_service.run, download["id"])
    return download


@router.get("/models/downloads", response_model=list[ModelDownloadResponse])
def model_downloads() -> list[dict]:
    return model_download_service.list()


@router.get("/models/downloads/{download_id}", response_model=ModelDownloadResponse)
def model_download(download_id: str) -> dict:
    try:
        return model_download_service.get(download_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/models/discover", response_model=list[RegisteredModel])
def discover_models() -> list[dict]:
    try:
        model_registry_service.discover_local_adapters()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return model_registry_service.list_models()


@router.post("/models/{model_id}/promote", response_model=ModelActionResponse)
def promote_model(model_id: str) -> ModelActionResponse:
    try:
        model_registry_service.promote(model_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ModelActionResponse(status="promoted", model=RegisteredModel(**_registered_model(model_id)))


@router.get("/models/{model_id}/verify")
def verify_model(model_id: str) -> dict:
    try:
        return model_registry_service.verify(model_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/models/rollback", response_model=ModelActionResponse)
def rollback_model(payload: ModelBackendRequest) -> ModelActionResponse:
    try:
        model = model_registry_service.rollback(payload.backend)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ModelActionResponse(status="rolled_back", model=RegisteredModel(**_registered_model(model["id"])))


@router.delete("/models/{model_id}", response_model=ModelActionResponse)
def remove_model(model_id: str, delete_files: bool = False) -> ModelActionResponse:
    try:
        model_registry_service.unregister(model_id, delete_files=delete_files)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ModelActionResponse(status="removed")


@router.get("/ml-status", response_model=MLEnvironmentStatus)
def ml_status() -> MLEnvironmentStatus:
    return MLEnvironmentStatus(**collect_status())


@router.get("/diagnostics", response_model=DiagnosticsResponse)
def diagnostics() -> DiagnosticsResponse:
    return DiagnosticsResponse(**diagnostics_service.collect())


@router.get("/qwen-status", response_model=QwenProjectStatusResponse)
def qwen_status() -> QwenProjectStatusResponse:
    return QwenProjectStatusResponse(**qwen_project_status_service.get_status())


@router.get("/acceptance-audit", response_model=list[AcceptanceAuditItem])
def acceptance_audit() -> list[AcceptanceAuditItem]:
    return [AcceptanceAuditItem(**check.__dict__) for check in check_results()]


@router.get("/dictionary", response_model=DictionaryResponse)
def user_dictionary() -> DictionaryResponse:
    return DictionaryResponse(words=user_dictionary_service.list_words())


@router.post("/dictionary", response_model=DictionaryResponse)
def add_dictionary_entry(payload: DictionaryEntryRequest) -> DictionaryResponse:
    try:
        words = user_dictionary_service.add(payload.word)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return DictionaryResponse(words=words)


@router.delete("/dictionary/{word}", response_model=DictionaryResponse)
def remove_dictionary_entry(word: str) -> DictionaryResponse:
    return DictionaryResponse(words=user_dictionary_service.remove(word))


@router.get("/experiments/page-summary", response_model=list[ExperimentSummaryItem])
def page_experiment_summary() -> list:
    return experiment_results_service.list_page_summaries()


@router.get("/gold-review")
def gold_review_items(review_status: str | None = None) -> list[dict]:
    try:
        return gold_review_service.list(status=review_status)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/gold-review/summary", response_model=GoldReviewSummary)
def gold_review_summary() -> GoldReviewSummary:
    return GoldReviewSummary(**gold_review_service.summary())


@router.post("/gold-review/finalize")
def finalize_gold_review() -> dict:
    try:
        return gold_review_service.finalize(
            settings.exports_dir / "benchmarks" / "gold-v1.jsonl"
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.patch("/gold-review/{benchmark_id}")
def update_gold_review(benchmark_id: str, payload: GoldReviewUpdate) -> dict:
    try:
        return gold_review_service.update(
            benchmark_id,
            review_status=payload.review_status,
            reference_text=payload.reference_text,
            reviewer=payload.reviewer,
            review_notes=payload.review_notes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/gold-review/{benchmark_id}/image")
def gold_review_image(benchmark_id: str) -> FileResponse:
    try:
        path = gold_review_service.image_path(benchmark_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path)


@router.post("/jobs", response_model=OcrJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_ocr_job(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    title: str = Form(default="Новый документ"),
    document_target: Literal["fragment", "page"] = Form(default="page"),
    mode: PageRecognitionMode = Form(default=PAGE_MODE_TROCR_LINES),
    preprocess: bool = Form(default=True),
    model: str | None = Form(default=None),
    max_tokens: int = Form(default=1024),
    db: Session = Depends(get_db),
) -> dict:
    _validate_max_tokens(max_tokens)
    if document_target == "page":
        _ensure_qwen_environment_ready(mode)
    pages = []
    try:
        for uploaded_file in files:
            data = await uploaded_file.read()
            pages.extend(
                expand_document(
                    uploaded_file.filename or "document",
                    data,
                    uploaded_file.content_type,
                    pdf_render_dpi=settings.pdf_render_dpi,
                    max_pages=settings.max_document_pages - len(pages),
                    max_file_size_bytes=settings.max_upload_size_mb * 1024 * 1024,
                    max_image_pixels=settings.max_image_pixels,
                )
            )
            if len(pages) > settings.max_document_pages:
                raise DocumentImportError(
                    f"Документ содержит более {settings.max_document_pages} страниц."
                )
        job = job_service.create(
            db,
            pages,
            title=title,
            document_target=document_target,
            recognition_mode=mode,
            preprocessing_applied=preprocess,
            model_override=model,
            max_tokens=max_tokens,
        )
    except (DocumentImportError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    background_tasks.add_task(job_service.process_job, job.id, recognition_service)
    return job_service.serialize(db, job)


@router.get("/jobs", response_model=list[OcrJobResponse])
def list_ocr_jobs(db: Session = Depends(get_db), limit: int = 50) -> list[dict]:
    return [job_service.serialize(db, job) for job in job_service.list(db, limit=limit)]


@router.post("/jobs/folder", response_model=OcrJobResponse, status_code=status.HTTP_202_ACCEPTED)
def create_folder_ocr_job(
    payload: FolderJobRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> dict:
    _validate_max_tokens(payload.max_tokens)
    if payload.document_target == "page":
        _ensure_qwen_environment_ready(payload.mode)
    try:
        pages = load_image_folder(
            payload.folder,
            recursive=payload.recursive,
            max_pages=settings.max_document_pages,
            max_file_size_bytes=settings.max_upload_size_mb * 1024 * 1024,
            max_image_pixels=settings.max_image_pixels,
            sort_by=payload.sort_by,
        )
        job = job_service.create(
            db,
            pages,
            title=payload.title,
            document_target=payload.document_target,
            recognition_mode=payload.mode,
            preprocessing_applied=payload.preprocess,
            model_override=payload.model,
            max_tokens=payload.max_tokens,
        )
    except (DocumentImportError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    background_tasks.add_task(job_service.process_job, job.id, recognition_service)
    return job_service.serialize(db, job)


@router.get("/jobs/{job_id}", response_model=OcrJobResponse)
def get_ocr_job(job_id: str, db: Session = Depends(get_db)) -> dict:
    try:
        return job_service.serialize(db, job_service.get(db, job_id))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/jobs/{job_id}/cancel", response_model=OcrJobResponse)
def cancel_ocr_job(job_id: str, db: Session = Depends(get_db)) -> dict:
    try:
        return job_service.serialize(db, job_service.cancel(db, job_id))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/jobs/{job_id}/pause", response_model=OcrJobResponse)
def pause_ocr_job(job_id: str, db: Session = Depends(get_db)) -> dict:
    try:
        return job_service.serialize(db, job_service.pause(db, job_id))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/jobs/{job_id}/resume", response_model=OcrJobResponse)
def resume_ocr_job(
    job_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> dict:
    try:
        job = job_service.resume(db, job_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    background_tasks.add_task(job_service.process_job, job.id, recognition_service)
    return job_service.serialize(db, job)


@router.post("/jobs/{job_id}/retry", response_model=OcrJobResponse)
def retry_ocr_job(
    job_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> dict:
    try:
        job = job_service.retry(db, job_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    background_tasks.add_task(job_service.process_job, job.id, recognition_service)
    return job_service.serialize(db, job)


@router.post("/jobs/{job_id}/document", response_model=DocumentDraftResponse, status_code=status.HTTP_201_CREATED)
def create_document_from_job(
    job_id: str,
    payload: DocumentFromJobRequest,
    db: Session = Depends(get_db),
) -> dict:
    try:
        draft = document_draft_service.create_from_job(db, job_id, title=payload.title)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return document_draft_service.serialize(draft)


@router.post("/documents", response_model=DocumentDraftResponse, status_code=status.HTTP_201_CREATED)
def create_document_draft(payload: DocumentDraftCreate, db: Session = Depends(get_db)) -> dict:
    try:
        draft = document_draft_service.create(
            db,
            title=payload.title,
            source_job_id=payload.source_job_id,
            pages=payload.pages,
            document_text=payload.document_text,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return document_draft_service.serialize(draft)


@router.get("/documents", response_model=list[DocumentDraftResponse])
def list_document_drafts(db: Session = Depends(get_db), limit: int = 50) -> list[dict]:
    return [document_draft_service.serialize(draft) for draft in document_draft_service.list(db, limit=limit)]


@router.get("/documents/{draft_id}", response_model=DocumentDraftResponse)
def get_document_draft(draft_id: str, db: Session = Depends(get_db)) -> dict:
    try:
        return document_draft_service.serialize(document_draft_service.get(db, draft_id))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/documents/{draft_id}/revisions", response_model=list[DocumentRevisionResponse])
def document_revisions(
    draft_id: str,
    db: Session = Depends(get_db),
    limit: int = 50,
) -> list[dict]:
    try:
        revisions = document_draft_service.list_revisions(db, draft_id, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return [document_draft_service.serialize_revision(revision) for revision in revisions]


@router.post("/documents/{draft_id}/restore/{revision}", response_model=DocumentDraftResponse)
def restore_document_revision(
    draft_id: str,
    revision: int,
    payload: DocumentRestoreRequest,
    db: Session = Depends(get_db),
) -> dict:
    try:
        draft = document_draft_service.restore_revision(
            db,
            draft_id,
            revision,
            expected_revision=payload.expected_revision,
        )
    except ValueError as exc:
        code = 409 if "revision" in str(exc) and "not found" not in str(exc) else 400
        raise HTTPException(status_code=code, detail=str(exc)) from exc
    return document_draft_service.serialize(draft)


@router.patch("/documents/{draft_id}", response_model=DocumentDraftResponse)
def update_document_draft(
    draft_id: str,
    payload: DocumentDraftUpdate,
    db: Session = Depends(get_db),
) -> dict:
    try:
        draft = document_draft_service.update(
            db,
            draft_id,
            expected_revision=payload.expected_revision,
            title=payload.title,
            pages=payload.pages,
            document_text=payload.document_text,
        )
    except ValueError as exc:
        code = 409 if "revision" in str(exc) else 400
        raise HTTPException(status_code=code, detail=str(exc)) from exc
    return document_draft_service.serialize(draft)


@router.get("/documents/{draft_id}/export")
def export_document_draft(
    draft_id: str,
    format: Literal["txt", "json", "docx", "pdf"] = "docx",
    db: Session = Depends(get_db),
) -> Response:
    try:
        draft = document_draft_service.get(db, draft_id)
        results, page_images = _document_export_data(db, draft.pages)
        pages = build_export_pages(results)
        if format == "txt":
            content = export_text(pages).encode("utf-8-sig")
            media_type = "text/plain; charset=utf-8"
        elif format == "json":
            content = export_json(pages, title=draft.title)
            media_type = "application/json"
        elif format == "docx":
            content = export_docx(pages, title=draft.title)
            media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        else:
            content = export_searchable_pdf(pages, page_images)
            media_type = "application/pdf"
    except (DocumentImportError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    filename = quote(f"{draft.title}.{format}")
    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{filename}"},
    )


@router.delete("/documents/{draft_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document_draft(draft_id: str, db: Session = Depends(get_db)) -> None:
    try:
        document_draft_service.delete(db, draft_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _document_export_data(
    db: Session,
    draft_pages: list[dict],
) -> tuple[list[dict], dict[int, bytes]]:
    ordered_pages = sorted(draft_pages, key=lambda page: int(page.get("page_number", 0)))
    recognition_ids = [
        int(page["recognition_id"])
        for page in ordered_pages
        if page.get("recognition_id") is not None
    ]
    records = {
        record.id: record
        for record in db.query(RecognitionRecord).filter(RecognitionRecord.id.in_(recognition_ids)).all()
    }
    results: list[dict] = []
    images: dict[int, bytes] = {}
    for index, draft_page in enumerate(ordered_pages, start=1):
        recognition_id = draft_page.get("recognition_id")
        record = records.get(int(recognition_id)) if recognition_id is not None else None
        merged = {
            "page_number": int(draft_page.get("page_number") or index),
            "recognition_id": recognition_id,
            "filename": draft_page.get("filename") or (record.original_filename if record else f"page_{index}.png"),
            "recognized_text": record.recognized_text if record else "",
            "corrected_text": record.corrected_text if record else None,
            "model_name": record.model_name if record else None,
            "recognition_mode": record.recognition_mode if record else None,
            "processing_time_ms": record.processing_time_ms if record else None,
            "needs_review": bool(record.needs_review) if record else False,
            "line_coordinates": record.line_coordinates if record else [],
        }
        merged.update(draft_page)
        results.append(merged)
        if record is not None:
            image_path = Path(record.processed_image_path or record.stored_image_path)
            if image_path.is_file():
                images[merged["page_number"]] = image_path.read_bytes()
    return results, images


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
        .limit(min(limit, 500))
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
        return feedback_service.save(
            record,
            feedback.rating,
            db,
            corrected_text=feedback.corrected_text,
            include_for_training=feedback.include_for_training,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/feedback/{record_id}/exclude", response_model=RecognitionHistoryItem)
def exclude_feedback_from_training(
    record_id: int,
    db: Session = Depends(get_db),
) -> RecognitionRecord:
    record = db.get(RecognitionRecord, record_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Запись не найдена.")
    return feedback_service.exclude_from_training(record, db)


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
