import json
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.api.schemas import FeedbackRequest, HealthResponse, ModelInfo, RecognitionHistoryItem, RecognitionResponse
from app.core.config import get_settings
from app.ocr.factory import build_recognizer
from app.services.recognition_service import RecognitionService
from app.services.feedback_service import FeedbackService
from app.storage.database import get_db
from app.storage.models import RecognitionRecord


router = APIRouter()
settings = get_settings()
recognition_service = RecognitionService(build_recognizer())
feedback_service = FeedbackService()


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
        preprocessing_applied=record.preprocessing_applied,
        recognition_mode=record.recognition_mode,
        line_count=record.line_count,
        line_image_urls=record.line_image_urls,
        feedback_rating=record.feedback_rating,
        feedback_saved_for_training=record.feedback_saved_for_training,
        original_image_url=f"/api/history/{record.id}/image/original",
        processed_image_url=f"/api/history/{record.id}/image/processed" if record.processed_image_path else None,
    )


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", app_name=settings.app_name, version=settings.app_version)


@router.get("/models", response_model=list[ModelInfo])
def models() -> list[ModelInfo]:
    return [
        ModelInfo(
            backend=settings.recognizer_backend,
            model_name=recognition_service.recognizer.model_name,
            local_inference=True,
        )
    ]


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


@router.post("/recognize-page", response_model=RecognitionResponse)
async def recognize_page(file: UploadFile = File(...), db: Session = Depends(get_db)) -> RecognitionResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Загрузите файл изображения.")

    try:
        record = await recognition_service.recognize_page_upload(file, db)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return _build_recognition_response(record)


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
    record = db.get(RecognitionRecord, record_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Запись не найдена.")
    try:
        return feedback_service.save(record, feedback.rating, db)
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
