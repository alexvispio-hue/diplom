from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.api.schemas import HealthResponse, ModelInfo, RecognitionHistoryItem, RecognitionResponse
from app.core.config import get_settings
from app.ocr.factory import build_recognizer
from app.services.recognition_service import RecognitionService
from app.storage.database import get_db
from app.storage.models import RecognitionRecord


router = APIRouter()
settings = get_settings()
recognition_service = RecognitionService(build_recognizer())


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", app_name=settings.app_name, version=settings.app_version)


@router.get("/models", response_model=list[ModelInfo])
def models() -> list[ModelInfo]:
    return [
        ModelInfo(
            backend=settings.recognizer_backend,
            model_name=settings.trocr_model_name,
            local_inference=True,
        )
    ]


@router.post("/recognize", response_model=RecognitionResponse)
async def recognize_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> RecognitionResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Загрузите файл изображения.")

    try:
        record = await recognition_service.recognize_upload(file, db)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return RecognitionResponse(
        id=record.id,
        filename=record.original_filename,
        file_size_bytes=record.file_size_bytes,
        recognized_text=record.recognized_text,
        confidence=record.confidence,
        model_name=record.model_name,
        processing_time_ms=record.processing_time_ms,
        created_at=record.created_at,
        original_image_url=f"/api/history/{record.id}/image/original",
        processed_image_url=f"/api/history/{record.id}/image/processed" if record.processed_image_path else None,
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
