from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import desc
from sqlalchemy.orm import Session
from starlette.datastructures import Headers, UploadFile

from app.core.config import get_settings
from app.services.document_service import PreparedPage
from app.services.recognition_service import RecognitionService
from app.storage.database import SessionLocal
from app.storage.models import OcrJob, OcrJobPage, RecognitionRecord


ACTIVE_JOB_STATUSES = {"queued", "running", "paused"}
TERMINAL_JOB_STATUSES = {"completed", "failed", "cancelled"}


class OcrJobService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._processing_lock: asyncio.Lock | None = None

    def create(
        self,
        db: Session,
        pages: list[PreparedPage],
        *,
        title: str,
        document_target: str,
        recognition_mode: str,
        preprocessing_applied: bool,
        model_override: str | None,
        max_tokens: int,
    ) -> OcrJob:
        if not pages:
            raise ValueError("OCR job must contain at least one page.")
        if len(pages) > self.settings.max_document_pages:
            raise ValueError(f"Document contains more than {self.settings.max_document_pages} pages.")

        job_id = str(uuid4())
        input_dir = self.settings.jobs_dir / job_id / "input"
        input_dir.mkdir(parents=True, exist_ok=False)
        job = OcrJob(
            id=job_id,
            title=title.strip() or "Новый документ",
            status="queued",
            document_target=document_target,
            recognition_mode=recognition_mode,
            preprocessing_applied=preprocessing_applied,
            model_override=model_override or None,
            max_tokens=max_tokens,
            total_pages=len(pages),
        )
        db.add(job)
        for index, page in enumerate(pages, start=1):
            suffix = Path(page.filename).suffix.lower() or ".png"
            input_path = input_dir / f"page_{index:04d}{suffix}"
            input_path.write_bytes(page.data)
            db.add(
                OcrJobPage(
                    job_id=job_id,
                    page_number=index,
                    original_filename=page.filename,
                    input_path=str(input_path),
                    content_type=page.content_type,
                    status="queued",
                )
            )
        db.commit()
        db.refresh(job)
        return job

    def get(self, db: Session, job_id: str) -> OcrJob:
        job = db.get(OcrJob, job_id)
        if job is None:
            raise ValueError(f"OCR job was not found: {job_id}")
        return job

    def list(self, db: Session, limit: int = 50) -> list[OcrJob]:
        return db.query(OcrJob).order_by(desc(OcrJob.created_at)).limit(min(limit, 200)).all()

    def pages(self, db: Session, job_id: str) -> list[OcrJobPage]:
        self.get(db, job_id)
        return (
            db.query(OcrJobPage)
            .filter(OcrJobPage.job_id == job_id)
            .order_by(OcrJobPage.page_number)
            .all()
        )

    def cancel(self, db: Session, job_id: str) -> OcrJob:
        job = self.get(db, job_id)
        if job.status in TERMINAL_JOB_STATUSES:
            return job
        job.cancel_requested = True
        job.pause_requested = False
        if job.status in {"queued", "paused"}:
            job.status = "cancelled"
            for page in self.pages(db, job_id):
                if page.status == "queued":
                    page.status = "cancelled"
        job.updated_at = datetime.now(UTC)
        db.commit()
        db.refresh(job)
        return job

    def pause(self, db: Session, job_id: str) -> OcrJob:
        job = self.get(db, job_id)
        if job.status in TERMINAL_JOB_STATUSES:
            return job
        job.pause_requested = True
        if job.status == "queued":
            job.status = "paused"
        job.updated_at = datetime.now(UTC)
        db.commit()
        db.refresh(job)
        return job

    def resume(self, db: Session, job_id: str) -> OcrJob:
        job = self.get(db, job_id)
        if job.status in TERMINAL_JOB_STATUSES:
            raise ValueError("Completed, failed or cancelled OCR job cannot be resumed.")
        job.pause_requested = False
        if job.status == "paused":
            job.status = "queued"
        job.updated_at = datetime.now(UTC)
        db.commit()
        db.refresh(job)
        return job

    def retry(self, db: Session, job_id: str) -> OcrJob:
        job = self.get(db, job_id)
        pages = self.pages(db, job_id)
        retry_pages = [page for page in pages if page.status in {"failed", "cancelled"}]
        if not retry_pages and job.status != "failed":
            raise ValueError("OCR job does not contain failed or cancelled pages.")
        for page in retry_pages:
            page.status = "queued"
            page.error = None
            page.updated_at = datetime.now(UTC)
        job.status = "queued"
        job.cancel_requested = False
        job.pause_requested = False
        job.error = None
        self._refresh_counts(db, job)
        db.commit()
        db.refresh(job)
        return job

    def recover_interrupted(self, db: Session) -> int:
        jobs = db.query(OcrJob).filter(OcrJob.status == "running").all()
        for job in jobs:
            job.status = "paused" if job.pause_requested else "queued"
            job.error = "Recovered after application restart."
            job.updated_at = datetime.now(UTC)
            for page in self.pages(db, job.id):
                if page.status == "running":
                    page.status = "queued"
                    page.error = None
        db.commit()
        return len(jobs)

    async def process_pending(self, recognition_service: RecognitionService) -> None:
        lock = self._lock()
        async with lock:
            while True:
                with SessionLocal() as db:
                    job = (
                        db.query(OcrJob)
                        .filter(OcrJob.status == "queued")
                        .order_by(OcrJob.created_at)
                        .first()
                    )
                    if job is None:
                        return
                    job_id = job.id
                await self._process_job_unlocked(job_id, recognition_service)

    async def process_job(self, job_id: str, recognition_service: RecognitionService) -> None:
        lock = self._lock()
        async with lock:
            await self._process_job_unlocked(job_id, recognition_service)

    async def _process_job_unlocked(self, job_id: str, recognition_service: RecognitionService) -> None:
        with SessionLocal() as db:
            job = self.get(db, job_id)
            if job.status != "queued":
                return
            if job.pause_requested:
                job.status = "paused"
                job.updated_at = datetime.now(UTC)
                db.commit()
                return
            job.status = "running"
            job.updated_at = datetime.now(UTC)
            db.commit()

        page_ids: list[int]
        with SessionLocal() as db:
            page_ids = [page.id for page in self.pages(db, job_id) if page.status == "queued"]

        for page_id in page_ids:
            with SessionLocal() as db:
                job = self.get(db, job_id)
                page = db.get(OcrJobPage, page_id)
                if page is None:
                    continue
                if job.pause_requested:
                    job.status = "paused"
                    job.updated_at = datetime.now(UTC)
                    db.commit()
                    return
                if job.cancel_requested:
                    page.status = "cancelled"
                    page.updated_at = datetime.now(UTC)
                    db.commit()
                    continue
                page.status = "running"
                page.updated_at = datetime.now(UTC)
                db.commit()

                try:
                    input_path = Path(page.input_path)
                    if not input_path.exists():
                        raise FileNotFoundError(f"Job input page was not found: {input_path}")
                    with input_path.open("rb") as source:
                        upload = UploadFile(
                            file=source,
                            filename=page.original_filename,
                            headers=Headers({"content-type": page.content_type}),
                        )
                        if job.document_target == "fragment":
                            record = await recognition_service.recognize_upload(
                                upload,
                                db,
                                preprocessing_applied=job.preprocessing_applied,
                            )
                        else:
                            record = await recognition_service.recognize_page_upload(
                                upload,
                                db,
                                mode=job.recognition_mode,
                                preprocessing_applied=job.preprocessing_applied,
                                model_name=job.model_override,
                                max_tokens=job.max_tokens,
                                save_intermediate_images=True,
                                return_coordinates=True,
                            )
                    record.job_id = job_id
                    record.document_page_number = page.page_number
                    page.recognition_id = record.id
                    page.status = "completed"
                    page.error = None
                except Exception as exc:
                    page.status = "failed"
                    page.error = str(exc)
                page.updated_at = datetime.now(UTC)
                db.commit()

        with SessionLocal() as db:
            job = self.get(db, job_id)
            self._refresh_counts(db, job)
            if job.cancel_requested:
                job.status = "cancelled"
            elif job.completed_pages == 0 and job.failed_pages:
                job.status = "failed"
            else:
                job.status = "completed"
            job.updated_at = datetime.now(UTC)
            db.commit()

    def serialize(self, db: Session, job: OcrJob) -> dict[str, Any]:
        job_pages = self.pages(db, job.id)
        recognition_ids = [page.recognition_id for page in job_pages if page.recognition_id is not None]
        records = {
            record.id: record
            for record in db.query(RecognitionRecord).filter(RecognitionRecord.id.in_(recognition_ids)).all()
        }
        return {
            "id": job.id,
            "title": job.title,
            "status": job.status,
            "document_target": job.document_target,
            "recognition_mode": job.recognition_mode,
            "preprocessing_applied": job.preprocessing_applied,
            "model_override": job.model_override,
            "max_tokens": job.max_tokens,
            "total_pages": job.total_pages,
            "completed_pages": job.completed_pages,
            "failed_pages": job.failed_pages,
            "cancel_requested": job.cancel_requested,
            "pause_requested": job.pause_requested,
            "error": job.error,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
            "pages": [
                self._serialize_page(page, records.get(page.recognition_id))
                for page in job_pages
            ],
        }

    @staticmethod
    def _serialize_page(page: OcrJobPage, record: RecognitionRecord | None) -> dict[str, Any]:
        payload = {
            "id": page.id,
            "page_number": page.page_number,
            "original_filename": page.original_filename,
            "status": page.status,
            "recognition_id": page.recognition_id,
            "error": page.error,
        }
        if record is not None:
            payload.update(
                {
                    "recognized_text": record.recognized_text,
                    "corrected_text": record.corrected_text,
                    "reliability_score": record.reliability_score,
                    "needs_review": bool(record.needs_review),
                    "model_name": record.model_name,
                    "recognition_mode": record.recognition_mode,
                    "routing_reason": record.routing_reason,
                    "image_quality": record.image_quality,
                    "peak_vram_bytes": record.peak_vram_bytes,
                }
            )
        return payload

    def _refresh_counts(self, db: Session, job: OcrJob) -> None:
        pages = self.pages(db, job.id)
        job.completed_pages = sum(page.status == "completed" for page in pages)
        job.failed_pages = sum(page.status == "failed" for page in pages)

    def _lock(self) -> asyncio.Lock:
        if self._processing_lock is None:
            self._processing_lock = asyncio.Lock()
        return self._processing_lock
