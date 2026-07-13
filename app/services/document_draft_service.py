from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.storage.models import DocumentDraft, DocumentRevision, OcrJob, OcrJobPage, RecognitionRecord


class DocumentDraftService:
    def create(
        self,
        db: Session,
        *,
        title: str,
        pages: list[dict[str, Any]],
        document_text: str = "",
        source_job_id: str | None = None,
    ) -> DocumentDraft:
        self._validate_pages(pages)
        draft = DocumentDraft(
            id=str(uuid4()),
            title=title.strip() or "Новый документ",
            source_job_id=source_job_id,
            pages_json=json.dumps(pages, ensure_ascii=False),
            document_text=document_text,
            revision=1,
        )
        db.add(draft)
        db.commit()
        db.refresh(draft)
        return draft

    def create_from_job(
        self,
        db: Session,
        job_id: str,
        *,
        title: str | None = None,
    ) -> DocumentDraft:
        job = db.get(OcrJob, job_id)
        if job is None:
            raise ValueError(f"OCR job was not found: {job_id}")
        job_pages = (
            db.query(OcrJobPage)
            .filter(OcrJobPage.job_id == job_id)
            .order_by(OcrJobPage.page_number)
            .all()
        )
        recognition_ids = [page.recognition_id for page in job_pages if page.recognition_id is not None]
        records = {
            record.id: record
            for record in db.query(RecognitionRecord).filter(RecognitionRecord.id.in_(recognition_ids)).all()
        }
        pages: list[dict[str, Any]] = []
        for job_page in job_pages:
            if job_page.recognition_id is None or job_page.recognition_id not in records:
                continue
            record = records[job_page.recognition_id]
            text = record.corrected_text or record.recognized_text
            pages.append(
                {
                    "page_number": len(pages) + 1,
                    "source_page_number": job_page.page_number,
                    "recognition_id": record.id,
                    "filename": job_page.original_filename,
                    "text": text,
                    "recognized_text": record.recognized_text,
                    "needs_review": bool(record.needs_review),
                    "reliability_score": record.reliability_score,
                    "model_name": record.model_name,
                    "recognition_mode": record.recognition_mode,
                    "processing_time_ms": record.processing_time_ms,
                    "line_coordinates": record.line_coordinates,
                }
            )
        if not pages:
            raise ValueError("OCR job does not contain completed pages.")
        draft = self.create(
            db,
            title=title or job.title,
            source_job_id=job_id,
            pages=pages,
            document_text="\n\n\f\n\n".join(page["text"] for page in pages),
        )
        for record in records.values():
            record.document_id = draft.id
        db.commit()
        return draft

    def get(self, db: Session, draft_id: str) -> DocumentDraft:
        draft = db.get(DocumentDraft, draft_id)
        if draft is None:
            raise ValueError(f"Document draft was not found: {draft_id}")
        return draft

    def list(self, db: Session, limit: int = 50) -> list[DocumentDraft]:
        return db.query(DocumentDraft).order_by(desc(DocumentDraft.updated_at)).limit(min(limit, 200)).all()

    def update(
        self,
        db: Session,
        draft_id: str,
        *,
        expected_revision: int,
        title: str | None = None,
        pages: list[dict[str, Any]] | None = None,
        document_text: str | None = None,
    ) -> DocumentDraft:
        draft = self.get(db, draft_id)
        if draft.revision != expected_revision:
            raise ValueError(
                f"Document was changed by another operation: expected revision {expected_revision}, current {draft.revision}."
            )
        self._snapshot(db, draft)
        if pages is not None:
            self._validate_pages(pages)
            draft.pages_json = json.dumps(pages, ensure_ascii=False)
        if title is not None:
            draft.title = title.strip() or draft.title
        if document_text is not None:
            draft.document_text = document_text
        draft.revision += 1
        draft.updated_at = datetime.now(UTC)
        db.commit()
        db.refresh(draft)
        return draft

    def list_revisions(self, db: Session, draft_id: str, limit: int = 50) -> list[DocumentRevision]:
        self.get(db, draft_id)
        return (
            db.query(DocumentRevision)
            .filter(DocumentRevision.draft_id == draft_id)
            .order_by(desc(DocumentRevision.revision))
            .limit(min(limit, 200))
            .all()
        )

    def restore_revision(
        self,
        db: Session,
        draft_id: str,
        revision: int,
        *,
        expected_revision: int,
    ) -> DocumentDraft:
        draft = self.get(db, draft_id)
        if draft.revision != expected_revision:
            raise ValueError(
                f"Document was changed by another operation: expected revision {expected_revision}, current {draft.revision}."
            )
        source = (
            db.query(DocumentRevision)
            .filter(
                DocumentRevision.draft_id == draft_id,
                DocumentRevision.revision == revision,
            )
            .order_by(desc(DocumentRevision.id))
            .first()
        )
        if source is None:
            raise ValueError(f"Document revision was not found: {revision}")
        self._snapshot(db, draft)
        draft.title = source.title
        draft.pages_json = source.pages_json
        draft.document_text = source.document_text
        draft.revision += 1
        draft.updated_at = datetime.now(UTC)
        db.commit()
        db.refresh(draft)
        return draft

    def delete(self, db: Session, draft_id: str) -> None:
        draft = self.get(db, draft_id)
        db.query(DocumentRevision).filter(DocumentRevision.draft_id == draft_id).delete()
        db.delete(draft)
        db.commit()

    @staticmethod
    def serialize(draft: DocumentDraft) -> dict[str, Any]:
        return {
            "id": draft.id,
            "title": draft.title,
            "source_job_id": draft.source_job_id,
            "pages": draft.pages,
            "document_text": draft.document_text,
            "revision": draft.revision,
            "created_at": draft.created_at,
            "updated_at": draft.updated_at,
        }

    @staticmethod
    def serialize_revision(revision: DocumentRevision) -> dict[str, Any]:
        return {
            "revision": revision.revision,
            "title": revision.title,
            "pages": revision.pages,
            "document_text": revision.document_text,
            "created_at": revision.created_at,
        }

    @staticmethod
    def _snapshot(db: Session, draft: DocumentDraft) -> None:
        db.add(
            DocumentRevision(
                draft_id=draft.id,
                revision=draft.revision,
                title=draft.title,
                pages_json=draft.pages_json,
                document_text=draft.document_text,
            )
        )

    @staticmethod
    def _validate_pages(pages: list[dict[str, Any]]) -> None:
        page_numbers = [int(page.get("page_number", 0)) for page in pages]
        if any(number <= 0 for number in page_numbers):
            raise ValueError("Every document page must have a positive page_number.")
        if len(page_numbers) != len(set(page_numbers)):
            raise ValueError("Document page_number values must be unique.")
