from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.services.document_draft_service import DocumentDraftService
from app.storage.database import Base


def test_document_draft_uses_optimistic_revision(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'drafts.db'}")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    service = DocumentDraftService()
    draft = service.create(
        db,
        title="Конспект",
        pages=[{"page_number": 1, "recognition_id": 10, "text": "текст"}],
    )

    updated = service.update(
        db,
        draft.id,
        expected_revision=1,
        pages=[{"page_number": 1, "recognition_id": 10, "text": "исправлено"}],
        document_text="исправлено",
    )

    assert updated.revision == 2
    assert updated.pages[0]["text"] == "исправлено"
    with pytest.raises(ValueError, match="current 2"):
        service.update(db, draft.id, expected_revision=1, document_text="старое")
