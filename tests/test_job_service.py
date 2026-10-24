from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.services.document_service import PreparedPage
from app.services.job_service import OcrJobService
from app.storage.database import Base


def _session(tmp_path: Path):
    engine = create_engine(f"sqlite:///{tmp_path / 'jobs.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_job_create_cancel_retry_and_recovery(tmp_path: Path, monkeypatch) -> None:
    service = OcrJobService()
    service.settings.jobs_dir = tmp_path / "runtime" / "jobs"
    db = _session(tmp_path)
    page = PreparedPage("doc.pdf", 1, "page.png", "image/png", b"image")

    job = service.create(
        db,
        [page],
        title="Doc",
        document_target="page",
        recognition_mode="trocr_lines",
        preprocessing_applied=True,
        model_override=None,
        max_tokens=128,
    )
    assert job.status == "queued"
    assert Path(service.pages(db, job.id)[0].input_path).read_bytes() == b"image"

    paused = service.pause(db, job.id)
    assert paused.status == "paused"
    resumed = service.resume(db, job.id)
    assert resumed.status == "queued"

    cancelled = service.cancel(db, job.id)
    assert cancelled.status == "cancelled"
    retried = service.retry(db, job.id)
    assert retried.status == "queued"

    retried.status = "running"
    service.pages(db, job.id)[0].status = "running"
    db.commit()
    assert service.recover_interrupted(db) == 1
    assert service.get(db, job.id).status == "queued"
    assert service.pages(db, job.id)[0].status == "queued"
