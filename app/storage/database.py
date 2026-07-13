from collections.abc import Generator

from sqlalchemy import inspect, text
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings


class Base(DeclarativeBase):
    pass


settings = get_settings()
engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
CURRENT_SCHEMA_VERSION = 2


def init_db() -> None:
    from app.storage import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    _run_lightweight_migrations()


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _run_lightweight_migrations(target_engine=engine) -> None:
    inspector = inspect(target_engine)
    if "recognition_records" not in inspector.get_table_names():
        return

    columns = {column["name"] for column in inspector.get_columns("recognition_records")}
    with target_engine.begin() as connection:
        if "file_size_bytes" not in columns:
            connection.execute(
                text("ALTER TABLE recognition_records ADD COLUMN file_size_bytes INTEGER NOT NULL DEFAULT 0")
            )
        if "preprocessing_applied" not in columns:
            connection.execute(
                text("ALTER TABLE recognition_records ADD COLUMN preprocessing_applied BOOLEAN NOT NULL DEFAULT 1")
            )
        if "recognition_mode" not in columns:
            connection.execute(
                text("ALTER TABLE recognition_records ADD COLUMN recognition_mode VARCHAR(32) NOT NULL DEFAULT 'fragment'")
            )
        if "line_count" not in columns:
            connection.execute(
                text("ALTER TABLE recognition_records ADD COLUMN line_count INTEGER NOT NULL DEFAULT 1")
            )
        if "line_image_paths_json" not in columns:
            connection.execute(
                text("ALTER TABLE recognition_records ADD COLUMN line_image_paths_json TEXT")
            )
        if "line_bboxes_json" not in columns:
            connection.execute(
                text("ALTER TABLE recognition_records ADD COLUMN line_bboxes_json TEXT")
            )
        if "feedback_rating" not in columns:
            connection.execute(
                text("ALTER TABLE recognition_records ADD COLUMN feedback_rating VARCHAR(32)")
            )
        if "needs_review" not in columns:
            connection.execute(
                text("ALTER TABLE recognition_records ADD COLUMN needs_review BOOLEAN NOT NULL DEFAULT 0")
            )
        if "review_notes_json" not in columns:
            connection.execute(
                text("ALTER TABLE recognition_records ADD COLUMN review_notes_json TEXT")
            )
        if "corrected_text" not in columns:
            connection.execute(
                text("ALTER TABLE recognition_records ADD COLUMN corrected_text TEXT")
            )
        if "feedback_sample_path" not in columns:
            connection.execute(
                text("ALTER TABLE recognition_records ADD COLUMN feedback_sample_path VARCHAR(512)")
            )
        if "feedback_created_at" not in columns:
            connection.execute(
                text("ALTER TABLE recognition_records ADD COLUMN feedback_created_at DATETIME")
            )
        if "feedback_training_approved" not in columns:
            connection.execute(
                text(
                    "ALTER TABLE recognition_records ADD COLUMN "
                    "feedback_training_approved BOOLEAN NOT NULL DEFAULT 0"
                )
            )
        if "feedback_training_excluded" not in columns:
            connection.execute(
                text(
                    "ALTER TABLE recognition_records ADD COLUMN "
                    "feedback_training_excluded BOOLEAN NOT NULL DEFAULT 0"
                )
            )
        if "model_registry_id" not in columns:
            connection.execute(text("ALTER TABLE recognition_records ADD COLUMN model_registry_id VARCHAR(128)"))
        if "job_id" not in columns:
            connection.execute(text("ALTER TABLE recognition_records ADD COLUMN job_id VARCHAR(36)"))
        if "document_id" not in columns:
            connection.execute(text("ALTER TABLE recognition_records ADD COLUMN document_id VARCHAR(36)"))
        if "document_page_number" not in columns:
            connection.execute(text("ALTER TABLE recognition_records ADD COLUMN document_page_number INTEGER"))
        if "reliability_score" not in columns:
            connection.execute(text("ALTER TABLE recognition_records ADD COLUMN reliability_score FLOAT"))
        if "routing_reason" not in columns:
            connection.execute(text("ALTER TABLE recognition_records ADD COLUMN routing_reason TEXT"))
        if "image_quality_json" not in columns:
            connection.execute(text("ALTER TABLE recognition_records ADD COLUMN image_quality_json TEXT"))
        if "peak_vram_bytes" not in columns:
            connection.execute(text("ALTER TABLE recognition_records ADD COLUMN peak_vram_bytes INTEGER"))

    inspector = inspect(target_engine)
    if "ocr_jobs" in inspector.get_table_names():
        job_columns = {column["name"] for column in inspector.get_columns("ocr_jobs")}
        job_migrations = {
            "document_target": "ALTER TABLE ocr_jobs ADD COLUMN document_target VARCHAR(32) NOT NULL DEFAULT 'page'",
            "recognition_mode": "ALTER TABLE ocr_jobs ADD COLUMN recognition_mode VARCHAR(32) NOT NULL DEFAULT 'trocr_lines'",
            "preprocessing_applied": "ALTER TABLE ocr_jobs ADD COLUMN preprocessing_applied BOOLEAN NOT NULL DEFAULT 1",
            "model_override": "ALTER TABLE ocr_jobs ADD COLUMN model_override VARCHAR(255)",
            "max_tokens": "ALTER TABLE ocr_jobs ADD COLUMN max_tokens INTEGER NOT NULL DEFAULT 1024",
            "total_pages": "ALTER TABLE ocr_jobs ADD COLUMN total_pages INTEGER NOT NULL DEFAULT 0",
            "completed_pages": "ALTER TABLE ocr_jobs ADD COLUMN completed_pages INTEGER NOT NULL DEFAULT 0",
            "failed_pages": "ALTER TABLE ocr_jobs ADD COLUMN failed_pages INTEGER NOT NULL DEFAULT 0",
            "pause_requested": "ALTER TABLE ocr_jobs ADD COLUMN pause_requested BOOLEAN NOT NULL DEFAULT 0",
            "error": "ALTER TABLE ocr_jobs ADD COLUMN error TEXT",
            "created_at": "ALTER TABLE ocr_jobs ADD COLUMN created_at DATETIME",
            "updated_at": "ALTER TABLE ocr_jobs ADD COLUMN updated_at DATETIME",
        }
        with target_engine.begin() as connection:
            for column, statement in job_migrations.items():
                if column not in job_columns:
                    connection.execute(
                        text(statement)
                    )

    inspector = inspect(target_engine)
    if "schema_migrations" in inspector.get_table_names():
        with target_engine.begin() as connection:
            applied = {
                int(row[0])
                for row in connection.execute(text("SELECT version FROM schema_migrations")).fetchall()
            }
            for version in range(1, CURRENT_SCHEMA_VERSION + 1):
                if version not in applied:
                    connection.execute(
                        text("INSERT INTO schema_migrations (version, applied_at) VALUES (:version, CURRENT_TIMESTAMP)"),
                        {"version": version},
                    )


def schema_version(target_engine=engine) -> int:
    inspector = inspect(target_engine)
    if "schema_migrations" not in inspector.get_table_names():
        return 0
    with target_engine.connect() as connection:
        value = connection.execute(text("SELECT MAX(version) FROM schema_migrations")).scalar()
    return int(value or 0)
