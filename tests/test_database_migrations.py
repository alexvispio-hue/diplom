from sqlalchemy import create_engine, inspect, text

from app.storage.database import Base, CURRENT_SCHEMA_VERSION, _run_lightweight_migrations, schema_version
from app.storage import models  # noqa: F401


def test_migration_updates_previous_recognition_and_job_schema(tmp_path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'legacy.db'}")
    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE recognition_records ("
                "id INTEGER PRIMARY KEY, original_filename VARCHAR(255) NOT NULL, "
                "stored_image_path VARCHAR(512) NOT NULL, recognized_text TEXT NOT NULL, "
                "model_name VARCHAR(128) NOT NULL, processing_time_ms INTEGER NOT NULL, "
                "created_at DATETIME NOT NULL)"
            )
        )
        connection.execute(
            text(
                "CREATE TABLE ocr_jobs ("
                "id VARCHAR(36) PRIMARY KEY, title VARCHAR(255) NOT NULL, "
                "status VARCHAR(32) NOT NULL, cancel_requested BOOLEAN NOT NULL DEFAULT 0)"
            )
        )
    Base.metadata.create_all(engine)

    _run_lightweight_migrations(engine)

    recognition_columns = {item["name"] for item in inspect(engine).get_columns("recognition_records")}
    job_columns = {item["name"] for item in inspect(engine).get_columns("ocr_jobs")}
    assert "feedback_training_approved" in recognition_columns
    assert "model_registry_id" in recognition_columns
    assert "pause_requested" in job_columns
    assert schema_version(engine) == CURRENT_SCHEMA_VERSION
