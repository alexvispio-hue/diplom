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


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    _run_lightweight_migrations()


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _run_lightweight_migrations() -> None:
    inspector = inspect(engine)
    if "recognition_records" not in inspector.get_table_names():
        return

    columns = {column["name"] for column in inspector.get_columns("recognition_records")}
    with engine.begin() as connection:
        if "file_size_bytes" not in columns:
            connection.execute(
                text("ALTER TABLE recognition_records ADD COLUMN file_size_bytes INTEGER NOT NULL DEFAULT 0")
            )
        if "preprocessing_applied" not in columns:
            connection.execute(
                text("ALTER TABLE recognition_records ADD COLUMN preprocessing_applied BOOLEAN NOT NULL DEFAULT 1")
            )
