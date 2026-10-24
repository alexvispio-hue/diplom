import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import job_service, recognition_service, router
from app.core.config import get_settings
from app.storage.database import SessionLocal, init_db


settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    init_db()
    with SessionLocal() as db:
        job_service.recover_interrupted(db)
    recovery_task = asyncio.create_task(job_service.process_pending(recognition_service))
    try:
        yield
    finally:
        if not recovery_task.done():
            recovery_task.cancel()
            try:
                await recovery_task
            except asyncio.CancelledError:
                pass


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)
    app.include_router(router, prefix="/api")
    return app


app = create_app()
