import json
import types
from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api import routes
from app.main import app
from app.services.experiment_results_service import ExperimentSummaryRow
from app.storage.database import Base, get_db
from app.storage.models import RecognitionRecord


def test_health_endpoint() -> None:
    client = TestClient(app)

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_models_endpoint() -> None:
    client = TestClient(app)

    response = client.get("/api/models")

    assert response.status_code == 200
    models = response.json()
    assert {model["backend"] for model in models} == {"trocr", "qwen"}
    assert all(model["local_inference"] is True for model in models)
    assert all(model["model_name"] for model in models)


def test_ml_status_endpoint() -> None:
    client = TestClient(app)

    response = client.get("/api/ml-status")

    assert response.status_code == 200
    status = response.json()
    assert "packages" in status
    assert {package["package"] for package in status["packages"]} >= {"torch", "transformers", "peft"}
    assert isinstance(status["ready_for_trocr"], bool)
    assert isinstance(status["ready_for_qwen_inference"], bool)
    assert isinstance(status["ready_for_qwen_lora"], bool)


def test_qwen_status_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(
        routes,
        "qwen_project_status_service",
        types.SimpleNamespace(
            get_status=lambda: {
                "generated_at": "2026-06-29T00:00:00+00:00",
                "model": "Qwen/test",
                "install_profile": "cuda-cu130",
                "runtime_status": "blocked",
                "missing_packages": ["torch"],
                "recommended_next_command": "python scripts/run_qwen_execution_plan.py --step install_ml --allow-network",
                "sft_dataset": {"path": "data/training/qwen_sft_light/mixed.jsonl", "examples": 3},
                "steps": {"build_sft": "passed", "lora_dry_run": "passed"},
                "lora_dry_run_passed": True,
                "acceptance": {"passed": 13, "partial": 4, "failed": 0},
            }
        ),
    )
    client = TestClient(app)

    response = client.get("/api/qwen-status")

    assert response.status_code == 200
    body = response.json()
    assert body["generated_at"] == "2026-06-29T00:00:00+00:00"
    assert body["model"] == "Qwen/test"
    assert body["sft_dataset"]["examples"] == 3
    assert body["steps"]["lora_dry_run"] == "passed"
    assert body["acceptance"]["partial"] == 4


def test_acceptance_audit_endpoint() -> None:
    client = TestClient(app)

    response = client.get("/api/acceptance-audit")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 17
    assert {item["id"] for item in body} == set(range(1, 18))
    assert not [item for item in body if item["status"] == "partial"]
    assert "3 examples" in next(item for item in body if item["id"] == 13)["notes"]


def test_page_experiment_summary_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(
        routes,
        "experiment_results_service",
        types.SimpleNamespace(
            list_page_summaries=lambda: [
                ExperimentSummaryRow(
                    summary_path="page_experiment/summary.csv",
                    model="stub",
                    mode="trocr_lines",
                    dataset="pages.csv",
                    cer=0.1,
                    wer=0.2,
                    page_time=42.0,
                    line_time=21.0,
                    notes="items=1",
                )
            ]
        ),
    )
    client = TestClient(app)

    response = client.get("/api/experiments/page-summary")

    assert response.status_code == 200
    body = response.json()
    assert body[0]["model"] == "stub"
    assert body[0]["mode"] == "trocr_lines"
    assert body[0]["cer"] == 0.1


def test_qwen_model_display_name_includes_lora_adapter(monkeypatch) -> None:
    monkeypatch.setattr(
        routes,
        "settings",
        types.SimpleNamespace(
            qwen_model_name="Qwen/Qwen3-VL-2B-Instruct",
            qwen_lora_adapter_dir=Path("models/qwen-vl-ocr-lora"),
        ),
    )

    assert routes._qwen_model_display_name() == "Qwen/Qwen3-VL-2B-Instruct+lora:qwen-vl-ocr-lora"


def test_recognize_rejects_non_image_file() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/recognize",
        files={"file": ("note.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 400


def test_recognize_page_rejects_non_image_file() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/recognize-page",
        files={"file": ("note.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 400


def test_recognize_page_tz_route_rejects_non_image_file() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/recognize/page",
        files={"file": ("note.txt", b"hello", "text/plain")},
        data={"mode": "trocr_lines"},
    )

    assert response.status_code == 400


def test_recognize_page_qwen_route_uses_qwen_page_mode(monkeypatch) -> None:
    captured = {}

    async def fake_recognize_page_impl(file, db, mode="trocr_lines", preprocess=True, **kwargs):
        captured["mode"] = mode
        captured["preprocess"] = preprocess
        captured["kwargs"] = kwargs
        return {
            "id": 1,
            "filename": file.filename,
            "file_size_bytes": 5,
            "recognized_text": "page text",
            "confidence": None,
            "model_name": "qwen",
            "processing_time_ms": 10,
            "created_at": datetime.now(UTC),
            "preprocessing_applied": preprocess,
            "recognition_mode": mode,
            "line_count": 0,
            "line_image_urls": [],
            "feedback_rating": None,
            "corrected_text": None,
            "feedback_saved_for_training": False,
            "original_image_url": "/api/history/1/image/original",
            "processed_image_url": None,
        }

    monkeypatch.setattr(routes, "_recognize_page_impl", fake_recognize_page_impl)
    client = TestClient(app)

    response = client.post(
        "/api/recognize/page/qwen",
        files={"file": ("page.png", b"image", "image/png")},
        data={"preprocess": "false"},
    )

    assert response.status_code == 200
    assert captured == {"mode": "qwen_page", "preprocess": False, "kwargs": {"max_tokens": None}}
    assert response.json()["recognition_mode"] == "qwen_page"


def test_recognize_page_compare_route_uses_compare_mode(monkeypatch) -> None:
    captured = {}

    async def fake_recognize_page_impl(file, db, mode="trocr_lines", preprocess=True, **kwargs):
        captured["mode"] = mode
        captured["preprocess"] = preprocess
        captured["kwargs"] = kwargs
        return {
            "id": 1,
            "filename": file.filename,
            "file_size_bytes": 5,
            "recognized_text": "comparison",
            "confidence": None,
            "model_name": "trocr + qwen",
            "processing_time_ms": 10,
            "created_at": datetime.now(UTC),
            "preprocessing_applied": preprocess,
            "recognition_mode": mode,
            "line_count": 0,
            "line_image_urls": [],
            "feedback_rating": None,
            "corrected_text": None,
            "feedback_saved_for_training": False,
            "original_image_url": "/api/history/1/image/original",
            "processed_image_url": None,
        }

    monkeypatch.setattr(routes, "_recognize_page_impl", fake_recognize_page_impl)
    client = TestClient(app)

    response = client.post(
        "/api/recognize/page/compare",
        files={"file": ("page.png", b"image", "image/png")},
        data={"preprocess": "false", "max_tokens": "512"},
    )

    assert response.status_code == 200
    assert captured == {"mode": "compare", "preprocess": False, "kwargs": {"max_tokens": 512}}
    assert response.json()["recognition_mode"] == "compare"


def test_recognize_page_route_passes_tz_parameters(monkeypatch) -> None:
    captured = {}

    async def fake_recognize_page_impl(file, db, mode="trocr_lines", preprocess=True, **kwargs):
        captured["mode"] = mode
        captured["preprocess"] = preprocess
        captured["kwargs"] = kwargs
        return {
            "id": 1,
            "filename": file.filename,
            "file_size_bytes": 5,
            "recognized_text": "page",
            "confidence": None,
            "model_name": "stub",
            "processing_time_ms": 10,
            "created_at": datetime.now(UTC),
            "preprocessing_applied": preprocess,
            "recognition_mode": mode,
            "line_count": 0,
            "line_image_urls": [],
            "feedback_rating": None,
            "corrected_text": None,
            "feedback_saved_for_training": False,
            "original_image_url": "/api/history/1/image/original",
            "processed_image_url": None,
        }

    monkeypatch.setattr(routes, "_recognize_page_impl", fake_recognize_page_impl)
    client = TestClient(app)

    response = client.post(
        "/api/recognize/page",
        files={"file": ("page.png", b"image", "image/png")},
        data={
            "mode": "trocr_lines",
            "preprocess": "false",
            "model": "custom/model",
            "max_tokens": "256",
            "save_intermediate_images": "false",
            "return_coordinates": "false",
        },
    )

    assert response.status_code == 200
    assert captured == {
        "mode": "trocr_lines",
        "preprocess": False,
        "kwargs": {
            "model": "custom/model",
            "max_tokens": 256,
            "save_intermediate_images": False,
            "return_coordinates": False,
        },
    }


def test_recognize_page_rejects_invalid_max_tokens() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/recognize/page",
        files={"file": ("page.png", b"image", "image/png")},
        data={"mode": "trocr_lines", "max_tokens": "0"},
    )

    assert response.status_code == 400
    assert "max_tokens" in response.json()["detail"]


def test_recognition_response_includes_line_coordinates() -> None:
    record = RecognitionRecord(
        id=7,
        original_filename="page.png",
        file_size_bytes=123,
        stored_image_path="page.png",
        processed_image_path="annotated.png",
        preprocessing_applied=True,
        recognition_mode="trocr_lines",
        line_count=1,
        line_image_paths_json='["line.png"]',
        line_bboxes_json="[[10, 20, 110, 45]]",
        recognized_text="строка",
        confidence=None,
        model_name="stub",
        processing_time_ms=12,
        created_at=datetime.now(UTC),
    )

    response = routes._build_recognition_response(record)

    assert response.line_coordinates[0].model_dump() == {
        "line_number": 1,
        "x1": 10,
        "y1": 20,
        "x2": 110,
        "y2": 45,
    }


def test_qwen_page_mode_reports_missing_ml_dependencies(monkeypatch) -> None:
    monkeypatch.setattr(
        routes,
        "collect_status",
        lambda: {
            "packages": [
                {"package": "torch", "import_name": "torch", "installed": False, "version": None},
                {"package": "transformers", "import_name": "transformers", "installed": False, "version": None},
                {"package": "accelerate", "import_name": "accelerate", "installed": False, "version": None},
                {"package": "qwen-vl-utils", "import_name": "qwen_vl_utils", "installed": False, "version": None},
            ],
            "ready_for_trocr": False,
            "ready_for_qwen_inference": False,
            "ready_for_qwen_lora": False,
            "torch_cuda_available": None,
            "torch_device_count": None,
        },
    )
    client = TestClient(app)

    response = client.post(
        "/api/recognize/page",
        files={"file": ("page.png", b"image", "image/png")},
        data={"mode": "qwen_page"},
    )

    assert response.status_code == 503
    assert "Missing packages" in response.json()["detail"]
    assert "qwen-vl-utils" in response.json()["detail"]


def test_feedback_endpoint_saves_corrected_text(monkeypatch, tmp_path: Path) -> None:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)

    image_path = tmp_path / "sample.png"
    Image.new("RGB", (20, 10), "white").save(image_path)
    with Session(engine) as db:
        record = RecognitionRecord(
            original_filename="sample.png",
            stored_image_path=str(image_path),
            recognized_text="wrong text",
            model_name="stub",
            processing_time_ms=10,
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        record_id = record.id

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    monkeypatch.setattr(routes.feedback_service.settings, "feedback_dir", tmp_path / "feedback")
    app.dependency_overrides[get_db] = override_get_db
    try:
        client = TestClient(app)
        response = client.post(
            f"/api/history/{record_id}/feedback",
            json={"rating": "partial", "corrected_text": "corrected text"},
        )
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 200
    body = response.json()
    assert body["feedback_rating"] == "partial"
    assert body["corrected_text"] == "corrected text"
    assert body["feedback_saved_for_training"] is True


def test_feedback_alias_saves_corrected_text_by_recognition_id(monkeypatch, tmp_path: Path) -> None:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)

    image_path = tmp_path / "sample.png"
    Image.new("RGB", (20, 10), "white").save(image_path)
    with Session(engine) as db:
        record = RecognitionRecord(
            original_filename="sample.png",
            stored_image_path=str(image_path),
            recognized_text="wrong text",
            model_name="stub",
            processing_time_ms=10,
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        record_id = record.id

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    monkeypatch.setattr(routes.feedback_service.settings, "feedback_dir", tmp_path / "feedback")
    app.dependency_overrides[get_db] = override_get_db
    try:
        client = TestClient(app)
        response = client.post(
            "/api/feedback",
            json={"recognition_id": record_id, "rating": "mismatch", "corrected_text": "corrected alias"},
        )
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 200
    body = response.json()
    assert body["feedback_rating"] == "mismatch"
    assert body["corrected_text"] == "corrected alias"
    assert body["feedback_saved_for_training"] is True


def test_feedback_stats_endpoint_counts_user_ratings() -> None:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)

    with Session(engine) as db:
        for index, rating in enumerate(["exact", "partial", "mismatch", None], start=1):
            db.add(
                RecognitionRecord(
                    original_filename=f"sample_{index}.png",
                    stored_image_path=f"sample_{index}.png",
                    recognized_text="text",
                    model_name="stub",
                    processing_time_ms=10,
                    feedback_rating=rating,
                )
            )
        db.commit()

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    try:
        client = TestClient(app)
        response = client.get("/api/feedback/stats")
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 200
    stats = response.json()
    assert stats["total_results"] == 4
    assert stats["rated_results"] == 3
    assert stats["unrated_results"] == 1
    assert stats["exact_count"] == 1
    assert stats["partial_count"] == 1
    assert stats["mismatch_count"] == 1
    assert stats["exact_rate"] == 1 / 3
    assert stats["exact_rate_all"] == 1 / 4


def test_feedback_problems_endpoint_lists_review_queue(monkeypatch, tmp_path: Path) -> None:
    problems_dir = tmp_path / "feedback" / "problems"
    problems_dir.mkdir(parents=True)
    Image.new("RGB", (20, 10), "white").save(problems_dir / "recognition_9.png")
    (problems_dir / "recognition_9.json").write_text(
        json.dumps(
            {
                "recognition_id": 9,
                "original_filename": "sample.png",
                "image_filename": "recognition_9.png",
                "processed_image_filename": None,
                "line_image_filenames": [],
                "corrected_text_filename": None,
                "recognized_text": "wrong",
                "corrected_text": "right",
                "rating": "mismatch",
                "model_name": "stub",
                "recognition_mode": "fragment",
                "line_count": 0,
                "line_coordinates": [],
                "preprocessing_applied": True,
                "created_at": "2026-01-01T00:00:00",
                "feedback_created_at": "2026-01-01T00:00:01",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(routes.feedback_service.settings, "feedback_dir", tmp_path / "feedback")
    client = TestClient(app)

    response = client.get("/api/feedback/problems")

    assert response.status_code == 200
    problems = response.json()
    assert len(problems) == 1
    assert problems[0]["recognition_id"] == 9
    assert problems[0]["image_url"] == "/api/feedback/problems/9/image/original"
    assert problems[0]["recognized_text"] == "wrong"
    assert problems[0]["corrected_text"] == "right"


def test_feedback_problem_image_endpoint_serves_saved_sample(monkeypatch, tmp_path: Path) -> None:
    problems_dir = tmp_path / "feedback" / "problems"
    problems_dir.mkdir(parents=True)
    Image.new("RGB", (20, 10), "white").save(problems_dir / "recognition_9.png")
    (problems_dir / "recognition_9.json").write_text(
        json.dumps(
            {
                "recognition_id": 9,
                "original_filename": "sample.png",
                "image_filename": "recognition_9.png",
                "processed_image_filename": None,
                "line_image_filenames": [],
                "corrected_text_filename": None,
                "recognized_text": "wrong",
                "corrected_text": None,
                "rating": "partial",
                "model_name": "stub",
                "recognition_mode": "fragment",
                "line_count": 0,
                "line_coordinates": [],
                "preprocessing_applied": True,
                "created_at": "2026-01-01T00:00:00",
                "feedback_created_at": "2026-01-01T00:00:01",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(routes.feedback_service.settings, "feedback_dir", tmp_path / "feedback")
    client = TestClient(app)

    response = client.get("/api/feedback/problems/9/image/original")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
