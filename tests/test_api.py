from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoint() -> None:
    client = TestClient(app)

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_models_endpoint() -> None:
    client = TestClient(app)

    response = client.get("/api/models")

    assert response.status_code == 200
    assert response.json()[0]["local_inference"] is True


def test_recognize_rejects_non_image_file() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/recognize",
        files={"file": ("note.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 400
