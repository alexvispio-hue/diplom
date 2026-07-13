import json
from pathlib import Path

import pytest

from app.services.model_registry_service import ModelRegistryService


def _service(tmp_path: Path) -> ModelRegistryService:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    return ModelRegistryService(tmp_path / "registry.json", models_dir)


def test_qwen_promotion_requires_passing_quality_gate(tmp_path: Path) -> None:
    service = _service(tmp_path)
    adapter = tmp_path / "models" / "adapter"
    adapter.mkdir()
    failed_report = tmp_path / "failed.json"
    failed_report.write_text('{"quality_gate": {"passed": false}}', encoding="utf-8")
    service.register(
        {
            "id": "qwen-test",
            "backend": "qwen",
            "base_model": "Qwen/test",
            "local_path": str(adapter),
            "stage": "candidate",
            "quality_report": str(failed_report),
        }
    )

    with pytest.raises(ValueError, match="quality gate"):
        service.promote("qwen-test")

    passed_report = tmp_path / "passed.json"
    passed_report.write_text('{"quality_gate": {"passed": true}}', encoding="utf-8")
    registry = json.loads((tmp_path / "registry.json").read_text(encoding="utf-8"))
    registry["models"][0]["quality_report"] = str(passed_report)
    (tmp_path / "registry.json").write_text(json.dumps(registry), encoding="utf-8")

    promoted = service.promote("qwen-test")

    assert promoted["stage"] == "production"
    assert service.production_model("qwen")["id"] == "qwen-test"


def test_model_registry_rolls_back_and_protects_production_files(tmp_path: Path) -> None:
    service = _service(tmp_path)
    first = tmp_path / "models" / "first"
    second = tmp_path / "models" / "second"
    first.mkdir()
    second.mkdir()
    for model_id, path in (("trocr-one", first), ("trocr-two", second)):
        service.register(
            {
                "id": model_id,
                "backend": "trocr",
                "base_model": model_id,
                "local_path": str(path),
                "stage": "candidate",
            }
        )

    service.promote("trocr-one")
    service.promote("trocr-two")
    rolled_back = service.rollback("trocr")

    assert rolled_back["id"] == "trocr-one"
    with pytest.raises(ValueError, match="Production model"):
        service.unregister("trocr-one", delete_files=True)
    assert first.exists()
