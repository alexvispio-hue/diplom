import json
from pathlib import Path

import pytest

from app.training.run_manifest import create_run_manifest, sha256_file, update_run_manifest


def test_run_manifest_records_datasets_and_valid_transitions(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "val.jsonl"
    train_path.write_text("{}\n", encoding="utf-8")
    eval_path.write_text("{}\n", encoding="utf-8")
    manifest_path = tmp_path / "run.json"

    created = create_run_manifest(
        manifest_path,
        project_root=tmp_path,
        run_name="test",
        base_model="Qwen/test",
        train_dataset=train_path,
        eval_dataset=eval_path,
        arguments={"epochs": 1},
    )
    assert created["status"] == "created"
    assert created["datasets"]["train"]["sha256"] == sha256_file(train_path)

    validated = update_run_manifest(manifest_path, "validated")
    running = update_run_manifest(manifest_path, "running")
    completed = update_run_manifest(manifest_path, "training_completed", metrics={"train_loss": 1.0})

    assert validated["status"] == "validated"
    assert running["status"] == "running"
    assert completed["metrics"]["train_loss"] == 1.0
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["status"] == "training_completed"


def test_run_manifest_rejects_invalid_transition(tmp_path: Path) -> None:
    dataset_path = tmp_path / "train.jsonl"
    dataset_path.write_text("{}\n", encoding="utf-8")
    manifest_path = tmp_path / "run.json"
    create_run_manifest(
        manifest_path,
        project_root=tmp_path,
        run_name="test",
        base_model="Qwen/test",
        train_dataset=dataset_path,
        eval_dataset=None,
        arguments={},
    )

    with pytest.raises(ValueError, match="Invalid run transition"):
        update_run_manifest(manifest_path, "production")
