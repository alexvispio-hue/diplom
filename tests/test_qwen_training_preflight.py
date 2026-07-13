from __future__ import annotations

import json

from app.training.qwen_training_preflight import run_preflight, sha256_file


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _example(image_path: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": [{"type": "image", "image": image_path}]},
            {"role": "assistant", "content": "текст"},
        ]
    }


def test_preflight_validates_dataset_hashes_and_images(tmp_path):
    image = tmp_path / "page.jpg"
    image.write_bytes(b"image")
    train = tmp_path / "train.jsonl"
    evaluation = tmp_path / "val.jsonl"
    metadata = tmp_path / "metadata.jsonl"
    train_rows = [_example(str(image))]
    eval_rows = [_example(str(image))]
    metadata_rows = [{"image_path": str(image), "needs_review": False}]
    _write_jsonl(train, train_rows)
    _write_jsonl(evaluation, eval_rows)
    _write_jsonl(metadata, metadata_rows)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "validation": {"passed": True},
                "splits": {
                    "train": {"examples": 1, "sha256": sha256_file(train)},
                    "val": {"examples": 1, "sha256": sha256_file(evaluation)},
                },
                "metadata": {"examples": 1, "sha256": sha256_file(metadata)},
            }
        ),
        encoding="utf-8",
    )
    report = run_preflight(
        project_root=tmp_path,
        train_dataset=train,
        eval_dataset=evaluation,
        review_metadata=metadata,
        dataset_manifest=manifest,
        model="local-model",
        check_runtime=False,
        check_model_cache=False,
    )
    assert report.status == "passed"
    assert report.metrics["train_examples"] == 1


def test_preflight_blocks_missing_images(tmp_path):
    train = tmp_path / "train.jsonl"
    evaluation = tmp_path / "val.jsonl"
    metadata = tmp_path / "metadata.jsonl"
    _write_jsonl(train, [_example("missing.jpg")])
    _write_jsonl(evaluation, [_example("missing.jpg")])
    _write_jsonl(metadata, [{"image_path": "missing.jpg", "needs_review": False}])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "validation": {"passed": True},
                "splits": {
                    "train": {"examples": 1, "sha256": sha256_file(train)},
                    "val": {"examples": 1, "sha256": sha256_file(evaluation)},
                },
                "metadata": {"examples": 1, "sha256": sha256_file(metadata)},
            }
        ),
        encoding="utf-8",
    )
    report = run_preflight(
        project_root=tmp_path,
        train_dataset=train,
        eval_dataset=evaluation,
        review_metadata=metadata,
        dataset_manifest=manifest,
        model="local-model",
        check_runtime=False,
        check_model_cache=False,
    )
    assert report.status == "blocked"
    assert report.checks["train_images"] is False
