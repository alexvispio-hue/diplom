from pathlib import Path

from app.training.dataset_manifest import build_dataset_manifest


def test_dataset_manifest_hashes_all_splits(tmp_path: Path) -> None:
    for split in ("train", "val", "test"):
        (tmp_path / f"{split}.jsonl").write_text("{}\n{}\n", encoding="utf-8")
    (tmp_path / "validation_report.json").write_text('{"passed": true}', encoding="utf-8")

    manifest = build_dataset_manifest(
        tmp_path,
        name="pages",
        version="v1",
        source="source",
        source_revision="abc",
        license_name="apache-2.0",
        conversion_commit="def",
    )

    assert manifest["splits"]["train"]["examples"] == 2
    assert len(manifest["splits"]["val"]["sha256"]) == 64
    assert manifest["validation"]["passed"] is True
