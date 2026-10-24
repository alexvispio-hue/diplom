from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.training.run_manifest import atomic_write_json, sha256_file


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as source:
        return sum(1 for line in source if line.strip())


def build_dataset_manifest(
    dataset_dir: Path,
    *,
    name: str,
    version: str,
    source: str,
    source_revision: str | None,
    license_name: str,
    conversion_commit: str | None,
    split_names: tuple[str, ...] = ("train", "val", "test"),
) -> dict[str, Any]:
    splits: dict[str, Any] = {}
    for split_name in split_names:
        path = dataset_dir / f"{split_name}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Dataset split was not found: {path}")
        splits[split_name] = {
            "path": path.name,
            "examples": count_jsonl(path),
            "sha256": sha256_file(path),
        }

    validation_path = dataset_dir / "validation_report.json"
    validation = (
        json.loads(validation_path.read_text(encoding="utf-8"))
        if validation_path.exists()
        else None
    )
    metadata_path = dataset_dir / "page_metadata.jsonl"

    return {
        "schema_version": 1,
        "name": name,
        "version": version,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": source,
        "source_revision": source_revision,
        "license": license_name,
        "conversion_commit": conversion_commit,
        "split_strategy": "author-disjoint; all variants of a page remain in one split",
        "splits": splits,
        "metadata": (
            {
                "path": metadata_path.name,
                "examples": count_jsonl(metadata_path),
                "sha256": sha256_file(metadata_path),
            }
            if metadata_path.exists()
            else None
        ),
        "validation": validation,
    }


def write_dataset_manifest(path: Path, manifest: dict[str, Any]) -> None:
    atomic_write_json(path, manifest)
