from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from app.core.config import get_settings
from app.services.model_registry_service import MODEL_ID_PATTERN, ModelRegistryService, model_artifact_info
from app.training.run_manifest import atomic_write_json


class ModelDownloadService:
    def __init__(
        self,
        state_path: Path | None = None,
        models_dir: Path | None = None,
    ) -> None:
        settings = get_settings()
        self.state_path = state_path or settings.model_download_state_path
        self.models_dir = (models_dir or settings.models_dir).resolve()
        self.registry = ModelRegistryService(models_dir=self.models_dir)

    def list(self) -> list[dict[str, Any]]:
        return list(reversed(self._read().get("downloads", [])))

    def get(self, download_id: str) -> dict[str, Any]:
        for item in self._read().get("downloads", []):
            if item["id"] == download_id:
                return item
        raise ValueError(f"Model download was not found: {download_id}")

    def create(
        self,
        *,
        model_id: str,
        repo_id: str,
        backend: str,
        revision: str | None,
        expected_sha256: str | None,
        stage: str,
    ) -> dict[str, Any]:
        if not MODEL_ID_PATTERN.fullmatch(model_id):
            raise ValueError("Model id may contain only letters, digits, dots, underscores, and hyphens.")
        target = (self.models_dir / model_id).resolve()
        try:
            target.relative_to(self.models_dir)
        except ValueError as exc:
            raise ValueError("Invalid local model id.") from exc
        if target.exists():
            raise ValueError(f"Model directory already exists: {target}")
        item = {
            "id": str(uuid4()),
            "model_id": model_id,
            "repo_id": repo_id,
            "backend": backend,
            "revision": revision,
            "expected_sha256": expected_sha256,
            "stage": stage,
            "target_path": str(target),
            "status": "queued",
            "downloaded_bytes": 0,
            "total_bytes": None,
            "completed_files": 0,
            "total_files": None,
            "error": None,
            "created_at": _now(),
            "updated_at": _now(),
        }
        state = self._read()
        state.setdefault("downloads", []).append(item)
        self._write(state)
        return item

    def run(self, download_id: str) -> None:
        state = self._read()
        item = _find_download(state, download_id)
        if item["status"] != "queued":
            return
        item["status"] = "running"
        item["updated_at"] = _now()
        self._write(state)
        target = Path(item["target_path"])
        try:
            from huggingface_hub import HfApi, hf_hub_download

            info = HfApi().model_info(
                item["repo_id"],
                revision=item.get("revision"),
                files_metadata=True,
            )
            files = [sibling for sibling in info.siblings if _safe_repo_file(sibling.rfilename)]
            total_bytes = sum(int(sibling.size or 0) for sibling in files)
            free_bytes = shutil.disk_usage(self.models_dir.anchor or self.models_dir).free
            if total_bytes and free_bytes < int(total_bytes * 1.1):
                raise OSError(
                    f"Not enough free disk space: need about {int(total_bytes * 1.1)} bytes, have {free_bytes}."
                )
            target.mkdir(parents=True, exist_ok=False)
            item["total_bytes"] = total_bytes or None
            item["total_files"] = len(files)
            self._write(state)
            for sibling in files:
                local_path = Path(
                    hf_hub_download(
                        repo_id=item["repo_id"],
                        filename=sibling.rfilename,
                        revision=item.get("revision"),
                        local_dir=target,
                    )
                )
                item["downloaded_bytes"] += int(sibling.size or local_path.stat().st_size)
                item["completed_files"] += 1
                item["updated_at"] = _now()
                self._write(state)

            size_bytes, checksum = model_artifact_info(target)
            expected = item.get("expected_sha256")
            if expected and checksum.casefold() != expected.casefold():
                raise ValueError("Downloaded model checksum does not match expected SHA-256.")
            self.registry.register(
                {
                    "id": item["model_id"],
                    "backend": item["backend"],
                    "base_model": item["repo_id"],
                    "revision": item.get("revision") or getattr(info, "sha", None),
                    "local_path": str(target),
                    "stage": item["stage"],
                    "source": f"huggingface:{item['repo_id']}",
                    "size_bytes": size_bytes,
                    "checksum_sha256": checksum,
                    "notes": "Installed by the local model manager.",
                }
            )
            item["status"] = "completed"
            item["checksum_sha256"] = checksum
            item["downloaded_bytes"] = size_bytes
            item["updated_at"] = _now()
            self._write(state)
        except Exception as exc:
            item["status"] = "failed"
            item["error"] = str(exc)
            item["updated_at"] = _now()
            self._write(state)

    def _read(self) -> dict[str, Any]:
        if not self.state_path.is_file():
            return {"schema_version": 1, "downloads": []}
        import json

        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def _write(self, state: dict[str, Any]) -> None:
        atomic_write_json(self.state_path, state)


def _find_download(state: dict[str, Any], download_id: str) -> dict[str, Any]:
    for item in state.get("downloads", []):
        if item["id"] == download_id:
            return item
    raise ValueError(f"Model download was not found: {download_id}")


def _safe_repo_file(filename: str) -> bool:
    path = PurePosixPath(filename)
    return bool(filename) and not path.is_absolute() and ".." not in path.parts


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
