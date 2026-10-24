from __future__ import annotations

import json
import re
import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.config import BASE_DIR, get_settings
from app.training.run_manifest import atomic_write_json


MODEL_STAGES = {"baseline", "experimental", "candidate", "production", "rejected"}
MODEL_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{1,127}$")
DEFAULT_REGISTRY_PATH = BASE_DIR / "data" / "runtime" / "model_registry.json"
DEFAULT_MODELS_DIR = BASE_DIR / "models"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class ModelRegistryService:
    def __init__(
        self,
        registry_path: Path | None = None,
        models_dir: Path | None = None,
    ) -> None:
        settings = get_settings()
        self.registry_path = registry_path or settings.model_registry_path
        self.models_dir = (models_dir or settings.models_dir).resolve()

    def read(self) -> dict[str, Any]:
        if not self.registry_path.exists():
            return {
                "schema_version": 1,
                "updated_at": _now(),
                "models": [],
                "production": {},
                "promotion_history": [],
            }
        try:
            return json.loads(self.registry_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Model registry is not valid JSON: {self.registry_path}") from exc

    def list_models(self) -> list[dict[str, Any]]:
        registry = self.read()
        production = registry.get("production", {})
        result = []
        for model in registry.get("models", []):
            item = dict(model)
            item["is_production"] = production.get(item["backend"]) == item["id"]
            local_path = item.get("local_path")
            item["installed"] = bool(local_path and Path(local_path).exists()) or not local_path
            result.append(item)
        return result

    def register(self, model: dict[str, Any]) -> dict[str, Any]:
        model_id = str(model.get("id") or "")
        if not MODEL_ID_PATTERN.fullmatch(model_id):
            raise ValueError("Model id may contain only letters, digits, dots, underscores, and hyphens.")
        backend = str(model.get("backend") or "")
        if backend not in {"trocr", "qwen"}:
            raise ValueError("Model backend must be 'trocr' or 'qwen'.")
        stage = str(model.get("stage") or "experimental")
        if stage not in MODEL_STAGES or stage == "production":
            raise ValueError("Register a model as baseline, experimental, candidate, or rejected; use promote for production.")

        local_path = model.get("local_path")
        if local_path:
            local_path = str(self._validate_model_path(Path(local_path), must_exist=True))

        registry = self.read()
        if any(item["id"] == model_id for item in registry["models"]):
            raise ValueError(f"Model is already registered: {model_id}")
        item = {
            "id": model_id,
            "backend": backend,
            "base_model": str(model.get("base_model") or ""),
            "revision": model.get("revision"),
            "adapter": model.get("adapter"),
            "local_path": local_path,
            "stage": stage,
            "quality_report": model.get("quality_report"),
            "created_at": _now(),
            "notes": model.get("notes"),
            "source": model.get("source"),
            "size_bytes": model.get("size_bytes"),
            "checksum_sha256": model.get("checksum_sha256"),
        }
        if local_path:
            size_bytes, checksum = model_artifact_info(Path(local_path))
            item["size_bytes"] = int(item["size_bytes"] or size_bytes)
            item["checksum_sha256"] = item["checksum_sha256"] or checksum
        registry["models"].append(item)
        self._write(registry)
        return item

    def discover_local_adapters(self) -> list[dict[str, Any]]:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        registry = self.read()
        known_paths = {item.get("local_path") for item in registry["models"]}
        discovered = []
        config_paths = {
            *self.models_dir.glob("*/adapter_config.json"),
            *self.models_dir.glob("*/config.json"),
        }
        for config_path in sorted(config_paths):
            adapter_dir = config_path.parent.resolve()
            if str(adapter_dir) in known_paths:
                continue
            config = json.loads(config_path.read_text(encoding="utf-8"))
            model_id = _unique_model_id(registry["models"], adapter_dir.name)
            is_adapter = config_path.name == "adapter_config.json"
            model_type = str(config.get("model_type") or "").lower()
            backend = "qwen" if "qwen" in f"{adapter_dir.name} {model_type}".lower() else "trocr"
            item = {
                "id": model_id,
                "backend": backend,
                "base_model": str(
                    config.get("base_model_name_or_path")
                    or config.get("_name_or_path")
                    or adapter_dir
                ),
                "revision": config.get("revision"),
                "adapter": adapter_dir.name if is_adapter else None,
                "local_path": str(adapter_dir),
                "stage": "experimental" if is_adapter else "baseline",
                "quality_report": None,
                "created_at": _now(),
                "notes": f"Discovered from local {config_path.name}.",
                "source": "local-discovery",
            }
            item["size_bytes"], item["checksum_sha256"] = model_artifact_info(adapter_dir)
            registry["models"].append(item)
            known_paths.add(str(adapter_dir))
            discovered.append(item)
        if discovered:
            self._write(registry)
        return discovered

    def promote(self, model_id: str, *, allow_unverified: bool = False) -> dict[str, Any]:
        registry = self.read()
        model = _find_model(registry, model_id)
        quality_report = model.get("quality_report")
        if model["backend"] == "qwen" and not allow_unverified:
            if not quality_report:
                raise ValueError("Qwen model cannot be promoted without a quality report.")
            report_path = Path(quality_report)
            if not report_path.exists():
                raise ValueError(f"Quality report was not found: {report_path}")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            gate = report.get("quality_gate", report)
            if gate.get("passed") is not True:
                raise ValueError("Qwen quality gate has not passed.")

        backend = model["backend"]
        if model.get("local_path") and not self.verify(model_id)["valid"]:
            raise ValueError("Model checksum verification failed.")
        previous_id = registry.setdefault("production", {}).get(backend)
        if previous_id and previous_id != model_id:
            _find_model(registry, previous_id)["stage"] = "candidate"
        model["stage"] = "production"
        registry["production"][backend] = model_id
        registry.setdefault("promotion_history", []).append(
            {
                "backend": backend,
                "from": previous_id,
                "to": model_id,
                "at": _now(),
            }
        )
        self._write(registry)
        return model

    def rollback(self, backend: str) -> dict[str, Any]:
        registry = self.read()
        history = registry.get("promotion_history", [])
        current_id = registry.get("production", {}).get(backend)
        previous_id = None
        for event in reversed(history):
            if event.get("backend") == backend and event.get("to") == current_id and event.get("from"):
                previous_id = event["from"]
                break
        if not previous_id:
            raise ValueError(f"No rollback target is available for backend: {backend}")
        current = _find_model(registry, current_id)
        previous = _find_model(registry, previous_id)
        current["stage"] = "candidate"
        previous["stage"] = "production"
        registry["production"][backend] = previous_id
        registry.setdefault("promotion_history", []).append(
            {"backend": backend, "from": current_id, "to": previous_id, "at": _now(), "rollback": True}
        )
        self._write(registry)
        return previous

    def unregister(self, model_id: str, *, delete_files: bool = False) -> None:
        registry = self.read()
        model = _find_model(registry, model_id)
        if registry.get("production", {}).get(model["backend"]) == model_id:
            raise ValueError("Production model cannot be removed; promote or rollback first.")
        if delete_files and model.get("local_path"):
            path = self._validate_model_path(Path(model["local_path"]), must_exist=True)
            if path.is_dir():
                import shutil

                shutil.rmtree(path)
            else:
                path.unlink()
        registry["models"] = [item for item in registry["models"] if item["id"] != model_id]
        self._write(registry)

    def production_model(self, backend: str, *, verify_integrity: bool = True) -> dict[str, Any] | None:
        registry = self.read()
        model_id = registry.get("production", {}).get(backend)
        if not model_id:
            return None
        model = _find_model(registry, model_id)
        if verify_integrity and model.get("local_path") and not self.verify(model_id)["valid"]:
            raise ValueError(f"Production model failed checksum verification: {model_id}")
        return model

    def verify(self, model_id: str) -> dict[str, Any]:
        model = _find_model(self.read(), model_id)
        local_path = model.get("local_path")
        if not local_path:
            return {"model_id": model_id, "valid": True, "reason": "remote-reference"}
        path = self._validate_model_path(Path(local_path), must_exist=True)
        size_bytes, checksum = model_artifact_info(path)
        expected = model.get("checksum_sha256")
        valid = expected is not None and checksum.casefold() == str(expected).casefold()
        return {
            "model_id": model_id,
            "valid": valid,
            "size_bytes": size_bytes,
            "checksum_sha256": checksum,
            "expected_sha256": expected,
            "reason": "verified" if valid else "checksum-mismatch-or-missing",
        }

    def _validate_model_path(self, path: Path, *, must_exist: bool) -> Path:
        resolved = path.resolve()
        try:
            resolved.relative_to(self.models_dir)
        except ValueError as exc:
            raise ValueError(f"Model path must stay inside {self.models_dir}.") from exc
        if must_exist and not resolved.exists():
            raise ValueError(f"Model path does not exist: {resolved}")
        return resolved

    def _write(self, registry: dict[str, Any]) -> None:
        registry["updated_at"] = _now()
        atomic_write_json(self.registry_path, registry)


def _find_model(registry: dict[str, Any], model_id: str | None) -> dict[str, Any]:
    for item in registry.get("models", []):
        if item.get("id") == model_id:
            return item
    raise ValueError(f"Registered model was not found: {model_id}")


def _unique_model_id(models: list[dict[str, Any]], preferred: str) -> str:
    existing = {item["id"] for item in models}
    if preferred not in existing:
        return preferred
    index = 2
    while f"{preferred}-{index}" in existing:
        index += 1
    return f"{preferred}-{index}"


def model_artifact_info(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    if path.is_file():
        files = [path]
        root = path.parent
    else:
        files = sorted(
            item
            for item in path.rglob("*")
            if item.is_file()
            and ".cache" not in item.relative_to(path).parts
            and item.suffix.lower() != ".lock"
        )
        root = path
    total_size = 0
    for file_path in files:
        relative = file_path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        with file_path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                total_size += len(chunk)
                digest.update(chunk)
    return total_size, digest.hexdigest()
