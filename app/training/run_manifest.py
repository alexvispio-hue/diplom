from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any
from uuid import uuid4


RUN_STATUSES = {
    "created",
    "validated",
    "running",
    "interrupted",
    "failed",
    "training_completed",
    "evaluation_completed",
    "candidate_accepted",
    "candidate_rejected",
    "production",
}

ALLOWED_TRANSITIONS = {
    "created": {"validated", "failed"},
    "validated": {"running", "failed"},
    "running": {"interrupted", "failed", "training_completed"},
    "interrupted": {"running", "failed"},
    "failed": {"validated", "running"},
    "training_completed": {"evaluation_completed", "failed"},
    "evaluation_completed": {"candidate_accepted", "candidate_rejected", "failed"},
    "candidate_accepted": {"production", "candidate_rejected"},
    "candidate_rejected": {"candidate_accepted"},
    "production": {"candidate_rejected"},
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Run manifest was not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def git_commit_sha(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def collect_runtime_metadata() -> dict[str, Any]:
    packages = {}
    for package in ("torch", "transformers", "peft", "accelerate", "trl", "pillow"):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None

    gpu: dict[str, Any] = {"available": False, "count": 0, "devices": []}
    try:
        import torch

        gpu["available"] = bool(torch.cuda.is_available())
        gpu["count"] = int(torch.cuda.device_count())
        if gpu["available"]:
            gpu["devices"] = [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_bytes": int(torch.cuda.get_device_properties(index).total_memory),
                }
                for index in range(torch.cuda.device_count())
            ]
        gpu["cuda_version"] = getattr(torch.version, "cuda", None)
    except (ImportError, RuntimeError):
        pass

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "gpu": gpu,
    }


def create_run_manifest(
    path: Path,
    *,
    project_root: Path,
    run_name: str,
    base_model: str,
    train_dataset: Path,
    eval_dataset: Path | None,
    arguments: dict[str, Any],
    initial_adapter: Path | None = None,
    dataset_manifest: Path | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id or f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}",
        "run_name": run_name,
        "status": "created",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "git_commit": git_commit_sha(project_root),
        "base_model": base_model,
        "initial_adapter": str(initial_adapter) if initial_adapter else None,
        "datasets": {
            "train": {
                "path": str(train_dataset),
                "sha256": sha256_file(train_dataset),
            },
            "eval": (
                {"path": str(eval_dataset), "sha256": sha256_file(eval_dataset)}
                if eval_dataset
                else None
            ),
            "manifest": str(dataset_manifest) if dataset_manifest else None,
        },
        "arguments": arguments,
        "runtime": collect_runtime_metadata(),
        "events": [{"status": "created", "at": utc_now()}],
        "metrics": {},
        "artifacts": {},
    }
    atomic_write_json(path, payload)
    return payload


def update_run_manifest(
    path: Path,
    status: str,
    *,
    metrics: dict[str, Any] | None = None,
    artifacts: dict[str, Any] | None = None,
    message: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    if status not in RUN_STATUSES:
        raise ValueError(f"Unknown run status: {status}")

    payload = read_manifest(path)
    current_status = payload.get("status", "created")
    if status != current_status and not force:
        allowed = ALLOWED_TRANSITIONS.get(current_status, set())
        if status not in allowed:
            raise ValueError(f"Invalid run transition: {current_status} -> {status}")

    now = utc_now()
    payload["status"] = status
    payload["updated_at"] = now
    event: dict[str, Any] = {"status": status, "at": now}
    if message:
        event["message"] = message
    payload.setdefault("events", []).append(event)
    if metrics:
        payload.setdefault("metrics", {}).update(metrics)
    if artifacts:
        payload.setdefault("artifacts", {}).update(artifacts)
    atomic_write_json(path, payload)
    return payload
