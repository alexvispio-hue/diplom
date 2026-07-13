from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REQUIRED_TRAINING_PACKAGES = {
    "torch": "torch",
    "transformers": "transformers",
    "peft": "peft",
    "accelerate": "accelerate",
    "trl": "trl",
    "qwen-vl-utils": "qwen_vl_utils",
}


@dataclass
class PreflightReport:
    status: str = "passed"
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)

    def fail(self, check: str, message: str) -> None:
        self.checks[check] = False
        self.errors.append(message)
        self.status = "blocked"

    def pass_check(self, check: str) -> None:
        self.checks[check] = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "errors": self.errors,
            "warnings": self.warnings,
            "checks": self.checks,
            "metrics": self.metrics,
            "runtime": self.runtime,
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def image_path_from_example(example: dict[str, Any]) -> Path | None:
    messages = example.get("messages") or []
    if not messages:
        return None
    for part in messages[0].get("content", []):
        if part.get("type") == "image" and part.get("image"):
            return Path(str(part["image"]))
    return None


def _check_jsonl_dataset(
    report: PreflightReport,
    path: Path,
    expected_count: int | None,
    expected_sha256: str | None,
    label: str,
) -> list[dict[str, Any]]:
    if not path.is_file():
        report.fail(f"{label}_exists", f"{label} dataset was not found: {path}")
        return []
    examples = read_jsonl(path)
    report.metrics[f"{label}_examples"] = len(examples)
    if expected_count is not None and len(examples) != expected_count:
        report.fail(
            f"{label}_count",
            f"{label} dataset has {len(examples)} examples; manifest expects {expected_count}.",
        )
    else:
        report.pass_check(f"{label}_count")
    actual_sha256 = sha256_file(path)
    report.metrics[f"{label}_sha256"] = actual_sha256
    if expected_sha256 and actual_sha256.casefold() != expected_sha256.casefold():
        report.fail(
            f"{label}_sha256",
            f"{label} dataset SHA-256 does not match the dataset manifest.",
        )
    else:
        report.pass_check(f"{label}_sha256")
    return examples


def _check_image_references(
    report: PreflightReport,
    examples: list[dict[str, Any]],
    project_root: Path,
    label: str,
) -> None:
    missing = []
    malformed = 0
    for example in examples:
        image_path = image_path_from_example(example)
        if image_path is None:
            malformed += 1
            continue
        resolved = image_path if image_path.is_absolute() else project_root / image_path
        if not resolved.is_file():
            missing.append(str(image_path))
    report.metrics[f"{label}_missing_images"] = len(missing)
    report.metrics[f"{label}_malformed_examples"] = malformed
    if missing:
        report.fail(f"{label}_images", f"{label} dataset references {len(missing)} missing images.")
    elif malformed:
        report.fail(f"{label}_examples", f"{label} dataset contains {malformed} malformed image examples.")
    else:
        report.pass_check(f"{label}_images")


def _check_runtime(report: PreflightReport, *, minimum_vram_bytes: int) -> None:
    missing = []
    versions = {}
    for package, import_name in REQUIRED_TRAINING_PACKAGES.items():
        try:
            module = importlib.import_module(import_name)
            versions[package] = getattr(module, "__version__", "installed")
        except ImportError:
            missing.append(package)
    report.runtime["packages"] = versions
    if missing:
        report.fail("training_dependencies", "Missing training packages: " + ", ".join(missing))
    else:
        report.pass_check("training_dependencies")

    try:
        import torch

        report.runtime["cuda_available"] = bool(torch.cuda.is_available())
        report.runtime["cuda_version"] = getattr(torch.version, "cuda", None)
        if not torch.cuda.is_available():
            report.fail("cuda", "CUDA is not available; the configured Qwen run requires an NVIDIA GPU.")
            return
        devices = []
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_bytes": int(properties.total_memory),
                }
            )
        report.runtime["devices"] = devices
        report.runtime["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
        if max(item["total_memory_bytes"] for item in devices) < minimum_vram_bytes:
            report.fail(
                "vram",
                f"No CUDA device has the required {minimum_vram_bytes} bytes of VRAM.",
            )
        else:
            report.pass_check("vram")
        if not torch.cuda.is_bf16_supported():
            report.fail("bf16", "The configured run uses BF16, but this CUDA device does not report BF16 support.")
        else:
            report.pass_check("bf16")
    except (ImportError, RuntimeError) as exc:
        report.fail("cuda", f"Could not inspect CUDA runtime: {exc}")


def _check_model_cache(report: PreflightReport, model: str, project_root: Path) -> None:
    local_path = Path(model).expanduser()
    if local_path.is_dir() or (project_root / local_path).is_dir():
        report.pass_check("base_model")
        report.runtime["base_model"] = str(local_path)
        return
    try:
        from huggingface_hub import snapshot_download

        cached_path = snapshot_download(model, local_files_only=True)
        report.runtime["base_model"] = str(cached_path)
        report.pass_check("base_model")
    except Exception as exc:
        report.fail(
            "base_model",
            f"Base model is not available in the local Hugging Face cache: {model} ({exc})",
        )


def run_preflight(
    *,
    project_root: Path,
    train_dataset: Path,
    eval_dataset: Path,
    review_metadata: Path,
    dataset_manifest: Path,
    model: str,
    initial_adapter: Path | None = None,
    minimum_vram_bytes: int = 14 * 1024**3,
    check_runtime: bool = True,
    check_model_cache: bool = True,
) -> PreflightReport:
    report = PreflightReport()
    try:
        manifest = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        report.fail("dataset_manifest", f"Could not read dataset manifest: {exc}")
        return report
    if not manifest.get("validation", {}).get("passed"):
        report.fail("dataset_validation", "Dataset manifest is not marked as validated.")
    else:
        report.pass_check("dataset_validation")

    splits = manifest.get("splits", {})
    train_examples = _check_jsonl_dataset(
        report,
        train_dataset,
        splits.get("train", {}).get("examples"),
        splits.get("train", {}).get("sha256"),
        "train",
    )
    eval_examples = _check_jsonl_dataset(
        report,
        eval_dataset,
        splits.get("val", {}).get("examples"),
        splits.get("val", {}).get("sha256"),
        "eval",
    )
    _check_image_references(report, train_examples, project_root, "train")
    _check_image_references(report, eval_examples, project_root, "eval")

    if not review_metadata.is_file():
        report.fail("review_metadata", f"Review metadata was not found: {review_metadata}")
    else:
        metadata = read_jsonl(review_metadata)
        report.metrics["review_metadata_examples"] = len(metadata)
        expected_metadata = manifest.get("metadata", {}).get("examples")
        if expected_metadata is not None and len(metadata) != expected_metadata:
            report.fail("review_metadata_count", "Review metadata count does not match the dataset manifest.")
        else:
            report.pass_check("review_metadata_count")
        expected_sha256 = manifest.get("metadata", {}).get("sha256")
        if expected_sha256 and sha256_file(review_metadata).casefold() != expected_sha256.casefold():
            report.fail("review_metadata_sha256", "Review metadata SHA-256 does not match the dataset manifest.")
        else:
            report.pass_check("review_metadata_sha256")

    if initial_adapter:
        adapter_path = initial_adapter if initial_adapter.is_absolute() else project_root / initial_adapter
        adapter_files = (adapter_path / "adapter_config.json", adapter_path / "adapter_model.safetensors")
        if not adapter_path.is_dir() or not all(path.is_file() for path in adapter_files):
            report.fail("initial_adapter", f"Initial adapter is incomplete: {adapter_path}")
        else:
            report.pass_check("initial_adapter")

    if check_runtime:
        _check_runtime(report, minimum_vram_bytes=minimum_vram_bytes)
    if check_model_cache:
        _check_model_cache(report, model, project_root)
    return report
