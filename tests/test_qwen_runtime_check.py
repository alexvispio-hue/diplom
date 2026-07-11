from pathlib import Path

from app.ocr.base import RecognitionResult
from scripts.qwen_runtime_check import build_runtime_report, exit_code_for, parse_nvidia_smi_row, recommend_qwen_profile


def qwen_status(ready: bool) -> dict:
    packages = [
        {"package": "torch", "installed": ready, "version": "test"},
        {"package": "torchvision", "installed": ready, "version": "test"},
        {"package": "transformers", "installed": ready, "version": "test"},
        {"package": "accelerate", "installed": ready, "version": "test"},
        {"package": "qwen-vl-utils", "installed": ready, "version": "test"},
    ]
    return {
        "packages": packages,
        "ready_for_trocr": ready,
        "ready_for_qwen_inference": ready,
        "ready_for_qwen_lora": ready,
        "torch_cuda_available": False,
        "torch_device_count": 0,
    }


class FakeQwenRecognizer:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or "fake-qwen"

    def recognize(self, image_path: Path, max_new_tokens: int | None = None) -> RecognitionResult:
        return RecognitionResult(text=f"line:{image_path.name}:{max_new_tokens}", confidence=None, model_name=self.model_name)

    def recognize_page(self, image_path: Path, max_new_tokens: int | None = None) -> RecognitionResult:
        return RecognitionResult(text=f"page:{image_path.name}:{max_new_tokens}", confidence=None, model_name=self.model_name)


def resource_status() -> dict:
    return {
        "disk_free_gb": 100.0,
        "disk_total_gb": 500.0,
        "gpus": [],
        "recommendation": {"preset": "external_or_cpu", "model": None, "best_gpu_vram_mb": 0, "notes": []},
    }


def test_runtime_report_blocks_when_qwen_dependencies_are_missing(tmp_path: Path) -> None:
    report = build_runtime_report(
        status=qwen_status(False),
        image_path=tmp_path / "image.png",
        run_smoke=True,
        page_mode=False,
        model_name=None,
        max_new_tokens=None,
        recognizer_factory=FakeQwenRecognizer,
        resources=resource_status(),
    )

    assert report["status"] == "blocked"
    assert set(report["missing_packages"]) == {"torch", "torchvision", "transformers", "accelerate", "qwen-vl-utils"}
    assert report["smoke"] == {"ran": False}
    assert exit_code_for(report) == 2


def test_runtime_report_can_stop_after_dependency_gate_without_loading_model(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"not-used")

    report = build_runtime_report(
        status=qwen_status(True),
        image_path=image_path,
        run_smoke=False,
        page_mode=True,
        model_name="Qwen/test",
        max_new_tokens=16,
        recognizer_factory=FakeQwenRecognizer,
        resources=resource_status(),
    )

    assert report["status"] == "ready"
    assert report["smoke"] == {"ran": False}
    assert exit_code_for(report) == 0


def test_runtime_report_runs_smoke_with_injected_recognizer(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"fake")

    report = build_runtime_report(
        status=qwen_status(True),
        image_path=image_path,
        run_smoke=True,
        page_mode=True,
        model_name="Qwen/test",
        max_new_tokens=12,
        recognizer_factory=FakeQwenRecognizer,
        resources=resource_status(),
    )

    assert report["status"] == "passed"
    assert report["smoke"]["ok"] is True
    assert report["smoke"]["model_name"] == "Qwen/test"
    assert report["smoke"]["text"] == "page:image.png:12"
    assert exit_code_for(report) == 0


def test_parse_nvidia_smi_row_extracts_gpu_memory() -> None:
    gpu = parse_nvidia_smi_row("NVIDIA GeForce RTX 5070 Ti, 16303 MiB, 14421 MiB, 595.79")

    assert gpu == {
        "name": "NVIDIA GeForce RTX 5070 Ti",
        "memory_total_mb": 16303,
        "memory_free_mb": 14421,
        "driver_version": "595.79",
    }


def test_recommend_qwen_profile_prefers_light_for_16gb_gpu() -> None:
    recommendation = recommend_qwen_profile(
        [{"name": "RTX", "memory_total_mb": 16303, "memory_free_mb": 14000, "driver_version": "test"}],
        free_disk_gb=100.0,
    )

    assert recommendation["preset"] == "light"
    assert recommendation["model"] == "Qwen/Qwen3-VL-2B-Instruct"
