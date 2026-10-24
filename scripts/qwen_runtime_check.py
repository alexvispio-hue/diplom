import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.ocr.qwen_vl_recognizer import QwenVLRecognizer
from app.services.ml_environment_service import QWEN_RUNTIME_PACKAGES, collect_status, missing_packages


DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "runtime" / "qwen_runtime_check.json"
DEFAULT_IMAGE_CANDIDATES = (
    PROJECT_ROOT / "data" / "evaluation" / "page_experiment_smoke" / "processed" / "trocr_lines" / "2013_page.png",
    PROJECT_ROOT / "data" / "evaluation" / "page_experiment_smoke" / "processed" / "trocr_lines" / "2013_lines" / "line_000.png",
    PROJECT_ROOT / "data" / "evaluation" / "segmentation_smoke" / "processed" / "2013" / "annotated.png",
)
QWEN_RESOURCE_HINTS = {
    "main": {
        "model": "Qwen/Qwen3-VL-8B-Instruct",
        "minimum_vram_mb": 20000,
        "recommended_vram_mb": 24000,
        "minimum_free_disk_gb": 35,
    },
    "light": {
        "model": "Qwen/Qwen3-VL-2B-Instruct",
        "minimum_vram_mb": 10000,
        "recommended_vram_mb": 12000,
        "minimum_free_disk_gb": 18,
    },
    "fallback_3b": {
        "model": "Qwen/Qwen2.5-VL-3B-Instruct",
        "minimum_vram_mb": 12000,
        "recommended_vram_mb": 16000,
        "minimum_free_disk_gb": 22,
    },
}


def parse_nvidia_smi_row(row: str) -> dict | None:
    parts = [part.strip() for part in row.split(",")]
    if len(parts) != 4:
        return None
    name, total, free, driver = parts
    try:
        total_mb = int(total.split()[0])
        free_mb = int(free.split()[0])
    except (IndexError, ValueError):
        return None
    return {
        "name": name,
        "memory_total_mb": total_mb,
        "memory_free_mb": free_mb,
        "driver_version": driver,
    }


def collect_gpu_resources(
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> list[dict]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.free,driver_version",
        "--format=csv,noheader",
    ]
    try:
        completed = runner(command, capture_output=True, text=True, check=False)
    except (FileNotFoundError, OSError):
        return []
    if completed.returncode != 0:
        return []
    return [
        gpu
        for line in completed.stdout.splitlines()
        if (gpu := parse_nvidia_smi_row(line)) is not None
    ]


def recommend_qwen_profile(gpus: list[dict], free_disk_gb: float) -> dict:
    best_vram = max((gpu["memory_total_mb"] for gpu in gpus), default=0)
    notes: list[str] = []

    if best_vram >= QWEN_RESOURCE_HINTS["main"]["recommended_vram_mb"] and free_disk_gb >= QWEN_RESOURCE_HINTS["main"]["minimum_free_disk_gb"]:
        preset = "main"
    elif best_vram >= QWEN_RESOURCE_HINTS["light"]["minimum_vram_mb"] and free_disk_gb >= QWEN_RESOURCE_HINTS["light"]["minimum_free_disk_gb"]:
        preset = "light"
        notes.append("Use the 2B model first; available VRAM is below the practical 8B recommendation.")
    else:
        preset = "external_or_cpu"
        notes.append("Local GPU/disk resources are below the recommended Qwen smoke profile.")

    return {
        "preset": preset,
        "model": QWEN_RESOURCE_HINTS.get(preset, {}).get("model"),
        "best_gpu_vram_mb": best_vram,
        "notes": notes,
    }


def collect_resource_status(
    root: Path = PROJECT_ROOT,
    gpu_runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> dict:
    disk = shutil.disk_usage(root)
    free_disk_gb = round(disk.free / (1024**3), 2)
    total_disk_gb = round(disk.total / (1024**3), 2)
    gpus = collect_gpu_resources(runner=gpu_runner)
    recommendation = recommend_qwen_profile(gpus, free_disk_gb)
    return {
        "disk_free_gb": free_disk_gb,
        "disk_total_gb": total_disk_gb,
        "gpus": gpus,
        "recommendation": recommendation,
    }


def find_default_image(root: Path = PROJECT_ROOT) -> Path | None:
    for candidate in DEFAULT_IMAGE_CANDIDATES:
        if candidate.exists():
            return candidate

    search_roots = (root / "data" / "evaluation", root / "data" / "training", root / "data" / "samples")
    suffixes = {".png", ".jpg", ".jpeg"}
    for search_root in search_roots:
        if not search_root.exists():
            continue
        images = sorted(path for path in search_root.rglob("*") if path.suffix.lower() in suffixes)
        if images:
            return images[0]
    return None


def build_runtime_report(
    *,
    status: dict,
    image_path: Path | None,
    run_smoke: bool,
    page_mode: bool,
    model_name: str | None,
    max_new_tokens: int | None,
    recognizer_factory: Callable[[str | None], QwenVLRecognizer] = QwenVLRecognizer,
    resources: dict | None = None,
) -> dict:
    missing = missing_packages(status, QWEN_RUNTIME_PACKAGES)
    resource_status = resources or collect_resource_status()
    report = {
        "status": "ready" if status["ready_for_qwen_inference"] else "blocked",
        "ready_for_qwen_inference": status["ready_for_qwen_inference"],
        "missing_packages": missing,
        "torch_cuda_available": status["torch_cuda_available"],
        "torch_device_count": status["torch_device_count"],
        "image": str(image_path) if image_path else None,
        "mode": "page" if page_mode else "line",
        "model": model_name,
        "resources": resource_status,
        "smoke": {"ran": False},
    }

    if not status["ready_for_qwen_inference"]:
        report["message"] = "Install missing Qwen runtime packages before running inference."
        return report
    if not run_smoke:
        report["message"] = "Qwen runtime packages are present. Add --run-smoke to run real inference."
        return report
    if image_path is None:
        report["status"] = "blocked"
        report["message"] = "No smoke image found. Pass --image explicitly."
        return report
    if not image_path.exists():
        report["status"] = "blocked"
        report["message"] = f"Smoke image not found: {image_path}"
        return report

    try:
        recognizer = recognizer_factory(model_name)
        result = (
            recognizer.recognize_page(image_path, max_new_tokens=max_new_tokens)
            if page_mode
            else recognizer.recognize(image_path, max_new_tokens=max_new_tokens)
        )
    except Exception as exc:
        report["status"] = "failed"
        report["message"] = str(exc)
        report["smoke"] = {"ran": True, "ok": False}
        return report

    report["status"] = "passed"
    report["message"] = "Qwen smoke inference completed."
    report["smoke"] = {
        "ran": True,
        "ok": True,
        "model_name": result.model_name,
        "text": result.text,
    }
    return report


def write_report(report: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def exit_code_for(report: dict) -> int:
    if report["status"] in {"ready", "passed"}:
        return 0
    if report["status"] == "blocked":
        return 2
    return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Qwen runtime readiness and optionally run smoke OCR.")
    parser.add_argument("--image", type=Path, help="Image for smoke inference. If omitted, a sample image is auto-detected.")
    parser.add_argument("--model", help="Override Qwen model id for this run.")
    parser.add_argument("--page", action="store_true", help="Use full-page OCR prompt instead of line prompt.")
    parser.add_argument("--max-new-tokens", type=int, help="Limit generated tokens during smoke inference.")
    parser.add_argument("--run-smoke", action="store_true", help="Run real Qwen inference after dependency checks.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="JSON report path.")
    parser.add_argument("--json", action="store_true", help="Print the report as JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_path = args.image or find_default_image()
    status = collect_status()
    report = build_runtime_report(
        status=status,
        image_path=image_path,
        run_smoke=args.run_smoke,
        page_mode=args.page,
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
    )
    write_report(report, args.output)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"status: {report['status']}")
        print(f"ready_for_qwen_inference: {report['ready_for_qwen_inference']}")
        if report["missing_packages"]:
            print("missing_packages: " + ", ".join(report["missing_packages"]))
        if report["image"]:
            print(f"image: {report['image']}")
        print(f"report: {args.output}")
        print(report["message"])
    return exit_code_for(report)


if __name__ == "__main__":
    raise SystemExit(main())
