import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.ml_environment_service import collect_status, package_status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check local ML dependencies for OCR experiments.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    status = collect_status()
    if args.json:
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return

    for package in status["packages"]:
        marker = "ok" if package["installed"] else "missing"
        version = f" {package['version']}" if package["version"] else ""
        print(f"{marker:7} {package['package']}{version}")
    print(f"ready_for_trocr: {status['ready_for_trocr']}")
    print(f"ready_for_qwen_inference: {status['ready_for_qwen_inference']}")
    print(f"ready_for_qwen_lora: {status['ready_for_qwen_lora']}")
    print(f"torch_cuda_available: {status['torch_cuda_available']}")
    print(f"torch_device_count: {status['torch_device_count']}")


if __name__ == "__main__":
    main()
