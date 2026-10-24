from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.training.qwen_training_preflight import run_preflight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the local Qwen training environment without training.")
    parser.add_argument("--train-dataset", type=Path, default=Path("data/training/hwr200_qwen_pages_full/train.jsonl"))
    parser.add_argument("--eval-dataset", type=Path, default=Path("data/training/hwr200_qwen_pages_full/val.jsonl"))
    parser.add_argument("--review-metadata", type=Path, default=Path("data/training/hwr200_qwen_pages_full/page_metadata.jsonl"))
    parser.add_argument("--dataset-manifest", type=Path, default=Path("manifests/datasets/hwr200-pages-v1.json"))
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--initial-adapter", type=Path, default=Path("models/qwen-vl-school-pages-lora"))
    parser.add_argument("--output", type=Path, default=Path("data/runtime/training_logs/qwen_training_preflight.json"))
    parser.add_argument("--minimum-vram-gb", type=float, default=14.0)
    parser.add_argument("--skip-runtime", action="store_true")
    parser.add_argument("--skip-model-cache", action="store_true")
    parser.add_argument("--no-initial-adapter", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_preflight(
        project_root=PROJECT_ROOT,
        train_dataset=(PROJECT_ROOT / args.train_dataset).resolve(),
        eval_dataset=(PROJECT_ROOT / args.eval_dataset).resolve(),
        review_metadata=(PROJECT_ROOT / args.review_metadata).resolve(),
        dataset_manifest=(PROJECT_ROOT / args.dataset_manifest).resolve(),
        model=args.model,
        initial_adapter=(PROJECT_ROOT / args.initial_adapter).resolve()
        if args.initial_adapter and not args.no_initial_adapter
        else None,
        minimum_vram_bytes=int(args.minimum_vram_gb * 1024**3),
        check_runtime=not args.skip_runtime,
        check_model_cache=not args.skip_model_cache,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    print(f"preflight_report: {args.output}")
    return 0 if report.status == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
