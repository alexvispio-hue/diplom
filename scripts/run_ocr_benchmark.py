from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.benchmark_service import summarize_predictions
from app.training.run_manifest import atomic_write_json, git_commit_sha, sha256_file, update_run_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a reviewed OCR benchmark through the local API.")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=["auto", "trocr_lines", "qwen_lines", "qwen_page", "qwen_layout", "compare", "hybrid"],
        required=True,
    )
    parser.add_argument("--api-url", default="http://127.0.0.1:8000/api")
    parser.add_argument("--model")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--no-preprocess", action="store_true")
    parser.add_argument("--allow-unverified", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--run-manifest", type=Path)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def main() -> None:
    args = parse_args()
    benchmark = read_jsonl(args.benchmark)
    if not args.allow_unverified:
        unverified = [item["benchmark_id"] for item in benchmark if item.get("review_status") != "verified"]
        if unverified:
            raise ValueError(
                f"Benchmark contains {len(unverified)} pages without verified human labels. "
                "Finalize the gold subset first."
            )
    if args.limit:
        benchmark = benchmark[: args.limit]
    session = requests.Session()
    health = session.get(f"{args.api_url.rstrip('/')}/health", timeout=10)
    health.raise_for_status()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = {item["benchmark_id"] for item in read_jsonl(args.output)} if args.output.is_file() else set()
    with args.output.open("a", encoding="utf-8", newline="\n") as target:
        for index, item in enumerate(benchmark, start=1):
            if item["benchmark_id"] in completed:
                continue
            image_path = (PROJECT_ROOT / item["image_path"]).resolve()
            with image_path.open("rb") as image:
                form = {
                    "mode": args.mode,
                    "preprocess": str(not args.no_preprocess).lower(),
                    "max_tokens": str(args.max_tokens),
                    "save_intermediate_images": "false",
                    "return_coordinates": "true",
                }
                if args.model:
                    form["model"] = args.model
                started = time.perf_counter()
                response = session.post(
                    f"{args.api_url.rstrip('/')}/recognize/page",
                    files={"file": (image_path.name, image, "image/png")},
                    data=form,
                    timeout=1800,
                )
            response.raise_for_status()
            result = response.json()
            diagnostics = session.get(f"{args.api_url.rstrip('/')}/diagnostics", timeout=30).json()
            record = {
                "benchmark_id": item["benchmark_id"],
                "source_dataset": item.get("source_dataset") or "hwr200",
                "condition": item.get("condition"),
                "image_path": item["image_path"],
                "reference_text": item["reference_text"],
                "prediction_text": result["recognized_text"],
                "mode": result["recognition_mode"],
                "model_name": result["model_name"],
                "model_registry_id": result.get("model_registry_id"),
                "latency_ms": result.get("processing_time_ms") or round((time.perf_counter() - started) * 1000, 3),
                "peak_vram_bytes": result.get("peak_vram_bytes"),
                "backend_memory_bytes": diagnostics["application"].get("process_memory_bytes"),
                "reliability_score": result.get("reliability_score"),
                "needs_review": result.get("needs_review"),
                "line_count": result.get("line_count"),
                "truncated": False,
            }
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
            target.flush()
            print(f"[{index}/{len(benchmark)}] {item['benchmark_id']}")

    predictions = read_jsonl(args.output)
    summary = summarize_predictions(predictions)
    subset_summaries = {
        source: summarize_predictions(
            [item for item in predictions if (item.get("source_dataset") or "hwr200") == source]
        ).to_dict()
        for source in sorted({str(item.get("source_dataset") or "hwr200") for item in predictions})
    }
    condition_summaries = {
        condition: summarize_predictions(
            [item for item in predictions if str(item.get("condition") or "unknown") == condition]
        ).to_dict()
        for condition in sorted({str(item.get("condition") or "unknown") for item in predictions})
    }
    report_path = args.output.with_suffix(".summary.json")
    report = {
        "schema_version": 1,
        "git_commit": git_commit_sha(PROJECT_ROOT),
        "benchmark": str(args.benchmark),
        "benchmark_sha256": sha256_file(args.benchmark),
        "predictions": str(args.output),
        "predictions_sha256": sha256_file(args.output),
        "mode": args.mode,
        "model_override": args.model,
        "summary": summary.to_dict(),
        "subsets": subset_summaries,
        "conditions": condition_summaries,
    }
    atomic_write_json(report_path, report)
    if args.run_manifest:
        update_run_manifest(
            args.run_manifest,
            "evaluation_completed",
            metrics={"gold_benchmark": summary.to_dict()},
            artifacts={"gold_predictions": str(args.output), "gold_report": str(report_path)},
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
