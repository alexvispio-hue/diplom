from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.benchmark_service import evaluate_quality_gate, summarize_predictions
from app.training.run_manifest import atomic_write_json


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def ensure_same_benchmark(candidate_records: list[dict], baseline_records: list[dict]) -> None:
    candidate_ids = {item.get("benchmark_id") for item in candidate_records}
    baseline_ids = {item.get("benchmark_id") for item in baseline_records}
    if candidate_ids != baseline_ids:
        missing = sorted(baseline_ids - candidate_ids)
        unexpected = sorted(candidate_ids - baseline_ids)
        raise ValueError(
            "Candidate and baseline predictions cover different benchmark pages. "
            f"Missing: {missing[:5]}; unexpected: {unexpected[:5]}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate OCR prediction JSONL without running a model.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--regression-predictions", type=Path)
    parser.add_argument("--regression-baseline", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if bool(args.regression_predictions) != bool(args.regression_baseline):
        raise ValueError("Provide both --regression-predictions and --regression-baseline.")
    candidate_records = read_jsonl(args.predictions)
    candidate = summarize_predictions(candidate_records)
    payload = {"summary": candidate.to_dict()}
    if args.baseline:
        baseline_records = read_jsonl(args.baseline)
        ensure_same_benchmark(candidate_records, baseline_records)
        baseline = summarize_predictions(baseline_records)
        school_candidate_records = [
            item for item in candidate_records if (item.get("source_dataset") or "").casefold() == "schooldataset"
        ]
        school_baseline_records = [
            item for item in baseline_records if (item.get("source_dataset") or "").casefold() == "schooldataset"
        ]
        regression_candidate = (
            summarize_predictions(read_jsonl(args.regression_predictions))
            if args.regression_predictions
            else None
        )
        regression_baseline = (
            summarize_predictions(read_jsonl(args.regression_baseline))
            if args.regression_baseline
            else None
        )
        payload["quality_gate"] = evaluate_quality_gate(
            baseline,
            candidate,
            school_baseline=(summarize_predictions(school_baseline_records) if school_baseline_records else None),
            school_candidate=(summarize_predictions(school_candidate_records) if school_candidate_records else None),
            regression_baseline=regression_baseline,
            regression_candidate=regression_candidate,
        ).to_dict()
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
