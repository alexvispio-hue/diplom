from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.training.gold_benchmark import build_gold_candidates, build_sft_gold_candidates, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select author-balanced HWR200 pages for human gold review.")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--test-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--include-review", action="store_true")
    parser.add_argument("--school-test-dataset", type=Path)
    parser.add_argument("--school-count", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = build_gold_candidates(
        args.metadata,
        args.test_dataset,
        count=args.count,
        include_review=args.include_review,
    )
    if args.school_count:
        if not args.school_test_dataset:
            raise ValueError("--school-test-dataset is required when --school-count is positive.")
        candidates.extend(
            build_sft_gold_candidates(
                args.school_test_dataset,
                count=args.school_count,
                dataset_name="school-notebooks-ru",
            )
        )
    write_jsonl(args.output, candidates)
    summary = {
        "output": str(args.output),
        "pages": len(candidates),
        "authors": len({item["author_id"] for item in candidates if "author_id" in item}),
        "conditions": Counter(item.get("condition") for item in candidates),
        "review_status": "pending_human_review",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=dict))


if __name__ == "__main__":
    main()
