import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.qwen_project_status_service import (
    DEFAULT_EXECUTION_PLAN,
    DEFAULT_EXECUTION_RESULTS,
    DEFAULT_RUNTIME_REPORT,
    DEFAULT_SFT_JSONL,
    QwenProjectStatusService,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize current Qwen project readiness.")
    parser.add_argument("--runtime-report", type=Path, default=DEFAULT_RUNTIME_REPORT)
    parser.add_argument("--execution-plan", type=Path, default=DEFAULT_EXECUTION_PLAN)
    parser.add_argument("--execution-results", type=Path, default=DEFAULT_EXECUTION_RESULTS)
    parser.add_argument("--sft-jsonl", type=Path, default=DEFAULT_SFT_JSONL)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    status = QwenProjectStatusService(
        runtime_report_path=args.runtime_report,
        execution_plan_path=args.execution_plan,
        execution_results_path=args.execution_results,
        sft_jsonl_path=args.sft_jsonl,
    ).get_status()
    if args.json:
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return

    print(f"generated_at: {status['generated_at']}")
    print(f"model: {status['model']}")
    print(f"install_profile: {status['install_profile']}")
    print(f"runtime_status: {status['runtime_status']}")
    print(f"missing_packages: {', '.join(status['missing_packages']) or 'none'}")
    print(f"sft_examples: {status['sft_dataset']['examples']}")
    print(f"lora_dry_run_passed: {status['lora_dry_run_passed']}")
    print(
        "acceptance: "
        f"{status['acceptance']['passed']} passed, "
        f"{status['acceptance']['partial']} partial, "
        f"{status['acceptance']['failed']} failed"
    )
    print(f"next: {status['recommended_next_command']}")


if __name__ == "__main__":
    main()
