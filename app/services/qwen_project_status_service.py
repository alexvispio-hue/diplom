import json
from datetime import UTC, datetime
from pathlib import Path

from scripts.audit_acceptance import check_results


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNTIME_REPORT = PROJECT_ROOT / "data" / "runtime" / "qwen_runtime_check.json"
DEFAULT_EXECUTION_PLAN = PROJECT_ROOT / "data" / "runtime" / "qwen_execution_plan.json"
DEFAULT_EXECUTION_RESULTS = PROJECT_ROOT / "data" / "runtime" / "qwen_execution_results.json"
DEFAULT_SFT_JSONL = PROJECT_ROOT / "data" / "training" / "qwen_sft_light" / "mixed.jsonl"


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def flatten_results(results_payload: dict) -> list[dict]:
    latest = results_payload.get("results", [])
    historical = [
        result
        for run in results_payload.get("runs", [])
        for result in run.get("results", [])
    ]
    return [*latest, *historical]


def step_status(results: list[dict], step_id: str) -> str:
    matching = [result.get("status", "unknown") for result in results if result.get("id") == step_id]
    if "passed" in matching:
        return "passed"
    if matching:
        return matching[-1]
    return "not_run"


def count_jsonl_file(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def recommended_next_command(steps: dict[str, str]) -> str:
    commands = {
        "install_ml": "python scripts/run_qwen_execution_plan.py --step install_ml --allow-network",
        "qwen_smoke": "python scripts/run_qwen_execution_plan.py --step qwen_smoke --allow-network",
        "build_sft": "python scripts/run_qwen_execution_plan.py --step build_sft",
        "lora_dry_run": "python scripts/run_qwen_execution_plan.py --step lora_dry_run",
        "page_experiment": "python scripts/run_qwen_execution_plan.py --step page_experiment",
        "lora_train": "python scripts/run_qwen_execution_plan.py --step lora_train --allow-network",
    }
    for step_id in ("install_ml", "qwen_smoke", "build_sft", "lora_dry_run", "page_experiment", "lora_train"):
        if steps.get(step_id) != "passed":
            return commands[step_id]
    return "All Qwen execution plan steps passed."


def build_status(
    runtime_report: dict,
    execution_plan: dict,
    execution_results: dict,
    sft_jsonl_path: Path,
) -> dict:
    flattened_results = flatten_results(execution_results)
    acceptance = check_results()
    acceptance_counts = {
        "passed": sum(1 for check in acceptance if check.status == "passed"),
        "partial": sum(1 for check in acceptance if check.status == "partial"),
        "failed": sum(1 for check in acceptance if check.status == "failed"),
    }
    missing_packages = runtime_report.get("missing_packages") or execution_plan.get("runtime_missing_packages") or []
    steps = {
        step_id: step_status(flattened_results, step_id)
        for step_id in ("install_ml", "qwen_smoke", "build_sft", "lora_dry_run", "page_experiment", "lora_train")
    }
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": execution_plan.get("model")
        or runtime_report.get("resources", {}).get("recommendation", {}).get("model"),
        "install_profile": execution_plan.get("install_profile"),
        "runtime_status": runtime_report.get("status", execution_plan.get("runtime_status", "unknown")),
        "missing_packages": missing_packages,
        "recommended_next_command": recommended_next_command(steps),
        "sft_dataset": {
            "path": str(sft_jsonl_path),
            "examples": count_jsonl_file(sft_jsonl_path),
        },
        "steps": steps,
        "lora_dry_run_passed": step_status(flattened_results, "lora_dry_run") == "passed",
        "acceptance": acceptance_counts,
    }


class QwenProjectStatusService:
    def __init__(
        self,
        runtime_report_path: Path = DEFAULT_RUNTIME_REPORT,
        execution_plan_path: Path = DEFAULT_EXECUTION_PLAN,
        execution_results_path: Path = DEFAULT_EXECUTION_RESULTS,
        sft_jsonl_path: Path = DEFAULT_SFT_JSONL,
    ) -> None:
        self.runtime_report_path = runtime_report_path
        self.execution_plan_path = execution_plan_path
        self.execution_results_path = execution_results_path
        self.sft_jsonl_path = sft_jsonl_path

    def get_status(self) -> dict:
        return build_status(
            runtime_report=read_json(self.runtime_report_path),
            execution_plan=read_json(self.execution_plan_path),
            execution_results=read_json(self.execution_results_path),
            sft_jsonl_path=self.sft_jsonl_path,
        )
