import json
from datetime import UTC, datetime
from pathlib import Path

from scripts.audit_acceptance import check_results


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNTIME_REPORT = PROJECT_ROOT / "data" / "runtime" / "qwen_runtime_check.json"
DEFAULT_EXECUTION_PLAN = PROJECT_ROOT / "data" / "runtime" / "qwen_execution_plan.json"
DEFAULT_EXECUTION_RESULTS = PROJECT_ROOT / "data" / "runtime" / "qwen_execution_results.json"
DEFAULT_SFT_JSONL = PROJECT_ROOT / "data" / "training" / "qwen_sft_light" / "mixed.jsonl"
DEFAULT_FULL_PIPELINE_STATUS = PROJECT_ROOT / "data" / "runtime" / "training_logs" / "hwr200_full_pipeline_status.json"
DEFAULT_FULL_RUN_MANIFEST = PROJECT_ROOT / "models" / "qwen-vl-school-hwr200-full-pages-lora" / "run_manifest.json"
DEFAULT_MODEL_REGISTRY = PROJECT_ROOT / "data" / "runtime" / "model_registry.json"
DEFAULT_DATASET_MANIFEST = PROJECT_ROOT / "manifests" / "datasets" / "hwr200-pages-v1.json"


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


def derive_pipeline_maturity(
    full_pipeline_status: dict,
    full_run_manifest: dict,
    model_registry: dict,
) -> tuple[str, bool, list[str]]:
    blockers = []
    phase = full_pipeline_status.get("phase")
    status = full_pipeline_status.get("status")
    run_status = full_run_manifest.get("status")
    production_id = model_registry.get("production", {}).get("qwen")

    if production_id:
        return "production", True, []
    if run_status == "candidate_accepted":
        blockers.append("Accepted Qwen candidate has not been promoted in the model registry.")
        return "candidate_accepted", False, blockers
    if run_status == "evaluation_completed":
        blockers.append("Full Qwen run is evaluated but has not passed the quality gate.")
        return "evaluation_completed", False, blockers
    if run_status == "training_completed":
        blockers.append("Full Qwen run has no completed evaluation report.")
        return "training_completed", False, blockers
    if run_status in {"running", "interrupted", "failed", "validated"}:
        blockers.append(f"Full Qwen run status is {run_status}; no evaluated candidate exists.")
        return run_status, False, blockers
    if phase == "dataset_ready" and status == "passed":
        blockers.append("Full HWR200 dataset is ready, but full Qwen fine-tuning is intentionally pending.")
        return "dataset_ready", False, blockers
    blockers.append("Full HWR200 dataset has not reached dataset_ready status.")
    return "not_ready", False, blockers


def build_status(
    runtime_report: dict,
    execution_plan: dict,
    execution_results: dict,
    sft_jsonl_path: Path,
    full_pipeline_status: dict | None = None,
    full_run_manifest: dict | None = None,
    model_registry: dict | None = None,
    dataset_manifest: dict | None = None,
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
    maturity, production_ready, blocking_reasons = derive_pipeline_maturity(
        full_pipeline_status or {},
        full_run_manifest or {},
        model_registry or {},
    )
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
        "pipeline_maturity": maturity,
        "production_ready": production_ready,
        "blocking_reasons": blocking_reasons,
        "full_dataset": {
            "name": (dataset_manifest or {}).get("name"),
            "version": (dataset_manifest or {}).get("version"),
            "splits": (dataset_manifest or {}).get("splits", {}),
            "validation_passed": bool((dataset_manifest or {}).get("validation", {}).get("passed")),
        },
    }


class QwenProjectStatusService:
    def __init__(
        self,
        runtime_report_path: Path = DEFAULT_RUNTIME_REPORT,
        execution_plan_path: Path = DEFAULT_EXECUTION_PLAN,
        execution_results_path: Path = DEFAULT_EXECUTION_RESULTS,
        sft_jsonl_path: Path = DEFAULT_SFT_JSONL,
        full_pipeline_status_path: Path = DEFAULT_FULL_PIPELINE_STATUS,
        full_run_manifest_path: Path = DEFAULT_FULL_RUN_MANIFEST,
        model_registry_path: Path = DEFAULT_MODEL_REGISTRY,
        dataset_manifest_path: Path = DEFAULT_DATASET_MANIFEST,
    ) -> None:
        self.runtime_report_path = runtime_report_path
        self.execution_plan_path = execution_plan_path
        self.execution_results_path = execution_results_path
        self.sft_jsonl_path = sft_jsonl_path
        self.full_pipeline_status_path = full_pipeline_status_path
        self.full_run_manifest_path = full_run_manifest_path
        self.model_registry_path = model_registry_path
        self.dataset_manifest_path = dataset_manifest_path

    def get_status(self) -> dict:
        return build_status(
            runtime_report=read_json(self.runtime_report_path),
            execution_plan=read_json(self.execution_plan_path),
            execution_results=read_json(self.execution_results_path),
            sft_jsonl_path=self.sft_jsonl_path,
            full_pipeline_status=read_json(self.full_pipeline_status_path),
            full_run_manifest=read_json(self.full_run_manifest_path),
            model_registry=read_json(self.model_registry_path),
            dataset_manifest=read_json(self.dataset_manifest_path),
        )
