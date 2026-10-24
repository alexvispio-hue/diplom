import json
from pathlib import Path

from app.services.qwen_project_status_service import (
    build_status,
    count_jsonl_file,
    derive_pipeline_maturity,
    flatten_results,
    recommended_next_command,
    step_status,
)


def test_flatten_results_reads_latest_and_historical_runs() -> None:
    payload = {
        "results": [{"id": "lora_dry_run", "status": "passed"}],
        "runs": [
            {"results": [{"id": "build_sft", "status": "passed"}]},
            {"results": [{"id": "install_ml", "status": "blocked"}]},
        ],
    }

    flattened = flatten_results(payload)

    assert [result["id"] for result in flattened] == ["lora_dry_run", "build_sft", "install_ml"]
    assert step_status(flattened, "build_sft") == "passed"
    assert step_status(flattened, "qwen_smoke") == "not_run"


def test_count_jsonl_file_counts_non_empty_lines(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "mixed.jsonl"
    jsonl_path.write_text("{}\n\n{}\n", encoding="utf-8")

    assert count_jsonl_file(jsonl_path) == 2


def test_build_status_summarizes_qwen_readiness(tmp_path: Path) -> None:
    sft_path = tmp_path / "mixed.jsonl"
    sft_path.write_text(json.dumps({"messages": []}) + "\n", encoding="utf-8")

    status = build_status(
        runtime_report={"status": "blocked", "missing_packages": ["torch"]},
        execution_plan={"model": "Qwen/test", "install_profile": "cuda-cu130"},
        execution_results={
            "runs": [
                {"results": [{"id": "build_sft", "status": "passed"}]},
                {"results": [{"id": "lora_dry_run", "status": "passed"}]},
            ]
        },
        sft_jsonl_path=sft_path,
    )

    assert status["generated_at"].endswith("+00:00")
    assert status["model"] == "Qwen/test"
    assert status["install_profile"] == "cuda-cu130"
    assert status["runtime_status"] == "blocked"
    assert status["missing_packages"] == ["torch"]
    assert status["sft_dataset"]["examples"] == 1
    assert status["steps"]["build_sft"] == "passed"
    assert status["lora_dry_run_passed"] is True
    assert status["pipeline_maturity"] == "not_ready"
    assert status["production_ready"] is False


def test_recommended_next_command_reports_completion_when_all_steps_passed() -> None:
    steps = {
        "install_ml": "passed",
        "qwen_smoke": "passed",
        "build_sft": "passed",
        "lora_dry_run": "passed",
        "page_experiment": "passed",
        "lora_train": "passed",
    }

    assert recommended_next_command(steps) == "All Qwen execution plan steps passed."


def test_full_pipeline_status_does_not_treat_smoke_as_production() -> None:
    maturity, ready, blockers = derive_pipeline_maturity(
        {"phase": "dataset_ready", "status": "passed"},
        {},
        {},
    )

    assert maturity == "dataset_ready"
    assert ready is False
    assert "fine-tuning" in blockers[0]
