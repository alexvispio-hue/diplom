from scripts.audit_acceptance import AcceptanceCheck
from scripts.generate_project_readiness_report import render_report


def test_render_report_contains_qwen_status_and_acceptance_rows() -> None:
    report = render_report(
        {
            "acceptance": {"passed": 13, "partial": 4, "failed": 0},
            "generated_at": "2026-06-29T00:00:00+00:00",
            "model": "Qwen/test",
            "install_profile": "cuda-cu130",
            "runtime_status": "blocked",
            "missing_packages": ["torch"],
            "sft_dataset": {"path": "mixed.jsonl", "examples": 3},
            "lora_dry_run_passed": True,
            "recommended_next_command": "python install",
            "steps": {"build_sft": "passed", "page_experiment": "blocked"},
        },
        [
            AcceptanceCheck(
                id=1,
                requirement="Requirement one",
                status="passed",
                evidence=["file.py"],
                notes="ok",
            )
        ],
    )

    assert "# Project readiness report" in report
    assert "- acceptance: 13 passed, 4 partial, 0 failed" in report
    assert "- generated at: `2026-06-29T00:00:00+00:00`" in report
    assert "| `build_sft` | `passed` |" in report
    assert "| 1 | `passed` | Requirement one | ok |" in report
    assert "Final Status" in report
    assert "open acceptance or execution-plan items" in report


def test_render_report_marks_project_ready_when_everything_passed() -> None:
    report = render_report(
        {
            "acceptance": {"passed": 17, "partial": 0, "failed": 0},
            "generated_at": "2026-06-29T00:00:00+00:00",
            "model": "Qwen/test",
            "install_profile": "cuda-cu130",
            "runtime_status": "passed",
            "missing_packages": [],
            "sft_dataset": {"path": "mixed.jsonl", "examples": 3},
            "lora_dry_run_passed": True,
            "recommended_next_command": "All Qwen execution plan steps passed.",
            "steps": {
                "install_ml": "passed",
                "qwen_smoke": "passed",
                "build_sft": "passed",
                "lora_dry_run": "passed",
                "page_experiment": "passed",
                "lora_train": "passed",
            },
        },
        [
            AcceptanceCheck(
                id=1,
                requirement="Requirement one",
                status="passed",
                evidence=["file.py"],
                notes="ok",
            )
        ],
    )

    assert "All acceptance criteria and all Qwen execution plan steps passed" in report
