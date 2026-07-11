import json

import pytest

from scripts.audit_acceptance import (
    check_results,
    count_jsonl_examples,
    execution_step_passed,
    qwen_lora_adapter_exists,
    qwen_lora_dry_run_passed,
    render_markdown,
    ru,
)


def test_acceptance_audit_covers_all_final_spec_criteria():
    checks = check_results()

    assert len(checks) == 17
    assert {check.id for check in checks} == set(range(1, 18))
    assert all(check.status in {"passed", "partial", "failed"} for check in checks)


def test_acceptance_audit_has_no_static_failures():
    checks = check_results()

    assert not [check for check in checks if check.status == "failed"]


def test_acceptance_audit_reports_qwen_runtime_experiment_sft_and_dry_run_evidence():
    if not (
        count_jsonl_examples("data/training/qwen_sft_light/mixed.jsonl")
        and execution_step_passed("qwen_smoke")
        and execution_step_passed("page_experiment")
        and execution_step_passed("lora_train")
        and qwen_lora_adapter_exists()
    ):
        pytest.skip("Local Qwen training artifacts are not available.")

    checks = {check.id: check for check in check_results()}

    assert count_jsonl_examples("data/training/qwen_sft_light/mixed.jsonl") == 3
    assert execution_step_passed("qwen_smoke") is True
    assert execution_step_passed("page_experiment") is True
    assert execution_step_passed("lora_train") is True
    assert qwen_lora_dry_run_passed() is True
    assert qwen_lora_adapter_exists() is True
    assert "3 examples" in checks[13].notes
    assert "Qwen line mode was executed" in checks[7].notes
    assert "Real TrOCR/Qwen comparison" in checks[14].notes
    assert "LoRA training passed" in checks[16].notes


def test_acceptance_audit_markdown_is_readable_and_generated_from_checks():
    markdown = render_markdown(check_results())

    assert ru("\\u0410\\u0443\\u0434\\u0438\\u0442 \\u0433\\u043e\\u0442\\u043e\\u0432\\u043d\\u043e\\u0441\\u0442\\u0438") in markdown
    assert ru("\\u0412 \\u0438\\u043d\\u0442\\u0435\\u0440\\u0444\\u0435\\u0439\\u0441\\u0435 \\u0435\\u0441\\u0442\\u044c \\u0440\\u0435\\u0436\\u0438\\u043c \\u0440\\u0430\\u0441\\u043f\\u043e\\u0437\\u043d\\u0430\\u0432\\u0430\\u043d\\u0438\\u044f \\u043f\\u043e\\u043b\\u043d\\u043e\\u0439 \\u0441\\u0442\\u0440\\u0430\\u043d\\u0438\\u0446\\u044b.") in markdown
    assert ru("\\u0418\\u0442\\u043e\\u0433") in markdown
    assert "Рџ" not in markdown


def test_qwen_lora_dry_run_passed_reads_historical_runs(tmp_path):
    report_path = tmp_path / "qwen_execution_results.json"
    report_path.write_text(
        json.dumps(
            {
                "results": [{"id": "install_ml", "status": "blocked"}],
                "runs": [
                    {"created_at": None, "results": [{"id": "build_sft", "status": "passed"}]},
                    {"created_at": None, "results": [{"id": "lora_dry_run", "status": "passed"}]},
                ],
            }
        ),
        encoding="utf-8",
    )

    assert qwen_lora_dry_run_passed(str(report_path)) is True
