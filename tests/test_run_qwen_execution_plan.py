import json
import subprocess

import pytest

from scripts.run_qwen_execution_plan import command_for_step, evaluate_step, run_step, select_steps, write_results


def plan() -> dict:
    return {
        "steps": [
            {"id": "install_ml", "title": "Install", "requires_network": True, "command": "python install.py"},
            {"id": "build_sft", "title": "Build SFT", "requires_network": False, "command": "python build.py"},
        ]
    }


def test_select_steps_can_pick_one_or_all() -> None:
    selected = select_steps(plan(), "build_sft", run_all=False)
    all_steps = select_steps(plan(), None, run_all=True)

    assert [step["id"] for step in selected] == ["build_sft"]
    assert [step["id"] for step in all_steps] == ["install_ml", "build_sft"]


def test_select_steps_rejects_unknown_step() -> None:
    with pytest.raises(ValueError, match="Unknown step"):
        select_steps(plan(), "missing", run_all=False)


def test_evaluate_step_blocks_network_without_flag() -> None:
    status, reason = evaluate_step(plan()["steps"][0], allow_network=False)

    assert status == "blocked"
    assert "--allow-network" in reason


def test_evaluate_step_blocks_unresolved_inputs() -> None:
    status, reason = evaluate_step(
        {"id": "build_sft", "requires_network": False, "blocked_by": ["pages_csv"]},
        allow_network=True,
    )

    assert status == "blocked"
    assert "pages_csv" in reason


def test_run_step_dry_run_does_not_call_runner() -> None:
    def fail_runner(*args, **kwargs):  # pragma: no cover - called only on regression
        raise AssertionError("runner should not be called")

    result = run_step(plan()["steps"][1], allow_network=False, dry_run=True, runner=fail_runner)

    assert result["status"] == "dry_run"
    assert result["returncode"] is None


def test_run_step_executes_ready_command_with_injected_runner() -> None:
    captured = {}

    def fake_runner(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="ok", stderr="")

    step = {
        "id": "build_sft",
        "title": "Build SFT",
        "requires_network": False,
        "command": "python build.py",
        "command_args": ["python", "build.py"],
    }
    result = run_step(step, allow_network=False, dry_run=False, runner=fake_runner)

    assert result["status"] == "passed"
    assert result["returncode"] == 0
    assert result["stdout"] == "ok"
    assert captured["args"][0] == ["python", "build.py"]
    assert captured["kwargs"]["shell"] is False


def test_command_for_step_falls_back_to_shell_command_string() -> None:
    step = {"command": "python build.py"}

    assert command_for_step(step) == "python build.py"


def test_write_results_preserves_previous_runs(tmp_path) -> None:
    results_path = tmp_path / "results.json"

    write_results([{"id": "build_sft", "status": "passed"}], results_path)
    write_results([{"id": "lora_dry_run", "status": "passed"}], results_path)

    payload = json.loads(results_path.read_text(encoding="utf-8"))
    assert payload["results"] == [{"id": "lora_dry_run", "status": "passed"}]
    assert [run["results"][0]["id"] for run in payload["runs"]] == ["build_sft", "lora_dry_run"]
