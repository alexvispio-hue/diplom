from scripts.build_qwen_execution_plan import build_plan, qwen_runtime_ready, render_markdown


def runtime_report() -> dict:
    return {
        "status": "blocked",
        "missing_packages": ["torch", "transformers"],
        "resources": {
            "gpus": [{"name": "RTX", "memory_total_mb": 16303}],
            "recommendation": {"model": "Qwen/Qwen3-VL-2B-Instruct", "preset": "light"},
        },
    }


def ready_runtime_report() -> dict:
    report = runtime_report()
    report["status"] = "ready"
    report["ready_for_qwen_inference"] = True
    report["missing_packages"] = []
    return report


def test_build_plan_uses_runtime_model_and_cuda_profile() -> None:
    plan = build_plan(
        runtime_report(),
        {
            "pages_csv": "data/training/page_lines/test_pages.csv",
            "lines_csv": "data/training/page_lines/test_lines.csv",
            "sft_jsonl": "data/training/qwen_sft_light/mixed.jsonl",
        },
        python_executable="python",
    )

    assert plan["model"] == "Qwen/Qwen3-VL-2B-Instruct"
    assert plan["install_profile"] == "cuda-cu130"
    assert plan["missing_inputs"] == []
    assert "--profile cuda-cu130 --install" in plan["steps"][0]["command"]
    assert plan["steps"][0]["command_args"] == [
        "python",
        "scripts/setup_ml_environment.py",
        "--profile",
        "cuda-cu130",
        "--install",
    ]
    assert "--model Qwen/Qwen3-VL-2B-Instruct" in plan["steps"][1]["command"]
    assert "--max-image-side 512 --gradient-checkpointing --fp16" in {step["id"]: step for step in plan["steps"]}["lora_train"]["command"]
    assert {step["id"]: step for step in plan["steps"]}["page_experiment"]["blocked_by"] == ["qwen_runtime"]


def test_build_plan_marks_missing_page_and_sft_inputs() -> None:
    plan = build_plan(
        runtime_report(),
        {"pages_csv": None, "lines_csv": "data/evaluation/layout_export_smoke/lines.csv", "sft_jsonl": None},
        python_executable="python",
    )

    steps = {step["id"]: step for step in plan["steps"]}
    assert plan["missing_inputs"] == ["pages_csv"]
    assert steps["page_experiment"]["blocked_by"] == ["pages_csv", "qwen_runtime"]
    assert steps["build_sft"]["blocked_by"] == ["pages_csv"]
    assert steps["lora_dry_run"]["blocked_by"] == ["sft_jsonl"]
    assert "<pages.csv>" in steps["page_experiment"]["command"]


def test_render_markdown_contains_commands_and_blockers() -> None:
    plan = build_plan(
        runtime_report(),
        {"pages_csv": None, "lines_csv": None, "sft_jsonl": None},
        python_executable="python",
    )

    markdown = render_markdown(plan)

    assert "# Qwen execution plan" in markdown
    assert "missing inputs: `pages_csv, lines_csv`" in markdown
    assert "scripts/setup_ml_environment.py" in markdown
    assert "blocked by: `pages_csv, qwen_runtime`" in markdown


def test_build_plan_unblocks_page_experiment_when_qwen_runtime_is_ready() -> None:
    plan = build_plan(
        ready_runtime_report(),
        {
            "pages_csv": "data/training/page_lines/test_pages.csv",
            "lines_csv": "data/training/page_lines/test_lines.csv",
            "sft_jsonl": "data/training/qwen_sft_light/mixed.jsonl",
        },
        python_executable="python",
    )

    assert qwen_runtime_ready(ready_runtime_report()) is True
    assert {step["id"]: step for step in plan["steps"]}["page_experiment"]["blocked_by"] == []
