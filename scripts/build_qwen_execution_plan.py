import argparse
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.qwen_runtime_check import DEFAULT_OUTPUT as RUNTIME_REPORT_PATH


DEFAULT_PLAN_JSON = PROJECT_ROOT / "data" / "runtime" / "qwen_execution_plan.json"
DEFAULT_PLAN_MD = PROJECT_ROOT / "docs" / "qwen_execution_plan.md"
DEFAULT_MODEL = "Qwen/Qwen3-VL-2B-Instruct"


def command_line(parts: list[str]) -> str:
    return subprocess.list2cmdline(parts)


def command_step(
    *,
    step_id: str,
    title: str,
    requires_network: bool,
    command_args: list[str],
    blocked_by: list[str] | None = None,
) -> dict:
    step = {
        "id": step_id,
        "title": title,
        "requires_network": requires_network,
        "command": command_line(command_args),
        "command_args": command_args,
    }
    if blocked_by is not None:
        step["blocked_by"] = blocked_by
    return step


def find_first_existing(candidates: list[Path]) -> Path | None:
    return next((candidate for candidate in candidates if candidate.exists()), None)


def load_runtime_report(path: Path = RUNTIME_REPORT_PATH) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def model_from_runtime_report(report: dict) -> str:
    recommendation = report.get("resources", {}).get("recommendation", {})
    return recommendation.get("model") or DEFAULT_MODEL


def install_profile_from_runtime_report(report: dict) -> str:
    gpus = report.get("resources", {}).get("gpus") or []
    return "cuda-cu130" if gpus else "cpu"


def qwen_runtime_ready(report: dict) -> bool:
    return bool(report.get("ready_for_qwen_inference")) or report.get("status") in {"ready", "passed"}


def discover_inputs(root: Path = PROJECT_ROOT) -> dict:
    pages_csv = find_first_existing(
        [
            root / "data" / "training" / "page_lines" / "test_pages.csv",
            root / "data" / "training" / "page_lines_smoke" / "test_pages.csv",
            root / "data" / "training" / "page_lines_experiment_smoke" / "test_pages.csv",
        ]
    )
    lines_csv = find_first_existing(
        [
            root / "data" / "training" / "page_lines" / "test_lines.csv",
            root / "data" / "training" / "page_lines_smoke" / "test_lines.csv",
            root / "data" / "evaluation" / "layout_export_smoke" / "lines.csv",
        ]
    )
    sft_jsonl = find_first_existing(
        [
            root / "data" / "training" / "qwen_sft_light" / "mixed.jsonl",
            root / "data" / "training" / "qwen_sft_smoke" / "mixed.jsonl",
            root / "data" / "evaluation" / "feedback_export_smoke" / "qwen_feedback.jsonl",
        ]
    )
    return {
        "pages_csv": str(pages_csv) if pages_csv else None,
        "lines_csv": str(lines_csv) if lines_csv else None,
        "sft_jsonl": str(sft_jsonl) if sft_jsonl else None,
    }


def build_plan(runtime_report: dict, inputs: dict, python_executable: str = "python") -> dict:
    model = model_from_runtime_report(runtime_report)
    install_profile = install_profile_from_runtime_report(runtime_report)
    pages_csv = inputs.get("pages_csv") or "<pages.csv>"
    lines_csv = inputs.get("lines_csv") or "<lines.csv>"
    sft_jsonl = inputs.get("sft_jsonl") or "data/training/qwen_sft_light/mixed.jsonl"
    missing_inputs = [name for name in ("pages_csv", "lines_csv") if not inputs.get(name)]
    qwen_runtime_blocker = [] if qwen_runtime_ready(runtime_report) else ["qwen_runtime"]

    steps = [
        command_step(
            step_id="install_ml",
            title="Install ML dependencies",
            requires_network=True,
            command_args=[
                python_executable,
                "scripts/setup_ml_environment.py",
                "--profile",
                install_profile,
                "--install",
            ],
        ),
        command_step(
            step_id="qwen_smoke",
            title="Run Qwen smoke inference",
            requires_network=True,
            command_args=[
                python_executable,
                "scripts/qwen_runtime_check.py",
                "--run-smoke",
                "--page",
                "--model",
                model,
                "--json",
            ],
        ),
        command_step(
            step_id="page_experiment",
            title="Compare TrOCR and Qwen page modes",
            requires_network=False,
            blocked_by=([] if inputs.get("pages_csv") else ["pages_csv"]) + qwen_runtime_blocker,
            command_args=[
                python_executable,
                "scripts/run_page_experiment.py",
                "--pages-csv",
                pages_csv,
                "--modes",
                "trocr_lines",
                "qwen_lines",
                "qwen_page",
                "hybrid",
                "--output-dir",
                "data/evaluation/page_experiment_qwen_light",
            ],
        ),
        command_step(
            step_id="build_sft",
            title="Build Qwen SFT JSONL",
            requires_network=False,
            blocked_by=missing_inputs,
            command_args=[
                python_executable,
                "scripts/build_qwen_sft_dataset.py",
                "--lines-csv",
                lines_csv,
                "--pages-csv",
                pages_csv,
                "--output",
                "data/training/qwen_sft_light/mixed.jsonl",
                "--include-layout",
            ],
        ),
        command_step(
            step_id="lora_dry_run",
            title="Validate Qwen SFT data before training",
            requires_network=False,
            blocked_by=[] if inputs.get("sft_jsonl") else ["sft_jsonl"],
            command_args=[
                python_executable,
                "scripts/train_qwen_vl_lora.py",
                "--dataset",
                sft_jsonl,
                "--model",
                model,
                "--dry-run",
                "--max-samples",
                "5",
            ],
        ),
        command_step(
            step_id="lora_train",
            title="Train Qwen LoRA adapter",
            requires_network=True,
            blocked_by=[] if inputs.get("sft_jsonl") else ["sft_jsonl"],
            command_args=[
                python_executable,
                "scripts/train_qwen_vl_lora.py",
                "--dataset",
                sft_jsonl,
                "--model",
                model,
                "--output-dir",
                "models/qwen-vl-ocr-lora",
                "--epochs",
                "1",
                "--batch-size",
                "1",
                "--gradient-accumulation-steps",
                "1",
                "--max-image-side",
                "512",
                "--gradient-checkpointing",
                "--fp16",
            ],
        ),
    ]
    return {
        "model": model,
        "install_profile": install_profile,
        "runtime_status": runtime_report.get("status", "unknown"),
        "runtime_missing_packages": runtime_report.get("missing_packages", []),
        "inputs": inputs,
        "missing_inputs": missing_inputs,
        "steps": steps,
    }


def render_markdown(plan: dict) -> str:
    lines = [
        "# Qwen execution plan",
        "",
        f"- model: `{plan['model']}`",
        f"- install profile: `{plan['install_profile']}`",
        f"- runtime status: `{plan['runtime_status']}`",
        f"- missing packages: `{', '.join(plan['runtime_missing_packages']) or 'none'}`",
        f"- missing inputs: `{', '.join(plan['missing_inputs']) or 'none'}`",
        "",
        "## Steps",
        "",
    ]
    for step in plan["steps"]:
        blocked_by = ", ".join(step.get("blocked_by", [])) or "none"
        lines.extend(
            [
                f"### {step['id']}",
                "",
                f"- title: {step['title']}",
                f"- requires network: `{step['requires_network']}`",
                f"- blocked by: `{blocked_by}`",
                "",
                "```bash",
                step["command"],
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def write_outputs(plan: dict, json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(plan), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a reproducible Qwen execution plan.")
    parser.add_argument("--runtime-report", type=Path, default=RUNTIME_REPORT_PATH)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_PLAN_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_PLAN_MD)
    parser.add_argument("--python", default=sys.executable, help="Python command to put into generated commands.")
    parser.add_argument("--json", action="store_true", help="Print the generated plan as JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime_report = load_runtime_report(args.runtime_report)
    plan = build_plan(runtime_report, discover_inputs(), python_executable=args.python)
    write_outputs(plan, args.output_json, args.output_md)
    if args.json:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
    else:
        print(f"model: {plan['model']}")
        print(f"install_profile: {plan['install_profile']}")
        print(f"runtime_status: {plan['runtime_status']}")
        print(f"missing_inputs: {', '.join(plan['missing_inputs']) or 'none'}")
        print(f"json: {args.output_json}")
        print(f"markdown: {args.output_md}")


if __name__ == "__main__":
    main()
