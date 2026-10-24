import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.qwen_project_status_service import QwenProjectStatusService
from scripts.audit_acceptance import check_results


DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "project_readiness_report.md"


def render_status_value(value: object) -> str:
    if value is None:
        return "none"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) or "none"
    return str(value)


def render_report(qwen_status: dict, acceptance_checks) -> str:
    acceptance = qwen_status["acceptance"]
    all_acceptance_passed = acceptance["partial"] == 0 and acceptance["failed"] == 0
    all_qwen_steps_passed = all(status == "passed" for status in qwen_status["steps"].values())
    lines = [
        "# Project readiness report",
        "",
        "## Summary",
        "",
        f"- acceptance: {acceptance['passed']} passed, {acceptance['partial']} partial, {acceptance['failed']} failed",
        f"- generated at: `{qwen_status['generated_at']}`",
        f"- Qwen model: `{render_status_value(qwen_status.get('model'))}`",
        f"- Qwen install profile: `{render_status_value(qwen_status.get('install_profile'))}`",
        f"- Qwen runtime status: `{qwen_status['runtime_status']}`",
        f"- missing packages: `{render_status_value(qwen_status['missing_packages'])}`",
        f"- SFT examples: `{qwen_status['sft_dataset']['examples']}`",
        f"- LoRA dry-run passed: `{qwen_status['lora_dry_run_passed']}`",
        f"- next command: `{qwen_status['recommended_next_command']}`",
        "",
        "## Qwen Steps",
        "",
        "| Step | Status |",
        "| --- | --- |",
    ]
    for step_id, status in qwen_status["steps"].items():
        lines.append(f"| `{step_id}` | `{status}` |")

    lines.extend(
        [
            "",
            "## Acceptance Criteria",
            "",
            "| N | Status | Requirement | Notes |",
            "| ---: | --- | --- | --- |",
        ]
    )
    for check in acceptance_checks:
        lines.append(
            f"| {check.id} | `{check.status}` | {check.requirement} | {check.notes or ''} |"
        )

    lines.extend(["", "## Final Status", ""])
    if all_acceptance_passed and all_qwen_steps_passed:
        lines.append(
            "All acceptance criteria and all Qwen execution plan steps passed. "
            "The project is ready for final review and demonstration."
        )
    else:
        lines.append(
            "The project still has open acceptance or execution-plan items. "
            "Use the next command above and the acceptance table to continue."
        )
    lines.append("")
    return "\n".join(lines)


def write_report(output_path: Path = DEFAULT_OUTPUT) -> str:
    qwen_status = QwenProjectStatusService().get_status()
    report = render_report(qwen_status, check_results())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Markdown readiness report for the project.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--print", action="store_true", help="Print the report after writing it.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = write_report(args.output)
    print(f"report: {args.output}")
    if args.print:
        print(report)


if __name__ == "__main__":
    main()
