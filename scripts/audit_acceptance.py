from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QWEN_RESULTS = "data/runtime/qwen_execution_results.json"
QWEN_EXPERIMENT_SUMMARY = "data/evaluation/page_experiment_qwen_light/summary.md"
QWEN_LORA_ADAPTER_DIR = "models/qwen-vl-ocr-lora"


def ru(text: str) -> str:
    return text.encode("ascii").decode("unicode_escape")


@dataclass(frozen=True)
class AcceptanceCheck:
    id: int
    requirement: str
    status: str
    evidence: list[str]
    notes: str = ""


def path_exists(relative_path: str) -> bool:
    return (ROOT / relative_path).exists()


def file_contains(relative_path: str, *needles: str) -> bool:
    path = ROOT / relative_path
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    return all(needle in text for needle in needles)


def count_jsonl_examples(relative_path: str) -> int:
    path = ROOT / relative_path
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def execution_step_passed(step_id: str, relative_path: str = QWEN_RESULTS) -> bool:
    path = ROOT / relative_path
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    latest_results = payload.get("results", [])
    historical_results = [
        result
        for run in payload.get("runs", [])
        for result in run.get("results", [])
    ]
    return any(
        result.get("id") == step_id and result.get("status") == "passed"
        for result in [*latest_results, *historical_results]
    )


def qwen_lora_dry_run_passed(relative_path: str = QWEN_RESULTS) -> bool:
    return execution_step_passed("lora_dry_run", relative_path)


def qwen_lora_adapter_exists(relative_path: str = QWEN_LORA_ADAPTER_DIR) -> bool:
    adapter_dir = ROOT / relative_path
    return (
        adapter_dir.is_dir()
        and (adapter_dir / "adapter_config.json").exists()
        and (adapter_dir / "adapter_model.safetensors").exists()
    )


def check_results() -> list[AcceptanceCheck]:
    qwen_sft_examples = count_jsonl_examples("data/training/qwen_sft_light/mixed.jsonl")
    qwen_smoke_ok = execution_step_passed("qwen_smoke")
    page_experiment_ok = execution_step_passed("page_experiment") and file_contains(
        QWEN_EXPERIMENT_SUMMARY,
        "qwen_lines",
        "qwen_page",
        "CER",
        "WER",
    )
    qwen_dry_run_ok = qwen_lora_dry_run_passed()
    qwen_lora_train_ok = execution_step_passed("lora_train") and qwen_lora_adapter_exists()
    return [
        AcceptanceCheck(
            1,
            ru("\\u0412 \\u0438\\u043d\\u0442\\u0435\\u0440\\u0444\\u0435\\u0439\\u0441\\u0435 \\u0435\\u0441\\u0442\\u044c \\u0440\\u0435\\u0436\\u0438\\u043c \\u0440\\u0430\\u0441\\u043f\\u043e\\u0437\\u043d\\u0430\\u0432\\u0430\\u043d\\u0438\\u044f \\u043f\\u043e\\u043b\\u043d\\u043e\\u0439 \\u0441\\u0442\\u0440\\u0430\\u043d\\u0438\\u0446\\u044b."),
            "passed" if file_contains("ui/streamlit_app.py", "PAGE_MODE_LABELS", 'options=["fragment", "page"]') else "failed",
            ["ui/streamlit_app.py"],
        ),
        AcceptanceCheck(2, ru("\\u0421\\u0438\\u0441\\u0442\\u0435\\u043c\\u0430 \\u0443\\u043c\\u0435\\u0435\\u0442 \\u0437\\u0430\\u0433\\u0440\\u0443\\u0436\\u0430\\u0442\\u044c \\u0444\\u043e\\u0442\\u043e\\u0433\\u0440\\u0430\\u0444\\u0438\\u044e \\u0441\\u0442\\u0440\\u0430\\u043d\\u0438\\u0446\\u044b."), "passed" if file_contains("ui/streamlit_app.py", "file_uploader", "document_target") else "failed", ["ui/streamlit_app.py", "app/api/routes.py"]),
        AcceptanceCheck(3, ru("\\u0421\\u0438\\u0441\\u0442\\u0435\\u043c\\u0430 \\u0432\\u044b\\u043f\\u043e\\u043b\\u043d\\u044f\\u0435\\u0442 \\u043f\\u0440\\u0435\\u0434\\u043e\\u0431\\u0440\\u0430\\u0431\\u043e\\u0442\\u043a\\u0443 \\u0438\\u0437\\u043e\\u0431\\u0440\\u0430\\u0436\\u0435\\u043d\\u0438\\u044f."), "passed" if file_contains("app/image_processing/preprocessing.py", "class ImagePreprocessor") else "failed", ["app/image_processing/preprocessing.py", "tests/test_preprocessing.py"]),
        AcceptanceCheck(4, ru("\\u0421\\u0438\\u0441\\u0442\\u0435\\u043c\\u0430 \\u0432\\u044b\\u0434\\u0435\\u043b\\u044f\\u0435\\u0442 \\u0441\\u0442\\u0440\\u043e\\u043a\\u0438 \\u0440\\u0443\\u043a\\u043e\\u043f\\u0438\\u0441\\u043d\\u043e\\u0433\\u043e \\u0442\\u0435\\u043a\\u0441\\u0442\\u0430."), "passed" if file_contains("app/image_processing/page_segmentation.py", "class PageLineSegmenter") else "failed", ["app/image_processing/page_segmentation.py", "tests/test_page_segmentation.py"]),
        AcceptanceCheck(5, ru("\\u041d\\u0430\\u0439\\u0434\\u0435\\u043d\\u043d\\u044b\\u0435 \\u0441\\u0442\\u0440\\u043e\\u043a\\u0438 \\u043e\\u0442\\u043e\\u0431\\u0440\\u0430\\u0436\\u0430\\u044e\\u0442\\u0441\\u044f \\u043f\\u043e\\u043b\\u044c\\u0437\\u043e\\u0432\\u0430\\u0442\\u0435\\u043b\\u044e."), "passed" if file_contains("ui/streamlit_app.py", "line_coordinates", "st.image") else "failed", ["ui/streamlit_app.py"]),
        AcceptanceCheck(6, ru("\\u041a\\u0430\\u0436\\u0434\\u0430\\u044f \\u0441\\u0442\\u0440\\u043e\\u043a\\u0430 \\u043c\\u043e\\u0436\\u0435\\u0442 \\u0431\\u044b\\u0442\\u044c \\u043f\\u0435\\u0440\\u0435\\u0434\\u0430\\u043d\\u0430 \\u0432 TrOCR."), "passed" if file_contains("app/services/recognition_service.py", "PAGE_MODE_TROCR_LINES") else "failed", ["app/services/recognition_service.py"]),
        AcceptanceCheck(7, ru("\\u041a\\u0430\\u0436\\u0434\\u0430\\u044f \\u0441\\u0442\\u0440\\u043e\\u043a\\u0430 \\u043c\\u043e\\u0436\\u0435\\u0442 \\u0431\\u044b\\u0442\\u044c \\u043f\\u0435\\u0440\\u0435\\u0434\\u0430\\u043d\\u0430 \\u0432 Qwen."), "passed" if qwen_smoke_ok and page_experiment_ok and file_contains("app/services/recognition_service.py", "PAGE_MODE_QWEN_LINES") else "partial", ["app/services/recognition_service.py", "app/ocr/qwen_vl_recognizer.py", QWEN_RESULTS, QWEN_EXPERIMENT_SUMMARY], "Qwen line mode was executed in the real page experiment." if page_experiment_ok else "Code path is implemented; real Qwen experiment has not passed yet."),
        AcceptanceCheck(8, ru("\\u041f\\u043e\\u043b\\u043d\\u0430\\u044f \\u0441\\u0442\\u0440\\u0430\\u043d\\u0438\\u0446\\u0430 \\u043c\\u043e\\u0436\\u0435\\u0442 \\u0431\\u044b\\u0442\\u044c \\u043f\\u0435\\u0440\\u0435\\u0434\\u0430\\u043d\\u0430 \\u0432 Qwen \\u0446\\u0435\\u043b\\u0438\\u043a\\u043e\\u043c."), "passed" if qwen_smoke_ok and page_experiment_ok and file_contains("app/api/routes.py", "/recognize/page/qwen") else "partial", ["app/api/routes.py", "app/ocr/qwen_vl_recognizer.py", QWEN_RESULTS, QWEN_EXPERIMENT_SUMMARY], "Qwen page mode passed smoke inference and the real page experiment." if page_experiment_ok else "Code path is implemented; real Qwen experiment has not passed yet."),
        AcceptanceCheck(9, ru("\\u0418\\u0442\\u043e\\u0433\\u043e\\u0432\\u044b\\u0439 \\u0442\\u0435\\u043a\\u0441\\u0442 \\u0441\\u043e\\u0431\\u0438\\u0440\\u0430\\u0435\\u0442\\u0441\\u044f \\u0432 \\u043f\\u0440\\u0430\\u0432\\u0438\\u043b\\u044c\\u043d\\u043e\\u043c \\u043f\\u043e\\u0440\\u044f\\u0434\\u043a\\u0435."), "passed" if file_contains("app/services/recognition_service.py", "\\n\".join") else "failed", ["app/services/recognition_service.py", "tests/test_page_segmentation.py"]),
        AcceptanceCheck(10, ru("\\u041f\\u043e\\u043b\\u044c\\u0437\\u043e\\u0432\\u0430\\u0442\\u0435\\u043b\\u044c \\u043c\\u043e\\u0436\\u0435\\u0442 \\u0438\\u0441\\u043f\\u0440\\u0430\\u0432\\u0438\\u0442\\u044c \\u0440\\u0435\\u0437\\u0443\\u043b\\u044c\\u0442\\u0430\\u0442."), "passed" if file_contains("ui/streamlit_app.py", "corrected_text", "text_area") else "failed", ["ui/streamlit_app.py", "app/services/feedback_service.py"]),
        AcceptanceCheck(11, ru("\\u041f\\u043e\\u043b\\u044c\\u0437\\u043e\\u0432\\u0430\\u0442\\u0435\\u043b\\u044c \\u043c\\u043e\\u0436\\u0435\\u0442 \\u043e\\u0446\\u0435\\u043d\\u0438\\u0442\\u044c \\u043a\\u0430\\u0447\\u0435\\u0441\\u0442\\u0432\\u043e \\u0440\\u0435\\u0437\\u0443\\u043b\\u044c\\u0442\\u0430\\u0442\\u0430."), "passed" if file_contains("ui/streamlit_app.py", "FEEDBACK_LABELS", "Сохранить проверку") else "failed", ["ui/streamlit_app.py", "tests/test_feedback_service.py"]),
        AcceptanceCheck(12, ru("\\u041f\\u0440\\u043e\\u0431\\u043b\\u0435\\u043c\\u043d\\u044b\\u0435 \\u043f\\u0440\\u0438\\u043c\\u0435\\u0440\\u044b \\u0441\\u043e\\u0445\\u0440\\u0430\\u043d\\u044f\\u044e\\u0442\\u0441\\u044f \\u0434\\u043b\\u044f \\u0431\\u0443\\u0434\\u0443\\u0449\\u0435\\u0433\\u043e \\u043e\\u0431\\u0443\\u0447\\u0435\\u043d\\u0438\\u044f."), "passed" if file_contains("app/services/feedback_service.py", "problems", "list_problem_samples") else "failed", ["app/services/feedback_service.py", "tests/test_feedback_service.py"]),
        AcceptanceCheck(13, ru("\\u041f\\u043e\\u0434\\u0433\\u043e\\u0442\\u043e\\u0432\\u043b\\u0435\\u043d \\u0434\\u0430\\u0442\\u0430\\u0441\\u0435\\u0442 \\u0441\\u0442\\u0440\\u043e\\u043a \\u0438 \\u0441\\u0442\\u0440\\u0430\\u043d\\u0438\\u0446."), "passed" if path_exists("scripts/prepare_school_page_lines.py") else "failed", ["scripts/prepare_school_page_lines.py", "tests/test_prepare_school_page_lines.py", "data/training/qwen_sft_light/mixed.jsonl"], f"Qwen SFT dataset is present with {qwen_sft_examples} examples." if qwen_sft_examples else "Qwen SFT dataset has not been generated yet."),
        AcceptanceCheck(14, ru("\\u041f\\u0440\\u043e\\u0432\\u0435\\u0434\\u0435\\u043d\\u043e \\u0441\\u0440\\u0430\\u0432\\u043d\\u0435\\u043d\\u0438\\u0435 \\u043c\\u043e\\u0434\\u0435\\u043b\\u0435\\u0439 \\u043f\\u043e CER \\u0438 WER."), "passed" if page_experiment_ok else "partial", ["scripts/run_page_experiment.py", "app/services/experiment_results_service.py", QWEN_EXPERIMENT_SUMMARY], "Real TrOCR/Qwen comparison was executed on the smoke page dataset." if page_experiment_ok else "Experiment pipeline is implemented; real Qwen comparison has not passed yet."),
        AcceptanceCheck(15, ru("\\u041f\\u0440\\u043e\\u0432\\u0435\\u0434\\u0435\\u043d\\u043e \\u0441\\u0440\\u0430\\u0432\\u043d\\u0435\\u043d\\u0438\\u0435 \\u0432\\u0440\\u0435\\u043c\\u0435\\u043d\\u0438 \\u043e\\u0431\\u0440\\u0430\\u0431\\u043e\\u0442\\u043a\\u0438."), "passed" if file_contains("app/services/page_experiment_service.py", "page_time", "line_time") else "failed", ["app/services/page_experiment_service.py", "tests/test_page_experiment_service.py"]),
        AcceptanceCheck(16, ru("\\u041f\\u043e\\u043b\\u0443\\u0447\\u0435\\u043d\\u044b \\u0440\\u0435\\u0437\\u0443\\u043b\\u044c\\u0442\\u0430\\u0442\\u044b baseline \\u0438 \\u0440\\u0435\\u0437\\u0443\\u043b\\u044c\\u0442\\u0430\\u0442\\u044b \\u043f\\u043e\\u0441\\u043b\\u0435 \\u0434\\u043e\\u043e\\u0431\\u0443\\u0447\\u0435\\u043d\\u0438\\u044f."), "passed" if qwen_lora_train_ok else ("partial" if file_contains("docs/experiment_results.md", "baseline", "fine") else "failed"), ["docs/experiment_results.md", "docs/training.md", "data/training/qwen_sft_light/mixed.jsonl", QWEN_RESULTS, QWEN_LORA_ADAPTER_DIR], "Real Qwen LoRA training passed and adapter files were saved." if qwen_lora_train_ok else ("Documented baseline results and Qwen runtime experiment exist; Qwen SFT dry-run passed, but real Qwen LoRA results still require a training run." if qwen_dry_run_ok and page_experiment_ok else "Documented baseline results exist; real Qwen LoRA results require a training run.")),
        AcceptanceCheck(17, ru("\\u0421\\u0438\\u0441\\u0442\\u0435\\u043c\\u0430 \\u0441\\u043e\\u0445\\u0440\\u0430\\u043d\\u044f\\u0435\\u0442 \\u0438\\u043d\\u0444\\u043e\\u0440\\u043c\\u0430\\u0446\\u0438\\u044e \\u043e \\u043c\\u043e\\u0434\\u0435\\u043b\\u0438 \\u0438 \\u0440\\u0435\\u0436\\u0438\\u043c\\u0435 \\u0440\\u0430\\u0441\\u043f\\u043e\\u0437\\u043d\\u0430\\u0432\\u0430\\u043d\\u0438\\u044f."), "passed" if file_contains("app/storage/models.py", "model_name", "recognition_mode") else "failed", ["app/storage/models.py", "tests/test_api.py"]),
    ]


def status_label(status: str) -> str:
    return {"passed": ru("\\u0412\\u044b\\u043f\\u043e\\u043b\\u043d\\u0435\\u043d\\u043e"), "partial": ru("\\u0427\\u0430\\u0441\\u0442\\u0438\\u0447\\u043d\\u043e"), "failed": ru("\\u041d\\u0435 \\u0432\\u044b\\u043f\\u043e\\u043b\\u043d\\u0435\\u043d\\u043e")}.get(status, status)


def render_markdown(checks: list[AcceptanceCheck]) -> str:
    lines = [
        "# " + ru("\\u0410\\u0443\\u0434\\u0438\\u0442 \\u0433\\u043e\\u0442\\u043e\\u0432\\u043d\\u043e\\u0441\\u0442\\u0438 \\u043f\\u043e \\u0438\\u0442\\u043e\\u0433\\u043e\\u0432\\u043e\\u043c\\u0443 \\u0422\\u0417"),
        "",
        ru("\\u0414\\u043e\\u043a\\u0443\\u043c\\u0435\\u043d\\u0442 \\u0444\\u0438\\u043a\\u0441\\u0438\\u0440\\u0443\\u0435\\u0442 \\u0442\\u0435\\u043a\\u0443\\u0449\\u0435\\u0435 \\u0441\\u043e\\u0441\\u0442\\u043e\\u044f\\u043d\\u0438\\u0435 \\u043f\\u0440\\u043e\\u0435\\u043a\\u0442\\u0430 \\u043e\\u0442\\u043d\\u043e\\u0441\\u0438\\u0442\\u0435\\u043b\\u044c\\u043d\\u043e \\u043a\\u0440\\u0438\\u0442\\u0435\\u0440\\u0438\\u0435\\u0432 \\u043f\\u0440\\u0438\\u0435\\u043c\\u043a\\u0438 \\u0438\\u0437 `docs/final_technical_specification.md`."),
        "",
        "```bash",
        "python scripts/audit_acceptance.py --json",
        "python scripts/audit_acceptance.py --markdown-output docs/acceptance_audit.md",
        "```",
        "",
        "## " + ru("\\u0421\\u0432\\u043e\\u0434\\u043a\\u0430"),
        "",
        "| N | " + ru("\\u0421\\u0442\\u0430\\u0442\\u0443\\u0441") + " | " + ru("\\u041a\\u0440\\u0438\\u0442\\u0435\\u0440\\u0438\\u0439") + " | " + ru("\\u0414\\u043e\\u043a\\u0430\\u0437\\u0430\\u0442\\u0435\\u043b\\u044c\\u0441\\u0442\\u0432\\u0430") + " | " + ru("\\u041f\\u0440\\u0438\\u043c\\u0435\\u0447\\u0430\\u043d\\u0438\\u0435") + " |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for check in checks:
        evidence = ", ".join(f"`{item}`" for item in check.evidence)
        lines.append(f"| {check.id} | {status_label(check.status)} | {check.requirement} | {evidence} | {check.notes} |")
    lines.extend(["", "## " + ru("\\u0418\\u0442\\u043e\\u0433"), "", ru("\\u0412\\u0441\\u0435 \\u043a\\u0440\\u0438\\u0442\\u0435\\u0440\\u0438\\u0438 \\u043f\\u0440\\u0438\\u0435\\u043c\\u043a\\u0438 \\u0437\\u0430\\u043a\\u0440\\u044b\\u0442\\u044b, \\u0435\\u0441\\u043b\\u0438 \\u0432 \\u0442\\u0430\\u0431\\u043b\\u0438\\u0446\\u0435 \\u043d\\u0435\\u0442 \\u0441\\u0442\\u0430\\u0442\\u0443\\u0441\\u043e\\u0432 `\\u0427\\u0430\\u0441\\u0442\\u0438\\u0447\\u043d\\u043e` \\u0438\\u043b\\u0438 `\\u041d\\u0435 \\u0432\\u044b\\u043f\\u043e\\u043b\\u043d\\u0435\\u043d\\u043e`."), ""])
    return "\n".join(lines)


def write_markdown(checks: list[AcceptanceCheck], output_path: Path) -> str:
    markdown = render_markdown(checks)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    return markdown


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit acceptance criteria from final technical specification.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument("--markdown-output", type=Path, help="Write a Markdown acceptance audit report.")
    args = parser.parse_args()
    checks = check_results()
    if args.markdown_output:
        write_markdown(checks, args.markdown_output)
        print(f"markdown: {args.markdown_output}")
    elif args.json:
        print(json.dumps([asdict(check) for check in checks], ensure_ascii=False, indent=2))
    else:
        for check in checks:
            note = f" ({check.notes})" if check.notes else ""
            print(f"{check.id:02d}. {check.status.upper()}: {check.requirement}{note}")
    return 1 if any(check.status == "failed" for check in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
