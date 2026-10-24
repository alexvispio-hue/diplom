import csv
from dataclasses import dataclass
from pathlib import Path

from app.ocr.base import BaseRecognizer
from app.services.page_evaluation_service import PageEvaluationService, PageEvaluationSummary
from app.services.recognition_service import PAGE_MODE_QWEN_LAYOUT, PAGE_MODE_QWEN_LINES, PAGE_MODE_QWEN_PAGE, PAGE_MODE_TROCR_LINES


@dataclass(frozen=True)
class PageExperimentResult:
    model_name: str
    mode: str
    dataset: str
    items: int
    average_cer: float
    average_wer: float
    average_page_time_ms: float
    average_line_time_ms: float
    output_csv_path: Path
    notes: str

    @property
    def average_processing_time_ms(self) -> float:
        return self.average_page_time_ms


class PageExperimentService:
    """Runs a reproducible comparison of full-page OCR modes."""

    def __init__(
        self,
        trocr_recognizer: BaseRecognizer,
        qwen_recognizer: BaseRecognizer | None = None,
        evaluator_factory=None,
    ) -> None:
        self.trocr_recognizer = trocr_recognizer
        self.qwen_recognizer = qwen_recognizer
        self.evaluator_factory = evaluator_factory or self._build_evaluator

    def run(
        self,
        pages_csv_path: Path,
        output_dir: Path,
        modes: list[str],
        preprocess: bool = True,
        limit: int | None = None,
    ) -> list[PageExperimentResult]:
        output_dir.mkdir(parents=True, exist_ok=True)
        processed_root = output_dir / "processed"
        results: list[PageExperimentResult] = []

        for mode in modes:
            self._validate_mode(mode)
            evaluator = self.evaluator_factory()
            mode_output = output_dir / f"{mode}.csv"
            summary = evaluator.evaluate_pages_manifest(
                pages_csv_path=pages_csv_path,
                output_csv_path=mode_output,
                processed_dir=processed_root / mode,
                mode=mode,
                preprocess=preprocess,
                limit=limit,
            )
            results.append(self._to_result(mode, summary, mode_output, pages_csv_path))

        self.write_summary_csv(output_dir / "summary.csv", results)
        self.write_summary_markdown(output_dir / "summary.md", pages_csv_path, results)
        return results

    def _build_evaluator(self) -> PageEvaluationService:
        return PageEvaluationService(
            trocr_recognizer=self.trocr_recognizer,
            qwen_recognizer=self.qwen_recognizer,
        )

    def _validate_mode(self, mode: str) -> None:
        if mode in {PAGE_MODE_QWEN_LINES, PAGE_MODE_QWEN_PAGE, PAGE_MODE_QWEN_LAYOUT} or mode == "hybrid":
            if self.qwen_recognizer is None:
                raise ValueError(f"Mode {mode} requires qwen_recognizer.")

    @staticmethod
    def _to_result(
        mode: str,
        summary: PageEvaluationSummary,
        output_csv_path: Path,
        pages_csv_path: Path,
    ) -> PageExperimentResult:
        model_name = PageExperimentService._summarize_model_names(summary)
        total_lines = sum(item.line_count for item in summary.items)
        total_time = sum(item.processing_time_ms for item in summary.items)
        average_line_time_ms = (total_time / total_lines) if total_lines else 0.0
        return PageExperimentResult(
            model_name=model_name,
            mode=mode,
            dataset=str(pages_csv_path),
            items=len(summary.items),
            average_cer=summary.average_cer,
            average_wer=summary.average_wer,
            average_page_time_ms=summary.average_processing_time_ms,
            average_line_time_ms=average_line_time_ms,
            output_csv_path=output_csv_path,
            notes=f"items={len(summary.items)}; output={output_csv_path}",
        )

    @staticmethod
    def _summarize_model_names(summary: PageEvaluationSummary) -> str:
        model_names = sorted({item.model_name for item in summary.items if item.model_name})
        if not model_names:
            return "unknown"
        if len(model_names) == 1:
            return model_names[0]
        return " + ".join(model_names)

    @staticmethod
    def write_summary_csv(path: Path, results: list[PageExperimentResult]) -> None:
        with path.open("w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(
                target,
                fieldnames=["model", "mode", "dataset", "CER", "WER", "page_time", "line_time", "notes"],
            )
            writer.writeheader()
            for result in results:
                writer.writerow(
                    {
                        "model": result.model_name,
                        "mode": result.mode,
                        "dataset": result.dataset,
                        "CER": f"{result.average_cer:.6f}",
                        "WER": f"{result.average_wer:.6f}",
                        "page_time": f"{result.average_page_time_ms:.2f}",
                        "line_time": f"{result.average_line_time_ms:.2f}",
                        "notes": result.notes,
                    }
                )

    @staticmethod
    def write_summary_markdown(path: Path, pages_csv_path: Path, results: list[PageExperimentResult]) -> None:
        lines = [
            "# Page OCR experiment",
            "",
            f"Dataset: `{pages_csv_path}`",
            "",
            "| Model | Mode | Items | CER | WER | Page time, ms | Line time, ms |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for result in results:
            lines.append(
                f"| `{result.model_name}` | `{result.mode}` | {result.items} | {result.average_cer:.4f} | "
                f"{result.average_wer:.4f} | {result.average_page_time_ms:.1f} | {result.average_line_time_ms:.1f} |"
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
