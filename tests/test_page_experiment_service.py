import csv
from pathlib import Path

import pytest

from app.ocr.base import BaseRecognizer, RecognitionResult
from app.ocr.metrics import TextRecognitionMetrics
from app.services.page_evaluation_service import PageEvaluationItem, PageEvaluationSummary
from app.services.page_experiment_service import PageExperimentService


class StubRecognizer(BaseRecognizer):
    model_name = "stub"

    def recognize(self, image_path: Path) -> RecognitionResult:
        return RecognitionResult(text="stub", confidence=None, model_name=self.model_name)


class StubEvaluator:
    def evaluate_pages_manifest(
        self,
        pages_csv_path: Path,
        output_csv_path: Path,
        processed_dir: Path,
        mode: str,
        preprocess: bool = True,
        limit: int | None = None,
    ) -> PageEvaluationSummary:
        output_csv_path.write_text("mode,recognized_text\n" f"{mode},text\n", encoding="utf-8")
        item = PageEvaluationItem(
            image_path=Path("page.png"),
            page_id="page_001",
            reference_text="text",
            recognized_text="text",
            metrics=TextRecognitionMetrics(
                character_error_rate=0.1,
                word_error_rate=0.2,
                character_distance=1,
                word_distance=1,
            ),
            model_name="stub",
            mode=mode,
            line_count=1,
            processing_time_ms=42,
        )
        return PageEvaluationSummary(
            items=[item],
            average_cer=0.1,
            average_wer=0.2,
            average_processing_time_ms=42,
        )


def test_page_experiment_writes_mode_results_and_summary(tmp_path: Path) -> None:
    pages_csv = tmp_path / "pages.csv"
    pages_csv.write_text("image_path,text_path,page_id\n", encoding="utf-8")
    output_dir = tmp_path / "experiment"

    results = PageExperimentService(
        trocr_recognizer=StubRecognizer(),
        evaluator_factory=lambda: StubEvaluator(),
    ).run(
        pages_csv_path=pages_csv,
        output_dir=output_dir,
        modes=["trocr_lines"],
        preprocess=False,
    )

    assert len(results) == 1
    assert (output_dir / "trocr_lines.csv").exists()
    assert (output_dir / "summary.csv").exists()
    with (output_dir / "summary.csv").open("r", encoding="utf-8", newline="") as source:
        summary_rows = list(csv.DictReader(source))
    assert summary_rows[0] == {
        "model": "stub",
        "mode": "trocr_lines",
        "dataset": str(pages_csv),
        "CER": "0.100000",
        "WER": "0.200000",
        "page_time": "42.00",
        "line_time": "42.00",
        "notes": f"items=1; output={output_dir / 'trocr_lines.csv'}",
    }
    summary_md = (output_dir / "summary.md").read_text(encoding="utf-8")
    assert "`trocr_lines`" in summary_md
    assert "`stub`" in summary_md
    assert "0.1000" in summary_md


def test_page_experiment_requires_qwen_for_qwen_modes(tmp_path: Path) -> None:
    pages_csv = tmp_path / "pages.csv"
    pages_csv.write_text("image_path,text_path,page_id\n", encoding="utf-8")

    with pytest.raises(ValueError, match="qwen_recognizer"):
        PageExperimentService(
            trocr_recognizer=StubRecognizer(),
            evaluator_factory=lambda: StubEvaluator(),
        ).run(
            pages_csv_path=pages_csv,
            output_dir=tmp_path / "experiment",
            modes=["hybrid"],
        )
