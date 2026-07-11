import csv
from pathlib import Path

from PIL import Image

from app.ocr.base import BaseRecognizer, RecognitionResult
from app.image_processing.page_segmentation import PageSegmentationResult, TextLine
from app.services.page_evaluation_service import PageEvaluationService


class StubRecognizer(BaseRecognizer):
    model_name = "stub"

    def __init__(self, text: str) -> None:
        self.text = text

    def recognize(self, image_path: Path) -> RecognitionResult:
        return RecognitionResult(text=self.text, confidence=None, model_name=self.model_name)


class PageStubRecognizer(StubRecognizer):
    def recognize_page(self, image_path: Path) -> RecognitionResult:
        return RecognitionResult(text=self.text, confidence=None, model_name=self.model_name)

    def recognize_layout(self, image_path: Path) -> RecognitionResult:
        return RecognitionResult(
            text='[{"line_number": 1, "text": "layout text", "box": [1, 2, 3, 4]}]',
            confidence=None,
            model_name=self.model_name,
        )


class StubPageSegmenter:
    def segment(self, source_path: Path, output_dir: Path, annotated_image_path: Path) -> PageSegmentationResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        annotated_image_path.parent.mkdir(parents=True, exist_ok=True)
        line_path = output_dir / "line_000.png"
        Image.new("RGB", (120, 30), "white").save(line_path)
        Image.new("RGB", (120, 30), "white").save(annotated_image_path)
        return PageSegmentationResult(
            annotated_image_path=annotated_image_path,
            lines=[TextLine(index=0, image_path=line_path, bbox=(0, 0, 120, 30))],
        )


def test_page_evaluation_qwen_page_writes_metrics(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (80, 40), "white").save(image_path)
    text_path = tmp_path / "page.txt"
    text_path.write_text("правильный текст", encoding="utf-8")
    pages_csv = tmp_path / "pages.csv"
    pages_csv.write_text(
        f"image_path,text_path,page_id\n{image_path},{text_path},page_001\n",
        encoding="utf-8",
    )
    output_csv = tmp_path / "results.csv"

    summary = PageEvaluationService(
        trocr_recognizer=StubRecognizer(""),
        qwen_recognizer=PageStubRecognizer("правильный текст"),
    ).evaluate_pages_manifest(
        pages_csv_path=pages_csv,
        output_csv_path=output_csv,
        processed_dir=tmp_path / "processed",
        mode="qwen_page",
        preprocess=False,
    )

    assert summary.average_cer == 0
    assert summary.average_wer == 0
    with output_csv.open("r", encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    assert rows[0]["mode"] == "qwen_page"
    assert rows[0]["recognized_text"] == "правильный текст"


def test_page_evaluation_requires_qwen_for_qwen_modes(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (80, 40), "white").save(image_path)
    text_path = tmp_path / "page.txt"
    text_path.write_text("текст", encoding="utf-8")
    pages_csv = tmp_path / "pages.csv"
    pages_csv.write_text(
        f"image_path,text_path,page_id\n{image_path},{text_path},page_001\n",
        encoding="utf-8",
    )

    service = PageEvaluationService(trocr_recognizer=StubRecognizer(""))
    try:
        service.evaluate_pages_manifest(
            pages_csv_path=pages_csv,
            output_csv_path=tmp_path / "results.csv",
            processed_dir=tmp_path / "processed",
            mode="qwen_page",
            preprocess=False,
        )
    except ValueError as exc:
        assert "Qwen" in str(exc)
    else:
        raise AssertionError("Qwen mode should require qwen_recognizer")


def test_page_evaluation_supports_hybrid_mode(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (160, 80), "white").save(image_path)
    text_path = tmp_path / "page.txt"
    text_path.write_text("longer qwen text", encoding="utf-8")
    pages_csv = tmp_path / "pages.csv"
    pages_csv.write_text(
        f"image_path,text_path,page_id\n{image_path},{text_path},page_001\n",
        encoding="utf-8",
    )
    output_csv = tmp_path / "results.csv"

    summary = PageEvaluationService(
        trocr_recognizer=StubRecognizer("short"),
        qwen_recognizer=StubRecognizer("longer qwen text"),
        page_segmenter=StubPageSegmenter(),
    ).evaluate_pages_manifest(
        pages_csv_path=pages_csv,
        output_csv_path=output_csv,
        processed_dir=tmp_path / "processed",
        mode="hybrid",
        preprocess=False,
    )

    assert summary.items[0].mode == "hybrid"
    assert summary.items[0].line_count == 1
    assert "longer qwen text" in summary.items[0].recognized_text
    with output_csv.open("r", encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    assert rows[0]["mode"] == "hybrid"


def test_page_evaluation_supports_qwen_layout_mode(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (80, 40), "white").save(image_path)
    text_path = tmp_path / "page.txt"
    text_path.write_text("layout text", encoding="utf-8")
    pages_csv = tmp_path / "pages.csv"
    pages_csv.write_text(
        f"image_path,text_path,page_id\n{image_path},{text_path},page_001\n",
        encoding="utf-8",
    )
    output_csv = tmp_path / "results.csv"

    summary = PageEvaluationService(
        trocr_recognizer=StubRecognizer(""),
        qwen_recognizer=PageStubRecognizer("ignored"),
    ).evaluate_pages_manifest(
        pages_csv_path=pages_csv,
        output_csv_path=output_csv,
        processed_dir=tmp_path / "processed",
        mode="qwen_layout",
        preprocess=False,
    )

    assert summary.items[0].mode == "qwen_layout"
    assert '"box": [1, 2, 3, 4]' in summary.items[0].recognized_text
    with output_csv.open("r", encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    assert rows[0]["mode"] == "qwen_layout"
