import csv
import time
from dataclasses import dataclass
from pathlib import Path

from app.image_processing.page_segmentation import PageLineSegmenter
from app.image_processing.preprocessing import ImagePreprocessor
from app.ocr.base import BaseRecognizer, RecognitionResult
from app.ocr.metrics import TextRecognitionMetrics, calculate_metrics
from app.services.recognition_service import (
    PAGE_MODE_HYBRID,
    PAGE_MODE_QWEN_LAYOUT,
    PAGE_MODE_QWEN_LINES,
    PAGE_MODE_QWEN_PAGE,
    PAGE_MODE_TROCR_LINES,
    RecognitionService,
)


@dataclass(frozen=True)
class PageEvaluationItem:
    image_path: Path
    page_id: str
    reference_text: str
    recognized_text: str
    metrics: TextRecognitionMetrics
    model_name: str
    mode: str
    line_count: int
    processing_time_ms: int


@dataclass(frozen=True)
class PageEvaluationSummary:
    items: list[PageEvaluationItem]
    average_cer: float
    average_wer: float
    average_processing_time_ms: float


class PageEvaluationService:
    def __init__(
        self,
        trocr_recognizer: BaseRecognizer,
        qwen_recognizer: BaseRecognizer | None = None,
        preprocessor: ImagePreprocessor | None = None,
        page_segmenter: PageLineSegmenter | None = None,
    ) -> None:
        self.trocr_recognizer = trocr_recognizer
        self.qwen_recognizer = qwen_recognizer
        self.preprocessor = preprocessor or ImagePreprocessor()
        self.page_segmenter = page_segmenter or PageLineSegmenter()

    def evaluate_pages_manifest(
        self,
        pages_csv_path: Path,
        output_csv_path: Path,
        processed_dir: Path,
        mode: str = PAGE_MODE_TROCR_LINES,
        preprocess: bool = True,
        limit: int | None = None,
    ) -> PageEvaluationSummary:
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)
        items: list[PageEvaluationItem] = []

        with pages_csv_path.open("r", encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            for row in reader:
                if limit is not None and len(items) >= limit:
                    break
                image_path = Path(row["image_path"])
                if not image_path.is_absolute():
                    image_path = (pages_csv_path.parent / image_path).resolve()
                    if not image_path.exists():
                        image_path = Path(row["image_path"]).resolve()
                text_path = Path(row["text_path"])
                if not text_path.is_absolute():
                    text_path = (pages_csv_path.parent / text_path).resolve()
                    if not text_path.exists():
                        text_path = Path(row["text_path"]).resolve()
                reference_text = text_path.read_text(encoding="utf-8").strip()
                page_id = row.get("page_id") or image_path.stem
                items.append(
                    self._evaluate_page(
                        image_path=image_path,
                        page_id=page_id,
                        reference_text=reference_text,
                        processed_dir=processed_dir,
                        mode=mode,
                        preprocess=preprocess,
                    )
                )

        self._write_results(output_csv_path, items)
        return PageEvaluationSummary(
            items=items,
            average_cer=self._average([item.metrics.character_error_rate for item in items]),
            average_wer=self._average([item.metrics.word_error_rate for item in items]),
            average_processing_time_ms=self._average([float(item.processing_time_ms) for item in items]),
        )

    def _evaluate_page(
        self,
        image_path: Path,
        page_id: str,
        reference_text: str,
        processed_dir: Path,
        mode: str,
        preprocess: bool,
    ) -> PageEvaluationItem:
        start = time.perf_counter()
        page_path = image_path
        if preprocess:
            page_path = processed_dir / f"{page_id}_page.png"
            self.preprocessor.preprocess(image_path, page_path)

        line_count = 0
        if mode == PAGE_MODE_QWEN_PAGE:
            recognizer = self._require_qwen()
            result = self._recognize_full_page(recognizer, page_path)
        elif mode == PAGE_MODE_QWEN_LAYOUT:
            recognizer = self._require_qwen()
            result = self._recognize_layout(recognizer, page_path)
        elif mode in {PAGE_MODE_TROCR_LINES, PAGE_MODE_QWEN_LINES, PAGE_MODE_HYBRID}:
            recognizer = self.trocr_recognizer if mode == PAGE_MODE_TROCR_LINES else self._require_qwen()
            line_dir = processed_dir / f"{page_id}_lines"
            annotated_path = processed_dir / f"{page_id}_annotated.png"
            segmentation = self.page_segmenter.segment(page_path, line_dir, annotated_path)
            line_count = len(segmentation.lines)
            if mode == PAGE_MODE_HYBRID:
                qwen_recognizer = self._require_qwen()
                trocr_results = [self.trocr_recognizer.recognize(line.image_path) for line in segmentation.lines]
                qwen_results = [qwen_recognizer.recognize(line.image_path) for line in segmentation.lines]
                line_results = trocr_results + qwen_results
                result = RecognitionResult(
                    text=RecognitionService._format_hybrid(trocr_results, qwen_results),
                    confidence=None,
                    model_name=f"{self.trocr_recognizer.model_name} + {qwen_recognizer.model_name}",
                )
                elapsed_ms = int((time.perf_counter() - start) * 1000)
                return PageEvaluationItem(
                    image_path=image_path,
                    page_id=page_id,
                    reference_text=reference_text,
                    recognized_text=result.text,
                    metrics=calculate_metrics(reference_text, result.text),
                    model_name=result.model_name,
                    mode=mode,
                    line_count=line_count,
                    processing_time_ms=elapsed_ms,
                )
            line_results = [recognizer.recognize(line.image_path) for line in segmentation.lines]
            result = RecognitionResult(
                text="\n".join(line_result.text.strip() for line_result in line_results if line_result.text.strip()),
                confidence=None,
                model_name=line_results[0].model_name if line_results else getattr(recognizer, "model_name", mode),
            )
        else:
            raise ValueError(f"Неподдерживаемый режим оценки страницы: {mode}.")

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return PageEvaluationItem(
            image_path=image_path,
            page_id=page_id,
            reference_text=reference_text,
            recognized_text=result.text,
            metrics=calculate_metrics(reference_text, result.text),
            model_name=result.model_name,
            mode=mode,
            line_count=line_count,
            processing_time_ms=elapsed_ms,
        )

    def _require_qwen(self) -> BaseRecognizer:
        if self.qwen_recognizer is None:
            raise ValueError("Для выбранного режима требуется Qwen-распознаватель.")
        return self.qwen_recognizer

    @staticmethod
    def _recognize_full_page(recognizer: BaseRecognizer, image_path: Path) -> RecognitionResult:
        if hasattr(recognizer, "recognize_page"):
            return recognizer.recognize_page(image_path)
        return recognizer.recognize(image_path)

    @staticmethod
    def _recognize_layout(recognizer: BaseRecognizer, image_path: Path) -> RecognitionResult:
        if hasattr(recognizer, "recognize_layout"):
            return recognizer.recognize_layout(image_path)
        return recognizer.recognize(image_path)

    @staticmethod
    def _write_results(output_csv_path: Path, items: list[PageEvaluationItem]) -> None:
        with output_csv_path.open("w", encoding="utf-8", newline="") as target:
            fieldnames = [
                "image_path",
                "page_id",
                "reference_text",
                "recognized_text",
                "model_name",
                "mode",
                "line_count",
                "processing_time_ms",
                "cer",
                "wer",
            ]
            writer = csv.DictWriter(target, fieldnames=fieldnames)
            writer.writeheader()
            for item in items:
                writer.writerow(
                    {
                        "image_path": str(item.image_path),
                        "page_id": item.page_id,
                        "reference_text": item.reference_text,
                        "recognized_text": item.recognized_text,
                        "model_name": item.model_name,
                        "mode": item.mode,
                        "line_count": item.line_count,
                        "processing_time_ms": item.processing_time_ms,
                        "cer": f"{item.metrics.character_error_rate:.6f}",
                        "wer": f"{item.metrics.word_error_rate:.6f}",
                    }
                )

    @staticmethod
    def _average(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0
