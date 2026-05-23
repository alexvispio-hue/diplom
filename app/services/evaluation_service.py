import csv
import time
from dataclasses import dataclass
from pathlib import Path

from app.image_processing.preprocessing import ImagePreprocessor
from app.ocr.base import BaseRecognizer
from app.ocr.metrics import TextRecognitionMetrics, calculate_metrics


@dataclass(frozen=True)
class EvaluationItem:
    image_path: Path
    reference_text: str
    recognized_text: str
    metrics: TextRecognitionMetrics
    model_name: str
    processing_time_ms: int


@dataclass(frozen=True)
class EvaluationSummary:
    items: list[EvaluationItem]
    average_cer: float
    average_wer: float


class EvaluationService:
    def __init__(self, recognizer: BaseRecognizer, preprocessor: ImagePreprocessor | None = None) -> None:
        self.recognizer = recognizer
        self.preprocessor = preprocessor or ImagePreprocessor()

    def evaluate_manifest(
        self,
        manifest_path: Path,
        output_csv_path: Path,
        processed_dir: Path,
    ) -> EvaluationSummary:
        items: list[EvaluationItem] = []
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

        with manifest_path.open("r", encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            for row in reader:
                image_path = (manifest_path.parent / row["image_path"]).resolve()
                reference_text = row["reference_text"]
                processed_path = processed_dir / f"{image_path.stem}.png"

                start = time.perf_counter()
                self.preprocessor.preprocess(image_path, processed_path)
                result = self.recognizer.recognize(processed_path)
                elapsed_ms = int((time.perf_counter() - start) * 1000)

                items.append(
                    EvaluationItem(
                        image_path=image_path,
                        reference_text=reference_text,
                        recognized_text=result.text,
                        metrics=calculate_metrics(reference_text, result.text),
                        model_name=result.model_name,
                        processing_time_ms=elapsed_ms,
                    )
                )

        self._write_results(output_csv_path, items)
        return EvaluationSummary(
            items=items,
            average_cer=self._average([item.metrics.character_error_rate for item in items]),
            average_wer=self._average([item.metrics.word_error_rate for item in items]),
        )

    @staticmethod
    def _write_results(output_csv_path: Path, items: list[EvaluationItem]) -> None:
        with output_csv_path.open("w", encoding="utf-8", newline="") as target:
            fieldnames = [
                "image_path",
                "reference_text",
                "recognized_text",
                "model_name",
                "processing_time_ms",
                "cer",
                "wer",
                "character_distance",
                "word_distance",
            ]
            writer = csv.DictWriter(target, fieldnames=fieldnames)
            writer.writeheader()
            for item in items:
                writer.writerow(
                    {
                        "image_path": str(item.image_path),
                        "reference_text": item.reference_text,
                        "recognized_text": item.recognized_text,
                        "model_name": item.model_name,
                        "processing_time_ms": item.processing_time_ms,
                        "cer": f"{item.metrics.character_error_rate:.6f}",
                        "wer": f"{item.metrics.word_error_rate:.6f}",
                        "character_distance": item.metrics.character_distance,
                        "word_distance": item.metrics.word_distance,
                    }
                )

    @staticmethod
    def _average(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0
