from pathlib import Path

from PIL import Image

from app.ocr.base import BaseRecognizer, RecognitionResult
from app.services.evaluation_service import EvaluationService


class StubRecognizer(BaseRecognizer):
    def recognize(self, image_path: Path) -> RecognitionResult:
        return RecognitionResult(text="hello word", confidence=None, model_name="stub")


def test_evaluation_service_writes_results(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (120, 40), "white").save(image_path)
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text("image_path,reference_text\nsample.png,hello world\n", encoding="utf-8")
    output_path = tmp_path / "results.csv"

    summary = EvaluationService(StubRecognizer()).evaluate_manifest(
        manifest_path=manifest_path,
        output_csv_path=output_path,
        processed_dir=tmp_path / "processed",
    )

    assert summary.average_cer > 0
    assert summary.average_wer == 0.5
    assert output_path.exists()
    assert "hello word" in output_path.read_text(encoding="utf-8")
