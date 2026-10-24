import csv
from pathlib import Path

from PIL import Image

from app.image_processing.page_segmentation import PageSegmentationResult, TextLine
from app.services.segmentation_evaluation_service import SegmentationEvaluationService, box_iou


class StubSegmenter:
    def segment(self, source_path: Path, output_dir: Path, annotated_image_path: Path) -> PageSegmentationResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        annotated_image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (200, 100), "white").save(output_dir / "line_000.png")
        Image.new("RGB", (200, 100), "white").save(output_dir / "line_001.png")
        Image.new("RGB", (200, 100), "white").save(annotated_image_path)
        return PageSegmentationResult(
            annotated_image_path=annotated_image_path,
            lines=[
                TextLine(index=0, image_path=output_dir / "line_000.png", bbox=(10, 10, 110, 30)),
                TextLine(index=1, image_path=output_dir / "line_001.png", bbox=(12, 50, 112, 70)),
            ],
        )


def test_box_iou() -> None:
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert round(box_iou((0, 0, 10, 10), (5, 0, 15, 10)), 2) == 0.33


def test_segmentation_evaluation_writes_metrics(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (200, 100), "white").save(image_path)
    lines_csv = tmp_path / "test_lines.csv"
    lines_csv.write_text(
        "image_path,text,page_id,line_number,x1,y1,x2,y2,source_image\n"
        f"line1.png,text,page_001,1,10,10,110,30,{image_path}\n"
        f"line2.png,text,page_001,2,12,50,112,70,{image_path}\n"
        f"line3.png,text,page_001,3,20,80,120,95,{image_path}\n",
        encoding="utf-8",
    )
    output_csv = tmp_path / "results.csv"

    summary = SegmentationEvaluationService(page_segmenter=StubSegmenter()).evaluate_lines_manifest(
        lines_csv_path=lines_csv,
        output_csv_path=output_csv,
        processed_dir=tmp_path / "processed",
    )

    assert summary.total_reference_lines == 3
    assert summary.total_predicted_lines == 2
    assert summary.total_matched_lines == 2
    assert summary.average_found_line_rate == 2 / 3
    with output_csv.open("r", encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    assert rows[0]["missed_lines"] == "1"
    assert rows[0]["extra_lines"] == "0"
    assert rows[0]["found_line_rate"] == "0.666667"
