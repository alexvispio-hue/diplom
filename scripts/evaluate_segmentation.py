import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.segmentation_evaluation_service import SegmentationEvaluationService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate OpenCV page line segmentation against a lines CSV manifest.")
    parser.add_argument("--lines-csv", type=Path, default=Path("data/training/page_lines/test_lines.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/evaluation/segmentation/results.csv"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/evaluation/segmentation/processed"))
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--limit-pages", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = SegmentationEvaluationService().evaluate_lines_manifest(
        lines_csv_path=args.lines_csv,
        output_csv_path=args.output,
        processed_dir=args.processed_dir,
        iou_threshold=args.iou_threshold,
        limit_pages=args.limit_pages,
    )
    print(f"pages: {len(summary.items)}")
    print(f"reference lines: {summary.total_reference_lines}")
    print(f"predicted lines: {summary.total_predicted_lines}")
    print(f"matched lines: {summary.total_matched_lines}")
    print(f"found line rate: {summary.average_found_line_rate:.4f}")
    print(f"average IoU: {summary.average_iou:.4f}")
    print(f"average coordinate deviation: {summary.average_coordinate_deviation:.2f}")
    print(f"results: {args.output}")


if __name__ == "__main__":
    main()
