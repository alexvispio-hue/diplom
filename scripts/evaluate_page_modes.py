import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.ocr.factory import build_recognizer
from app.services.page_evaluation_service import PageEvaluationService
from app.services.recognition_service import (
    PAGE_MODE_HYBRID,
    PAGE_MODE_QWEN_LAYOUT,
    PAGE_MODE_QWEN_LINES,
    PAGE_MODE_QWEN_PAGE,
    PAGE_MODE_TROCR_LINES,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate full-page OCR modes on a pages CSV manifest.")
    parser.add_argument("--pages-csv", type=Path, default=Path("data/training/page_lines/test_pages.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/evaluation/page_modes/results.csv"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/evaluation/page_modes/processed"))
    parser.add_argument(
        "--mode",
        choices=[PAGE_MODE_TROCR_LINES, PAGE_MODE_QWEN_LINES, PAGE_MODE_QWEN_PAGE, PAGE_MODE_QWEN_LAYOUT, PAGE_MODE_HYBRID],
        default=PAGE_MODE_TROCR_LINES,
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-preprocess", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qwen_recognizer = None
    if args.mode in {PAGE_MODE_QWEN_LINES, PAGE_MODE_QWEN_PAGE, PAGE_MODE_QWEN_LAYOUT, PAGE_MODE_HYBRID}:
        qwen_recognizer = build_recognizer("qwen")

    service = PageEvaluationService(
        trocr_recognizer=build_recognizer("trocr"),
        qwen_recognizer=qwen_recognizer,
    )
    summary = service.evaluate_pages_manifest(
        pages_csv_path=args.pages_csv,
        output_csv_path=args.output,
        processed_dir=args.processed_dir,
        mode=args.mode,
        preprocess=not args.no_preprocess,
        limit=args.limit,
    )

    print(f"items: {len(summary.items)}")
    print(f"mode: {args.mode}")
    print(f"average CER: {summary.average_cer:.4f}")
    print(f"average WER: {summary.average_wer:.4f}")
    print(f"average time ms: {summary.average_processing_time_ms:.1f}")
    print(f"results: {args.output}")


if __name__ == "__main__":
    main()
