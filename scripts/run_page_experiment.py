import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.ocr.factory import build_recognizer
from app.services.page_experiment_service import PageExperimentService
from app.services.recognition_service import (
    PAGE_MODE_HYBRID,
    PAGE_MODE_QWEN_LAYOUT,
    PAGE_MODE_QWEN_LINES,
    PAGE_MODE_QWEN_PAGE,
    PAGE_MODE_TROCR_LINES,
)


DEFAULT_MODES = [PAGE_MODE_TROCR_LINES, PAGE_MODE_QWEN_LINES, PAGE_MODE_QWEN_PAGE, PAGE_MODE_HYBRID]
ALL_MODES = [PAGE_MODE_TROCR_LINES, PAGE_MODE_QWEN_LINES, PAGE_MODE_QWEN_PAGE, PAGE_MODE_QWEN_LAYOUT, PAGE_MODE_HYBRID]
QWEN_MODES = {PAGE_MODE_QWEN_LINES, PAGE_MODE_QWEN_PAGE, PAGE_MODE_QWEN_LAYOUT, PAGE_MODE_HYBRID}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a reproducible full-page OCR experiment.")
    parser.add_argument("--pages-csv", type=Path, default=Path("data/training/page_lines/test_pages.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/evaluation/page_experiment"))
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES, choices=ALL_MODES)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-preprocess", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qwen_recognizer = build_recognizer("qwen") if QWEN_MODES.intersection(args.modes) else None
    service = PageExperimentService(
        trocr_recognizer=build_recognizer("trocr"),
        qwen_recognizer=qwen_recognizer,
    )
    results = service.run(
        pages_csv_path=args.pages_csv,
        output_dir=args.output_dir,
        modes=args.modes,
        preprocess=not args.no_preprocess,
        limit=args.limit,
    )

    print(f"dataset: {args.pages_csv}")
    print(f"output: {args.output_dir}")
    for result in results:
        print(
            f"{result.mode}: model={result.model_name}, items={result.items}, CER={result.average_cer:.4f}, "
            f"WER={result.average_wer:.4f}, page_time_ms={result.average_page_time_ms:.1f}, "
            f"line_time_ms={result.average_line_time_ms:.1f}"
        )
    print(f"summary: {args.output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
