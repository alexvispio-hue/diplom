import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.ocr.factory import build_recognizer
from app.services.evaluation_service import EvaluationService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate OCR quality on a CSV manifest.")
    parser.add_argument("--manifest", type=Path, default=Path("data/evaluation/manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/evaluation/results.csv"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/evaluation/processed"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = EvaluationService(build_recognizer())
    summary = service.evaluate_manifest(
        manifest_path=args.manifest,
        output_csv_path=args.output,
        processed_dir=args.processed_dir,
    )

    print(f"items: {len(summary.items)}")
    print(f"average CER: {summary.average_cer:.4f}")
    print(f"average WER: {summary.average_wer:.4f}")
    print(f"results: {args.output}")


if __name__ == "__main__":
    main()
