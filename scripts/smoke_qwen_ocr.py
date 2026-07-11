import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.ocr.qwen_vl_recognizer import QwenVLRecognizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a smoke OCR request through Qwen-VL.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", help="Override QWEN_MODEL_NAME for this run.")
    parser.add_argument("--page", action="store_true", help="Use full-page OCR prompt.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image.exists():
        raise SystemExit(f"Image not found: {args.image}")

    recognizer = QwenVLRecognizer(model_name=args.model)
    result = recognizer.recognize_page(args.image) if args.page else recognizer.recognize(args.image)
    print(f"model: {result.model_name}")
    print("recognized text:")
    print(result.text)


if __name__ == "__main__":
    main()
