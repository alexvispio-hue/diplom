import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.ocr.trocr_recognizer import TrOCRRecognizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare TrOCR models on local images.")
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("images", type=Path, nargs="+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recognizers = [TrOCRRecognizer(model_name=model) for model in args.model]
    print("image\t" + "\t".join(recognizer.model_name for recognizer in recognizers))
    for image_path in args.images:
        results = [recognizer.recognize(image_path).text for recognizer in recognizers]
        print(f"{image_path.name}\t" + "\t".join(results))


if __name__ == "__main__":
    main()
