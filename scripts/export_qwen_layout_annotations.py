import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.layout_annotation_service import LayoutAnnotationExporter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert qwen_layout CSV output into a draft lines.csv manifest.")
    parser.add_argument("--layout-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=Path("data/training/qwen_layout_draft/lines.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = LayoutAnnotationExporter().export(args.layout_csv, args.output_csv)
    print(f"saved lines: {rows}")
    print(f"output: {args.output_csv}")


if __name__ == "__main__":
    main()
