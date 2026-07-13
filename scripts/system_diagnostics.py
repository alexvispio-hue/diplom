import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.diagnostics_service import DiagnosticsService


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect local Handwriting OCR diagnostics without document text.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    args = parser.parse_args()
    payload = DiagnosticsService().collect()
    content = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
        print(args.output)
    else:
        print(content)


if __name__ == "__main__":
    main()
