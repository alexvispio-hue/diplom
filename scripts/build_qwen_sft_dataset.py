import argparse
import csv
import json
from pathlib import Path


LINE_PROMPT = "Распознай русский рукописный текст на изображении. Верни только текст."
PAGE_PROMPT = "Распознай весь русский рукописный текст. Сохрани порядок строк и переносы строк."
LAYOUT_PROMPT = "Найди строки рукописного текста и верни JSON с текстом и координатами."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Qwen-VL SFT JSONL dataset from OCR CSV manifests.")
    parser.add_argument("--lines-csv", type=Path, help="CSV with image_path,text and optional bbox columns.")
    parser.add_argument("--pages-csv", type=Path, help="CSV with image_path,text_path,page_id columns.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--include-layout",
        action="store_true",
        help="Add line-detection JSON examples when x1,y1,x2,y2 columns are present.",
    )
    return parser.parse_args()


def build_message(image_path: str, prompt: str, answer: str) -> dict:
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            },
            {"role": "assistant", "content": answer},
        ]
    }


def iter_line_examples(lines_csv: Path, include_layout: bool) -> list[dict]:
    examples: list[dict] = []
    with lines_csv.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        for row in reader:
            image_path = row["image_path"]
            text = row["text"]
            examples.append(build_message(image_path, LINE_PROMPT, text))
            if include_layout and all(row.get(column) for column in ("x1", "y1", "x2", "y2")):
                layout = [
                    {
                        "line_number": int(row.get("line_number") or 1),
                        "text": text,
                        "box": [int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])],
                    }
                ]
                examples.append(build_message(image_path, LAYOUT_PROMPT, json.dumps(layout, ensure_ascii=False)))
    return examples


def iter_page_examples(pages_csv: Path) -> list[dict]:
    examples: list[dict] = []
    with pages_csv.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        for row in reader:
            text_path = Path(row["text_path"])
            answer = text_path.read_text(encoding="utf-8").strip()
            examples.append(build_message(row["image_path"], PAGE_PROMPT, answer))
    return examples


def main() -> None:
    args = parse_args()
    examples: list[dict] = []
    if args.lines_csv:
        examples.extend(iter_line_examples(args.lines_csv, args.include_layout))
    if args.pages_csv:
        examples.extend(iter_page_examples(args.pages_csv))
    if not examples:
        raise SystemExit("No input examples were provided.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as target:
        for example in examples:
            target.write(json.dumps(example, ensure_ascii=False) + "\n")
    print(f"saved examples: {len(examples)}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
