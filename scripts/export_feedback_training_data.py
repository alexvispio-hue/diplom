import argparse
import csv
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.build_qwen_sft_dataset import LINE_PROMPT, PAGE_PROMPT, build_message


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export reviewed feedback samples into OCR training manifests.")
    parser.add_argument("--feedback-dir", type=Path, default=Path("data/training/user_feedback/problems"))
    parser.add_argument("--output-csv", type=Path, default=Path("data/training/user_feedback/reviewed_examples.csv"))
    parser.add_argument("--output-qwen-jsonl", type=Path, help="Optional Qwen SFT JSONL built from corrected_text.")
    parser.add_argument(
        "--only-reviewed",
        action="store_true",
        help="Skip samples without corrected_text. Recommended before adding data to training.",
    )
    parser.add_argument(
        "--include-unapproved",
        action="store_true",
        help="Include reviewed samples without explicit training consent.",
    )
    return parser.parse_args()


def iter_feedback_examples(
    feedback_dir: Path,
    only_reviewed: bool = True,
    *,
    only_approved: bool = True,
) -> list[dict]:
    examples: list[dict] = []
    for metadata_path in sorted(feedback_dir.glob("recognition_*.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        if only_approved and (
            not metadata.get("training_approved") or metadata.get("training_excluded")
        ):
            continue
        corrected_text = (metadata.get("corrected_text") or "").strip()
        if only_reviewed and not corrected_text:
            continue
        image_filename = metadata.get("image_filename")
        if not image_filename:
            continue
        image_path = feedback_dir / image_filename
        examples.append(
            {
                "image_path": image_path.as_posix(),
                "text": corrected_text,
                "recognized_text": metadata.get("recognized_text") or "",
                "recognition_id": metadata.get("recognition_id"),
                "recognition_mode": metadata.get("recognition_mode") or "",
                "model_name": metadata.get("model_name") or "",
                "rating": metadata.get("rating") or "",
                "training_approved": bool(metadata.get("training_approved")),
            }
        )
    return examples


def write_csv(examples: list[dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_path",
        "text",
        "recognized_text",
        "recognition_id",
        "recognition_mode",
        "model_name",
        "rating",
        "training_approved",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(examples)


def write_qwen_jsonl(examples: list[dict], output_jsonl: Path) -> None:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8", newline="\n") as target:
        for example in examples:
            if not example["text"].strip():
                continue
            prompt = PAGE_PROMPT if example["recognition_mode"] in {"qwen_page", "qwen_layout"} else LINE_PROMPT
            target.write(
                json.dumps(
                    build_message(example["image_path"], prompt, example["text"]),
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    args = parse_args()
    examples = iter_feedback_examples(
        args.feedback_dir,
        only_reviewed=args.only_reviewed,
        only_approved=not args.include_unapproved,
    )
    write_csv(examples, args.output_csv)
    print(f"saved examples: {len(examples)}")
    print(f"output: {args.output_csv}")
    if args.output_qwen_jsonl:
        write_qwen_jsonl(examples, args.output_qwen_jsonl)
        print(f"qwen jsonl: {args.output_qwen_jsonl}")


if __name__ == "__main__":
    main()
