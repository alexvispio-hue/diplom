import argparse
import csv
import time
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.ocr.metrics import calculate_metrics
from app.ocr.trocr_recognizer import TrOCRRecognizer
from app.image_processing.preprocessing import ImagePreprocessor
from app.training.cyrillic_dataset import load_cyrillic_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a TrOCR model on Cyrillic Handwriting Dataset.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/training/cyrillic_handwriting"))
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--model", default="kazars24/trocr-base-handwritten-ru")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("data/evaluation/cyrillic_baseline.csv"))
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/evaluation/processed_cyrillic"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels_path = args.dataset_dir / f"{args.split}.tsv"
    images_dir = args.dataset_dir / args.split
    samples = load_cyrillic_samples(labels_path, images_dir, args.limit)
    recognizer = TrOCRRecognizer(model_name=args.model)
    preprocessor = ImagePreprocessor()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str | int | float]] = []
    for sample in samples:
        recognition_path = sample.image_path
        if args.preprocess:
            recognition_path = args.processed_dir / f"{args.split}_{sample.image_path.name}"
            preprocessor.preprocess(sample.image_path, recognition_path)

        start = time.perf_counter()
        result = recognizer.recognize(recognition_path)
        processing_time_ms = int((time.perf_counter() - start) * 1000)
        metrics = calculate_metrics(sample.text, result.text)
        rows.append(
            {
                "image_path": str(sample.image_path),
                "reference_text": sample.text,
                "recognized_text": result.text,
                "preprocessed": args.preprocess,
                "cer": metrics.character_error_rate,
                "wer": metrics.word_error_rate,
                "processing_time_ms": processing_time_ms,
            }
        )

    with args.output.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    average_cer = sum(float(row["cer"]) for row in rows) / len(rows) if rows else 0.0
    average_wer = sum(float(row["wer"]) for row in rows) / len(rows) if rows else 0.0
    print(f"samples: {len(rows)}")
    print(f"model: {args.model}")
    print(f"preprocessed: {args.preprocess}")
    print(f"average CER: {average_cer:.4f}")
    print(f"average WER: {average_wer:.4f}")
    print(f"results: {args.output}")


if __name__ == "__main__":
    main()
