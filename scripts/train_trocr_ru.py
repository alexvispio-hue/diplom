import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.training.cyrillic_dataset import TrOCRTrainingDataset, load_cyrillic_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune TrOCR on Russian handwritten fragments.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/training/cyrillic_handwriting"))
    parser.add_argument("--base-model", default="kazars24/trocr-base-handwritten-ru")
    parser.add_argument("--output-dir", type=Path, default=Path("models/trocr-cyrillic-finetuned"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int, default=500)
    parser.add_argument("--skip-save", action="store_true")
    return parser.parse_args()


def main() -> None:
    import torch
    from torch.utils.data import DataLoader
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    args = parse_args()
    processor = TrOCRProcessor.from_pretrained(args.base_model)
    model = VisionEncoderDecoderModel.from_pretrained(args.base_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_samples = load_cyrillic_samples(
        args.dataset_dir / "train.tsv",
        args.dataset_dir / "train",
        args.max_train_samples,
    )
    eval_samples = load_cyrillic_samples(
        args.dataset_dir / "test.tsv",
        args.dataset_dir / "test",
        args.max_eval_samples,
    )
    train_dataset = TrOCRTrainingDataset(train_samples, processor)
    eval_dataset = TrOCRTrainingDataset(eval_samples, processor)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    print(f"device: {device}")
    print(f"train samples: {len(train_dataset)}")
    print(f"eval samples: {len(eval_dataset)}")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(
                pixel_values=batch["pixel_values"].to(device),
                labels=batch["labels"].to(device),
            )
            outputs.loss.backward()
            optimizer.step()
            total_loss += outputs.loss.item()

        model.eval()
        eval_loss = 0.0
        with torch.inference_mode():
            for batch in eval_loader:
                outputs = model(
                    pixel_values=batch["pixel_values"].to(device),
                    labels=batch["labels"].to(device),
                )
                eval_loss += outputs.loss.item()

        train_loss = total_loss / max(len(train_loader), 1)
        validation_loss = eval_loss / max(len(eval_loader), 1)
        print(f"epoch {epoch + 1}: train_loss={train_loss:.4f}, validation_loss={validation_loss:.4f}")

    if args.skip_save:
        print("checkpoint saving skipped")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(args.output_dir)
        processor.save_pretrained(args.output_dir)
        print(f"saved model: {args.output_dir}")


if __name__ == "__main__":
    main()
