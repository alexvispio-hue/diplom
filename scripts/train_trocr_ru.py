import argparse
import json
import time
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.training.cyrillic_dataset import CyrillicSample, TrOCRTrainingDataset, load_cyrillic_samples
from app.training.image_augmentation import HandwritingPhotoAugmenter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune TrOCR on Russian handwritten fragments.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/training/cyrillic_handwriting"))
    parser.add_argument("--extra-train-dataset-dir", type=Path, action="append", default=[])
    parser.add_argument("--eval-dataset-dir", type=Path)
    parser.add_argument("--base-model", default="kazars24/trocr-base-handwritten-ru")
    parser.add_argument("--output-dir", type=Path, default=Path("models/trocr-cyrillic-finetuned"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int, default=500)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--skip-save", action="store_true")
    return parser.parse_args()


def load_split(dataset_dir: Path, split: str, limit: int | None = None) -> list[CyrillicSample]:
    return load_cyrillic_samples(dataset_dir / f"{split}.tsv", dataset_dir / split, limit, check_files=False)


def main() -> None:
    import torch
    from torch.utils.data import DataLoader
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel, get_linear_schedule_with_warmup

    args = parse_args()
    processor = TrOCRProcessor.from_pretrained(args.base_model)
    model = VisionEncoderDecoderModel.from_pretrained(args.base_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_samples = load_split(args.dataset_dir, "train", args.max_train_samples)
    for extra_dataset_dir in args.extra_train_dataset_dir:
        train_samples.extend(load_split(extra_dataset_dir, "train", args.max_train_samples))
    eval_dataset_dir = args.eval_dataset_dir or args.dataset_dir
    eval_split = "val" if (eval_dataset_dir / "val.tsv").exists() else "test"
    eval_samples = load_split(eval_dataset_dir, eval_split, args.max_eval_samples)
    train_dataset = TrOCRTrainingDataset(
        train_samples,
        processor,
        transform=HandwritingPhotoAugmenter() if args.augment else None,
    )
    eval_dataset = TrOCRTrainingDataset(eval_samples, processor)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = max(len(train_loader) * args.epochs, 1)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    amp_enabled = device.type == "cuda" and not args.disable_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")
    print(f"mixed precision: {amp_enabled}")
    print(f"train samples: {len(train_dataset)}")
    print(f"eval samples: {len(eval_dataset)}")
    print(f"batch size: {args.batch_size}")
    print(f"training steps: {total_steps}")
    for epoch in range(args.epochs):
        epoch_started_at = time.perf_counter()
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp_enabled):
                outputs = model(
                    pixel_values=batch["pixel_values"].to(device, non_blocking=True),
                    labels=batch["labels"].to(device, non_blocking=True),
                )
            scaler.scale(outputs.loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += outputs.loss.item()
            if step % args.log_every == 0 or step == len(train_loader):
                average_loss = total_loss / step
                print(f"epoch {epoch + 1} step {step}/{len(train_loader)}: train_loss={average_loss:.4f}")

        model.eval()
        eval_loss = 0.0
        with torch.inference_mode():
            for batch in eval_loader:
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp_enabled):
                    outputs = model(
                        pixel_values=batch["pixel_values"].to(device, non_blocking=True),
                        labels=batch["labels"].to(device, non_blocking=True),
                    )
                eval_loss += outputs.loss.item()

        train_loss = total_loss / max(len(train_loader), 1)
        validation_loss = eval_loss / max(len(eval_loader), 1)
        elapsed_minutes = (time.perf_counter() - epoch_started_at) / 60
        print(
            f"epoch {epoch + 1}: train_loss={train_loss:.4f}, "
            f"validation_loss={validation_loss:.4f}, minutes={elapsed_minutes:.2f}"
        )
        if not args.skip_save:
            checkpoint_dir = args.output_dir / f"checkpoint-epoch-{epoch + 1}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(checkpoint_dir)
            processor.save_pretrained(checkpoint_dir)
            metrics = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "elapsed_minutes": elapsed_minutes,
            }
            (checkpoint_dir / "metrics.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"saved checkpoint: {checkpoint_dir}")

    if args.skip_save:
        print("checkpoint saving skipped")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(args.output_dir)
        processor.save_pretrained(args.output_dir)
        print(f"saved model: {args.output_dir}")


if __name__ == "__main__":
    main()
