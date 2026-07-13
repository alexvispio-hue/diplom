from __future__ import annotations

import argparse
import inspect
import json
import math
import random
import re
from pathlib import Path
import sys
import time
from typing import Any

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.training.run_manifest import (
    atomic_write_json,
    create_run_manifest,
    update_run_manifest,
)
from app.services.benchmark_service import summarize_predictions


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen-VL with LoRA on OCR JSONL messages.")
    parser.add_argument("--dataset", "--train-dataset", dest="dataset", type=Path, required=True)
    parser.add_argument("--eval-dataset", type=Path)
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--review-metadata", type=Path)
    parser.add_argument(
        "--review-policy",
        choices=["all", "trusted_only", "review_only"],
        default="all",
    )
    parser.add_argument(
        "--variant-policy",
        choices=["all", "one_per_group"],
        default="all",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--output-dir", type=Path, default=Path("models/qwen-vl-ocr-lora"))
    parser.add_argument("--run-name", default="qwen-vl-ocr-lora")
    parser.add_argument("--run-manifest", type=Path)
    parser.add_argument(
        "--initial-adapter",
        type=Path,
        help="Optional existing LoRA adapter to continue training instead of starting from a new adapter.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--skip-generation-eval", action="store_true")
    parser.add_argument("--max-image-side", type=int, default=512)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument(
        "--resume-from-checkpoint",
        nargs="?",
        const="latest",
        help="Resume from the latest checkpoint or from an explicit checkpoint path.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate datasets and write a run manifest without loading the model.")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load the processor and base model only from the local Hugging Face cache.",
    )
    args = parser.parse_args(argv)
    validate_arguments(args)
    return args


def validate_arguments(args: argparse.Namespace) -> None:
    if args.fp16 and args.bf16:
        raise ValueError("Choose only one mixed-precision mode: --fp16 or --bf16.")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.batch_size <= 0 or args.gradient_accumulation_steps <= 0:
        raise ValueError("Batch size and gradient accumulation must be positive.")
    if args.eval_dataset and args.eval_steps <= 0:
        raise ValueError("--eval-steps must be positive when validation is enabled.")
    if args.early_stopping_patience < 0:
        raise ValueError("--early-stopping-patience cannot be negative.")
    if args.review_policy != "all" and args.review_metadata is None:
        raise ValueError("--review-metadata is required when --review-policy filters examples.")


def load_examples(dataset_path: Path, max_samples: int | None = None) -> list[dict]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset was not found: {dataset_path}")
    examples: list[dict] = []
    with dataset_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                example = json.loads(line)
                validate_example(example)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"Invalid example at {dataset_path}:{line_number}: {exc}") from exc
            examples.append(example)
            if max_samples is not None and len(examples) >= max_samples:
                break
    if not examples:
        raise ValueError(f"Dataset does not contain examples: {dataset_path}")
    return examples


def validate_example(example: dict) -> None:
    messages = example.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        raise ValueError("Each example must contain two messages: user and assistant.")
    user, assistant = messages
    if user.get("role") != "user" or assistant.get("role") != "assistant":
        raise ValueError("Expected user and assistant messages.")
    content = user.get("content")
    if not isinstance(content, list):
        raise ValueError("User content must be a list with image and text parts.")
    if not any(part.get("type") == "image" and part.get("image") for part in content):
        raise ValueError("User message must contain an image part.")
    if not assistant.get("content"):
        raise ValueError("Assistant content must contain target text.")


def extract_image_path(example: dict) -> Path:
    for part in example["messages"][0]["content"]:
        if part.get("type") == "image":
            return Path(part["image"])
    raise ValueError("Image part is missing.")


def filter_examples_by_review(
    examples: list[dict],
    metadata_path: Path | None,
    review_policy: str,
) -> list[dict]:
    if review_policy == "all":
        return examples
    if metadata_path is None or not metadata_path.is_file():
        raise FileNotFoundError(f"Review metadata was not found: {metadata_path}")
    flags: dict[str, bool] = {}
    with metadata_path.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            item = json.loads(line)
            flags[Path(item["image_path"]).as_posix()] = bool(item.get("needs_review"))
    filtered = []
    for example in examples:
        image_path = extract_image_path(example).as_posix()
        needs_review = flags.get(image_path)
        if needs_review is None:
            raise ValueError(f"Image is missing from review metadata: {image_path}")
        if review_policy == "trusted_only" and not needs_review:
            filtered.append(example)
        elif review_policy == "review_only" and needs_review:
            filtered.append(example)
    if not filtered:
        raise ValueError(f"Review policy removed every training example: {review_policy}")
    return filtered


def infer_page_group(example: dict) -> str:
    image_path = extract_image_path(example)
    stem = re.sub(r"_(dark|light|scan)$", "", image_path.stem, flags=re.IGNORECASE)
    return f"{image_path.parent.as_posix()}/{stem}"


class GroupAwareSampler:
    def __init__(self, group_ids: list[str], *, seed: int, variant_policy: str) -> None:
        self.seed = seed
        self.variant_policy = variant_policy
        self.epoch = 0
        self.groups: dict[str, list[int]] = {}
        for index, group_id in enumerate(group_ids):
            self.groups.setdefault(group_id, []).append(index)

    def __iter__(self):
        generator = random.Random(self.seed + self.epoch)
        self.epoch += 1
        group_names = list(self.groups)
        generator.shuffle(group_names)
        if self.variant_policy == "one_per_group":
            indices = [generator.choice(self.groups[group]) for group in group_names]
            return iter(indices)

        queues = {group: list(indices) for group, indices in self.groups.items()}
        for indices in queues.values():
            generator.shuffle(indices)
        ordered: list[int] = []
        previous_group = None
        while group_names:
            eligible = [group for group in group_names if group != previous_group]
            choices = eligible or group_names
            largest = max(len(queues[group]) for group in choices)
            tied = [group for group in choices if len(queues[group]) == largest]
            group = generator.choice(tied)
            ordered.append(queues[group].pop())
            previous_group = group
            if not queues[group]:
                group_names.remove(group)
        return iter(ordered)

    def __len__(self) -> int:
        if self.variant_policy == "one_per_group":
            return len(self.groups)
        return sum(len(indices) for indices in self.groups.values())


class QwenOcrSftDataset:
    def __init__(self, examples: list[dict], processor, max_image_side: int | None = None) -> None:
        self.examples = examples
        self.processor = processor
        self.max_image_side = max_image_side
        self.group_ids = [infer_page_group(example) for example in examples]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        example = self.examples[index]
        image_path = extract_image_path(example)
        try:
            with Image.open(image_path) as source_image:
                image = resize_image(source_image.convert("RGB"), self.max_image_side)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Cannot load training image: {image_path}") from exc
        prompt_text = self.processor.apply_chat_template(
            [example["messages"][0]],
            tokenize=False,
            add_generation_prompt=True,
        )
        answer_text = example["messages"][1]["content"]
        eos_token = getattr(self.processor.tokenizer, "eos_token", None) or ""
        full_text = prompt_text + answer_text + eos_token
        inputs = self.processor(text=[full_text], images=[image], return_tensors="pt")
        labels = inputs["input_ids"].clone()
        answer_ids = self.processor.tokenizer(
            answer_text + eos_token,
            add_special_tokens=False,
        )["input_ids"]
        prompt_length = max(0, labels.shape[1] - len(answer_ids))
        labels[:, :prompt_length] = -100
        pad_token_id = getattr(self.processor.tokenizer, "pad_token_id", None)
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100
        return {
            key: value.squeeze(0) if hasattr(value, "squeeze") else value
            for key, value in {**inputs, "labels": labels}.items()
        }


def resize_image(image: Image.Image, max_image_side: int | None) -> Image.Image:
    if not max_image_side or max(image.size) <= max_image_side:
        return image
    resized = image.copy()
    resized.thumbnail((max_image_side, max_image_side), Image.Resampling.LANCZOS)
    return resized


def serializable_arguments(args: argparse.Namespace) -> dict[str, Any]:
    result = {}
    for key, value in vars(args).items():
        result[key] = str(value) if isinstance(value, Path) else value
    return result


def resolve_resume_value(value: str | None) -> bool | str | None:
    if value is None:
        return None
    if value == "latest":
        return True
    checkpoint = Path(value)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint was not found: {checkpoint}")
    return str(checkpoint)


def build_training_arguments(TrainingArguments, args: argparse.Namespace) -> Any:
    has_eval = args.eval_dataset is not None
    kwargs: dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "save_strategy": "steps",
        "remove_unused_columns": False,
        "fp16": args.fp16,
        "bf16": args.bf16,
        "seed": args.seed,
        "data_seed": args.seed,
        "dataloader_num_workers": args.dataloader_num_workers,
        "report_to": [],
        "logging_nan_inf_filter": True,
        "skip_memory_metrics": False,
    }
    if has_eval:
        signature = inspect.signature(TrainingArguments.__init__)
        strategy_key = "eval_strategy" if "eval_strategy" in signature.parameters else "evaluation_strategy"
        kwargs.update(
            {
                strategy_key: "steps",
                "eval_steps": args.eval_steps,
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,
            }
        )
    return TrainingArguments(**kwargs)


def build_group_aware_trainer_class(Trainer):
    class GroupAwareTrainer(Trainer):
        def _get_train_sampler(self, train_dataset=None):
            dataset = train_dataset or self.train_dataset
            if isinstance(dataset, QwenOcrSftDataset):
                return GroupAwareSampler(
                    dataset.group_ids,
                    seed=self.args.seed,
                    variant_policy=self.variant_policy,
                )
            return super()._get_train_sampler(train_dataset)

    return GroupAwareTrainer


def build_health_callback(TrainerCallback):
    class TrainingHealthCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            del args, state, kwargs
            for key, value in (logs or {}).items():
                if "loss" in key and isinstance(value, (float, int)) and not math.isfinite(float(value)):
                    raise FloatingPointError(f"Training produced a non-finite metric: {key}={value}")
            return control

    return TrainingHealthCallback()


def generation_evaluate(
    model,
    processor,
    examples: list[dict],
    *,
    max_image_side: int,
    max_new_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch

    predictions = []
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for example in examples:
        image_path = extract_image_path(example)
        with Image.open(image_path) as source_image:
            image = resize_image(source_image.convert("RGB"), max_image_side)
        prompt = processor.apply_chat_template(
            [example["messages"][0]],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(text=[prompt], images=[image], return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        latency_ms = (time.perf_counter() - started) * 1000
        prompt_tokens = int(inputs["input_ids"].shape[1])
        generated_tokens = generated[:, prompt_tokens:]
        prediction = processor.batch_decode(generated_tokens, skip_special_tokens=True)[0].strip()
        predictions.append(
            {
                "image_path": str(image_path),
                "reference_text": str(example["messages"][1]["content"]),
                "prediction_text": prediction,
                "latency_ms": round(latency_ms, 3),
                "truncated": int(generated_tokens.shape[1]) >= max_new_tokens,
                "peak_vram_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
            }
        )
    return predictions, summarize_predictions(predictions).to_dict()


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _manifest_path(args: argparse.Namespace) -> Path:
    return args.run_manifest or args.output_dir / "run_manifest.json"


def prepare_manifest(args: argparse.Namespace) -> Path:
    manifest_path = _manifest_path(args)
    create_run_manifest(
        manifest_path,
        project_root=PROJECT_ROOT,
        run_name=args.run_name,
        base_model=args.model,
        train_dataset=args.dataset,
        eval_dataset=args.eval_dataset,
        arguments=serializable_arguments(args),
        initial_adapter=args.initial_adapter,
        dataset_manifest=args.dataset_manifest,
    )
    return manifest_path


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    train_examples = load_examples(args.dataset, args.max_samples)
    train_examples = filter_examples_by_review(
        train_examples,
        args.review_metadata,
        args.review_policy,
    )
    eval_examples = (
        load_examples(args.eval_dataset, args.max_eval_samples)
        if args.eval_dataset
        else []
    )
    print(f"loaded train examples: {len(train_examples)}")
    print(f"loaded eval examples: {len(eval_examples)}")

    manifest_path = prepare_manifest(args)
    update_run_manifest(
        manifest_path,
        "validated",
        metrics={"train_examples": len(train_examples), "eval_examples": len(eval_examples)},
    )
    if args.dry_run:
        for example in train_examples[:3]:
            print(f"image: {extract_image_path(example)}")
            print(f"answer: {example['messages'][1]['content'][:120]}")
        print(f"run manifest: {manifest_path}")
        return

    try:
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import (
            AutoProcessor,
            EarlyStoppingCallback,
            Trainer,
            TrainerCallback,
            TrainingArguments,
        )
    except ImportError as exc:
        update_run_manifest(manifest_path, "failed", message="Training dependencies are missing.")
        raise RuntimeError(
            "Qwen LoRA training dependencies are missing. Install requirements-ml.txt first."
        ) from exc

    try:
        model_class = None
        for class_name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq"):
            try:
                module = __import__("transformers", fromlist=[class_name])
                model_class = getattr(module, class_name)
                break
            except (ImportError, AttributeError):
                continue
        if model_class is None:
            raise RuntimeError("Installed transformers version does not expose a Qwen-VL compatible model class.")

        processor = AutoProcessor.from_pretrained(
            args.model, trust_remote_code=True, local_files_only=args.local_files_only
        )
        model = model_class.from_pretrained(
            args.model,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype="auto",
            local_files_only=args.local_files_only,
        )
        if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            if hasattr(model, "config"):
                model.config.use_cache = False
        if args.initial_adapter:
            if not args.initial_adapter.is_dir():
                raise FileNotFoundError(f"Initial LoRA adapter was not found: {args.initial_adapter}")
            model = PeftModel.from_pretrained(model, str(args.initial_adapter), is_trainable=True)
            print(f"continuing from LoRA adapter: {args.initial_adapter}")
        else:
            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)

        train_dataset = QwenOcrSftDataset(train_examples, processor, max_image_side=args.max_image_side)
        eval_dataset = (
            QwenOcrSftDataset(eval_examples, processor, max_image_side=args.max_image_side)
            if eval_examples
            else None
        )
        training_args = build_training_arguments(TrainingArguments, args)
        callbacks = []
        if eval_dataset is not None and args.early_stopping_patience > 0:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))
        callbacks.append(build_health_callback(TrainerCallback))
        GroupAwareTrainer = build_group_aware_trainer_class(Trainer)
        trainer = GroupAwareTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=callbacks,
        )
        trainer.variant_policy = args.variant_policy

        update_run_manifest(manifest_path, "running")
        train_result = trainer.train(resume_from_checkpoint=resolve_resume_value(args.resume_from_checkpoint))
        trainer.save_state()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(args.output_dir)
        processor.save_pretrained(args.output_dir)
        train_metrics = dict(train_result.metrics)
        try:
            import torch

            train_metrics["peak_vram_bytes"] = (
                int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
            )
        except ImportError:
            train_metrics["peak_vram_bytes"] = None
        atomic_write_json(args.output_dir / "train_metrics.json", train_metrics)
        update_run_manifest(
            manifest_path,
            "training_completed",
            metrics={"train": train_metrics},
            artifacts={"adapter_dir": str(args.output_dir)},
        )

        if eval_dataset is not None:
            eval_metrics = dict(trainer.evaluate())
            atomic_write_json(args.output_dir / "eval_metrics.json", eval_metrics)
            generation_summary = None
            if not args.skip_generation_eval:
                predictions, generation_summary = generation_evaluate(
                    model,
                    processor,
                    eval_examples,
                    max_image_side=args.max_image_side,
                    max_new_tokens=args.max_new_tokens,
                )
                write_jsonl(args.output_dir / "eval_predictions.jsonl", predictions)
                atomic_write_json(args.output_dir / "eval_generation_summary.json", generation_summary)
            if generation_summary is None:
                update_run_manifest(
                    manifest_path,
                    "training_completed",
                    metrics={"validation_loss_eval": eval_metrics},
                    artifacts={"eval_metrics": str(args.output_dir / "eval_metrics.json")},
                )
            else:
                update_run_manifest(
                    manifest_path,
                    "evaluation_completed",
                    metrics={"eval": eval_metrics, "generation_eval": generation_summary},
                    artifacts={
                        "eval_metrics": str(args.output_dir / "eval_metrics.json"),
                        "eval_predictions": str(args.output_dir / "eval_predictions.jsonl"),
                    },
                )
        print(f"saved LoRA adapter: {args.output_dir}")
        print(f"run manifest: {manifest_path}")
    except KeyboardInterrupt:
        update_run_manifest(manifest_path, "interrupted", message="Training was interrupted by the user.")
        raise
    except Exception as exc:
        message = str(exc)
        lowered = message.casefold()
        failure_type = "oom" if "out of memory" in lowered else "invalid_metric" if "non-finite" in lowered else "runtime"
        update_run_manifest(
            manifest_path,
            "failed",
            metrics={"failure_type": failure_type},
            message=message,
            force=True,
        )
        raise


if __name__ == "__main__":
    main()
