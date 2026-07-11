import argparse
import json
from pathlib import Path
import sys

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen-VL with LoRA on OCR JSONL messages.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--output-dir", type=Path, default=Path("models/qwen-vl-ocr-lora"))
    parser.add_argument(
        "--initial-adapter",
        type=Path,
        help="Optional existing LoRA adapter to continue training instead of starting from a new adapter.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-image-side", type=int, default=512)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Load and validate dataset without loading the model.")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load the processor and base model only from the local Hugging Face cache.",
    )
    return parser.parse_args()


def load_examples(dataset_path: Path, max_samples: int | None = None) -> list[dict]:
    examples: list[dict] = []
    with dataset_path.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            example = json.loads(line)
            validate_example(example)
            examples.append(example)
            if max_samples is not None and len(examples) >= max_samples:
                break
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


class QwenOcrSftDataset:
    def __init__(self, examples: list[dict], processor, max_image_side: int | None = None) -> None:
        self.examples = examples
        self.processor = processor
        self.max_image_side = max_image_side

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        example = self.examples[index]
        image_path = extract_image_path(example)
        image = resize_image(Image.open(image_path).convert("RGB"), self.max_image_side)
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


def main() -> None:
    args = parse_args()
    examples = load_examples(args.dataset, args.max_samples)
    print(f"loaded examples: {len(examples)}")
    if args.dry_run:
        for example in examples[:3]:
            print(f"image: {extract_image_path(example)}")
            print(f"answer: {example['messages'][1]['content'][:120]}")
        return

    try:
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoProcessor, Trainer, TrainingArguments
    except ImportError as exc:
        raise RuntimeError(
            "Qwen LoRA training dependencies are missing. Install requirements-ml.txt first."
        ) from exc

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
    dataset = QwenOcrSftDataset(examples, processor, max_image_side=args.max_image_side)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        remove_unused_columns=False,
        fp16=args.fp16,
        bf16=args.bf16,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
    trainer.train()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"saved LoRA adapter: {args.output_dir}")


if __name__ == "__main__":
    main()
