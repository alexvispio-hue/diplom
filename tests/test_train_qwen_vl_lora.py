import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.train_qwen_vl_lora import (
    GroupAwareSampler,
    build_training_arguments,
    extract_image_path,
    infer_page_group,
    load_examples,
    parse_args,
    resize_image,
    resolve_resume_value,
    validate_example,
)


def build_example(image_path: str = "line.jpg", answer: str = "текст") -> dict:
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": "Распознай текст."},
                ],
            },
            {"role": "assistant", "content": answer},
        ]
    }


def test_load_examples_validates_qwen_sft_jsonl(tmp_path: Path) -> None:
    dataset_path = tmp_path / "qwen.jsonl"
    dataset_path.write_text(json.dumps(build_example(), ensure_ascii=False) + "\n", encoding="utf-8")

    examples = load_examples(dataset_path)

    assert len(examples) == 1
    assert extract_image_path(examples[0]) == Path("line.jpg")


def test_validate_example_rejects_missing_answer() -> None:
    example = build_example(answer="")

    with pytest.raises(ValueError, match="Assistant content"):
        validate_example(example)


def test_resize_image_keeps_aspect_ratio() -> None:
    image = Image.new("RGB", (2000, 1000), "white")

    resized = resize_image(image, 500)

    assert resized.size == (500, 250)


def test_parse_args_supports_validation_and_resume(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "val.jsonl"
    args = parse_args(
        [
            "--train-dataset",
            str(train_path),
            "--eval-dataset",
            str(eval_path),
            "--resume-from-checkpoint",
            "--bf16",
        ]
    )

    assert args.dataset == train_path
    assert args.eval_dataset == eval_path
    assert args.resume_from_checkpoint == "latest"
    assert resolve_resume_value(args.resume_from_checkpoint) is True


def test_build_training_arguments_enables_evaluation(tmp_path: Path) -> None:
    class FakeTrainingArguments:
        def __init__(self, output_dir, eval_strategy=None, **kwargs):
            self.output_dir = output_dir
            self.eval_strategy = eval_strategy
            self.kwargs = kwargs

    args = parse_args(
        [
            "--train-dataset",
            str(tmp_path / "train.jsonl"),
            "--eval-dataset",
            str(tmp_path / "val.jsonl"),
            "--output-dir",
            str(tmp_path / "model"),
        ]
    )

    training_args = build_training_arguments(FakeTrainingArguments, args)

    assert training_args.eval_strategy == "steps"
    assert training_args.kwargs["load_best_model_at_end"] is True
    assert training_args.kwargs["metric_for_best_model"] == "eval_loss"


def test_group_aware_sampler_rotates_variants_without_adjacent_duplicates() -> None:
    sampler = GroupAwareSampler(
        ["page-a", "page-a", "page-a", "page-b", "page-b", "page-c"],
        seed=42,
        variant_policy="all",
    )

    indices = list(sampler)
    groups = [["page-a", "page-a", "page-a", "page-b", "page-b", "page-c"][index] for index in indices]

    assert sorted(indices) == list(range(6))
    assert all(left != right for left, right in zip(groups, groups[1:], strict=False))


def test_group_aware_sampler_can_select_one_variant_per_page() -> None:
    sampler = GroupAwareSampler(["page-a", "page-a", "page-b"], seed=7, variant_policy="one_per_group")

    assert len(list(sampler)) == 2
    assert infer_page_group(build_example("images/author_1_page_001_dark.jpg")) == (
        infer_page_group(build_example("images/author_1_page_001_scan.jpg"))
    )
