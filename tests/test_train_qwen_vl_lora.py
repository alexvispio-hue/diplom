import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.train_qwen_vl_lora import extract_image_path, load_examples, resize_image, validate_example


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
