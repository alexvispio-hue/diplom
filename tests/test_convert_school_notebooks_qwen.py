import csv
import json
from pathlib import Path

from PIL import Image

from scripts.convert_school_notebooks_qwen import convert_split


def write_coco_fixture(root: Path) -> None:
    images_dir = root / "images"
    images_dir.mkdir(parents=True)
    Image.new("RGB", (240, 120), "white").save(images_dir / "page_001.jpg")
    annotations = {
        "categories": [
            {"id": 0, "name": "pupil_text"},
            {"id": 1, "name": "teacher_comment"},
            {"id": 2, "name": "paper"},
        ],
        "images": [{"height": 120, "width": 240, "id": 1, "file_name": "page_001.jpg"}],
        "annotations": [
            {
                "category_id": 0,
                "image_id": 1,
                "group_id": 10,
                "segmentation": [[10, 10, 50, 10, 50, 30, 10, 30]],
                "attributes": {"translation": "Привет"},
            },
            {
                "category_id": 0,
                "image_id": 1,
                "group_id": 10,
                "segmentation": [[60, 10, 100, 10, 100, 30, 60, 30]],
                "attributes": {"translation": ","},
            },
            {
                "category_id": 0,
                "image_id": 1,
                "group_id": 10,
                "segmentation": [[110, 10, 170, 10, 170, 30, 110, 30]],
                "attributes": {"translation": "мир"},
            },
            {
                "category_id": 1,
                "image_id": 1,
                "group_id": 11,
                "segmentation": [[20, 70, 130, 70, 130, 90, 20, 90]],
                "attributes": {"translation": "Вторая строка"},
            },
        ],
    }
    (root / "annotations_train.json").write_text(json.dumps(annotations, ensure_ascii=False), encoding="utf-8")


def test_convert_split_builds_page_layout_and_sft_jsonl(tmp_path: Path) -> None:
    source_dir = tmp_path / "schooldataset"
    output_dir = tmp_path / "school_qwen_pages"
    write_coco_fixture(source_dir)

    stats = convert_split(
        source_dir=source_dir,
        output_dir=output_dir,
        split="train",
        categories={"pupil_text", "teacher_comment"},
        padding=0,
        max_pages=None,
        include_line_examples=False,
    )

    assert stats["pages"] == 1
    assert stats["lines"] == 2
    assert stats["examples"] == 2

    with (output_dir / "train_pages.csv").open("r", encoding="utf-8", newline="") as source:
        page_rows = list(csv.DictReader(source))
    assert page_rows[0]["line_count"] == "2"
    text_path = Path(page_rows[0]["text_path"])
    assert text_path.read_text(encoding="utf-8") == "Привет, мир\nВторая строка"

    with (output_dir / "train_lines.csv").open("r", encoding="utf-8", newline="") as source:
        line_rows = list(csv.DictReader(source))
    assert line_rows[0]["text"] == "Привет, мир"
    assert line_rows[0]["x1"] == "10"
    assert line_rows[1]["text"] == "Вторая строка"

    examples = [
        json.loads(line)
        for line in (output_dir / "qwen_sft" / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert examples[0]["messages"][1]["content"] == "Привет, мир\nВторая строка"
    layout = json.loads(examples[1]["messages"][1]["content"])
    assert layout == [
        {"line_number": 1, "text": "Привет, мир", "box": [10, 10, 170, 30]},
        {"line_number": 2, "text": "Вторая строка", "box": [20, 70, 130, 90]},
    ]
