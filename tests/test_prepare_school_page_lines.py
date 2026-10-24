import csv
import json
from argparse import Namespace
from pathlib import Path

from PIL import Image

from scripts.prepare_school_page_lines import normalize_line_text, prepare_split


def write_school_fixture(root: Path) -> None:
    images_dir = root / "images"
    images_dir.mkdir(parents=True)
    Image.new("RGB", (240, 120), "white").save(images_dir / "page_001.jpg")
    annotations = {
        "categories": [
            {"id": 0, "name": "pupil_text"},
            {"id": 3, "name": "paper"},
            {"id": 4, "name": "text_line"},
        ],
        "images": [{"height": 120, "width": 240, "id": 1, "file_name": "page_001.jpg"}],
        "annotations": [
            {
                "category_id": 0,
                "image_id": 1,
                "group_id": 10,
                "segmentation": [[[10, 10, 50, 10, 50, 30, 10, 30]]],
                "attributes": {"translation": "Привет"},
            },
            {
                "category_id": 0,
                "image_id": 1,
                "group_id": 10,
                "segmentation": [[[60, 10, 100, 10, 100, 30, 60, 30]]],
                "attributes": {"translation": ","},
            },
            {
                "category_id": 0,
                "image_id": 1,
                "group_id": 10,
                "segmentation": [[[110, 10, 170, 10, 170, 30, 110, 30]]],
                "attributes": {"translation": "мир"},
            },
            {
                "category_id": 0,
                "image_id": 1,
                "group_id": 11,
                "segmentation": [[[20, 70, 130, 70, 130, 90, 20, 90]]],
                "attributes": {"translation": "Вторая строка"},
            },
        ],
    }
    (root / "annotations_train.json").write_text(json.dumps(annotations, ensure_ascii=False), encoding="utf-8")


def test_normalize_line_text_handles_punctuation() -> None:
    assert normalize_line_text(["Привет", ",", "мир", "!"]) == "Привет, мир!"


def test_prepare_split_builds_line_and_page_manifests(tmp_path: Path) -> None:
    source_dir = tmp_path / "schooldataset"
    output_dir = tmp_path / "page_lines"
    write_school_fixture(source_dir)
    args = Namespace(
        source_dir=source_dir,
        output_dir=output_dir,
        categories=["pupil_text"],
        padding=0,
        max_groups_per_split=None,
    )

    line_count, page_count = prepare_split(args, "train")

    assert line_count == 2
    assert page_count == 1
    with (output_dir / "train_lines.csv").open("r", encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    assert rows[0]["text"] == "Привет, мир"
    assert Path(rows[0]["image_path"]).exists()
    assert rows[1]["text"] == "Вторая строка"

    with (output_dir / "train_pages.csv").open("r", encoding="utf-8", newline="") as source:
        page_rows = list(csv.DictReader(source))
    assert len(page_rows) == 1
    text_path = Path(page_rows[0]["text_path"])
    assert text_path.read_text(encoding="utf-8") == "Привет, мир\nВторая строка"
