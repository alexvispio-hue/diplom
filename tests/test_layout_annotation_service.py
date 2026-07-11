import csv
from pathlib import Path

import pytest

from app.services.layout_annotation_service import LayoutAnnotationExporter, parse_layout_json


def test_parse_layout_json_accepts_plain_json() -> None:
    lines = parse_layout_json('[{"line_number": 2, "text": "вторая", "box": [10, 20, 110, 40]}]')

    assert len(lines) == 1
    assert lines[0].line_number == 2
    assert lines[0].text == "вторая"
    assert lines[0].box == (10, 20, 110, 40)


def test_parse_layout_json_accepts_fenced_json() -> None:
    lines = parse_layout_json(
        """
        ```json
        [{"line_number": 1, "text": "первая", "box": [1, 2, 30, 40]}]
        ```
        """
    )

    assert lines[0].text == "первая"


def test_parse_layout_json_rejects_invalid_box() -> None:
    with pytest.raises(ValueError, match="box"):
        parse_layout_json('[{"line_number": 1, "text": "bad", "box": [1, 2]}]')


def test_layout_annotation_exporter_writes_lines_csv(tmp_path: Path) -> None:
    layout_csv = tmp_path / "qwen_layout.csv"
    layout_csv.write_text(
        "image_path,page_id,recognized_text\n"
        "\"page.png\",page_001,\"[{\"\"line_number\"\": 1, \"\"text\"\": \"\"строка\"\", \"\"box\"\": [10, 20, 110, 40]}]\"\n",
        encoding="utf-8",
    )
    output_csv = tmp_path / "lines.csv"

    rows = LayoutAnnotationExporter().export(layout_csv, output_csv)

    assert rows == 1
    with output_csv.open("r", encoding="utf-8", newline="") as source:
        exported = list(csv.DictReader(source))
    assert exported[0]["image_path"] == "page.png"
    assert exported[0]["text"] == "строка"
    assert exported[0]["x1"] == "10"
    assert exported[0]["source_image"] == "page.png"


def test_layout_annotation_exporter_accepts_utf8_sig_input(tmp_path: Path) -> None:
    layout_csv = tmp_path / "qwen_layout.csv"
    layout_csv.write_text(
        "image_path,page_id,recognized_text\n"
        "\"page.png\",page_001,\"[{\"\"line_number\"\": 1, \"\"text\"\": \"\"строка\"\", \"\"box\"\": [10, 20, 110, 40]}]\"\n",
        encoding="utf-8-sig",
    )
    output_csv = tmp_path / "lines.csv"

    rows = LayoutAnnotationExporter().export(layout_csv, output_csv)

    assert rows == 1
