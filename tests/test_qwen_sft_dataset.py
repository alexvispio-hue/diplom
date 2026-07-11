import json
from pathlib import Path

from scripts.build_qwen_sft_dataset import iter_line_examples, iter_page_examples


def test_builds_qwen_line_and_layout_examples(tmp_path: Path) -> None:
    lines_csv = tmp_path / "lines.csv"
    lines_csv.write_text(
        "image_path,text,page_id,line_number,x1,y1,x2,y2\n"
        "data/lines/line_001.jpg,пример,page_001,1,10,20,300,60\n",
        encoding="utf-8",
    )

    examples = iter_line_examples(lines_csv, include_layout=True)

    assert len(examples) == 2
    line_example = examples[0]["messages"]
    assert line_example[0]["content"][0]["image"] == "data/lines/line_001.jpg"
    assert line_example[1]["content"] == "пример"
    layout_answer = json.loads(examples[1]["messages"][1]["content"])
    assert layout_answer == [{"line_number": 1, "text": "пример", "box": [10, 20, 300, 60]}]


def test_builds_qwen_page_examples(tmp_path: Path) -> None:
    text_path = tmp_path / "page_001.txt"
    text_path.write_text("первая строка\nвторая строка\n", encoding="utf-8")
    pages_csv = tmp_path / "pages.csv"
    pages_csv.write_text(
        f"image_path,text_path,page_id\n"
        f"data/pages/page_001.jpg,{text_path},page_001\n",
        encoding="utf-8",
    )

    examples = iter_page_examples(pages_csv)

    assert len(examples) == 1
    messages = examples[0]["messages"]
    assert messages[0]["content"][0]["image"] == "data/pages/page_001.jpg"
    assert messages[1]["content"] == "первая строка\nвторая строка"
