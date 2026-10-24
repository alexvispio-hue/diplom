import json
import zipfile
from pathlib import Path

from scripts.convert_hwr200_qwen import convert


def write_fixture(root: Path) -> None:
    annotations = {
        "full_text": "Точный текст одной страницы.",
        "sentences": [],
        "words_count": 4,
    }
    with zipfile.ZipFile(root / "annotations.zip", "w") as archive:
        archive.writestr("annotations/Originals/0/original0.json", json.dumps(annotations, ensure_ascii=False))
    with zipfile.ZipFile(root / "hwr200_0_19.zip", "w") as archive:
        for condition in ("Сканы", "ФотоСветлое", "ФотоТемное"):
            archive.writestr(f"hw_dataset/0/original0/{condition}/1.jpg", b"image")


def test_convert_extracts_only_complete_single_page_document_groups(tmp_path: Path) -> None:
    source = tmp_path / "hwr200"
    output = tmp_path / "converted"
    source.mkdir()
    write_fixture(source)

    stats = convert(source, output, {"Сканы", "ФотоСветлое", "ФотоТемное"}, 1, 0, None)

    assert stats["train"] == 3
    examples = [json.loads(line) for line in (output / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(examples) == 3
    assert examples[0]["messages"][1]["content"] == "Точный текст одной страницы."
    assert Path(examples[0]["messages"][0]["content"][0]["image"]).exists()
