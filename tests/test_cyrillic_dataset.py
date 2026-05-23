from pathlib import Path

from PIL import Image

from app.training.cyrillic_dataset import load_cyrillic_samples


def test_load_cyrillic_samples_reads_tsv_and_existing_images(tmp_path: Path) -> None:
    images_dir = tmp_path / "train"
    images_dir.mkdir()
    Image.new("RGB", (10, 10), "white").save(images_dir / "sample.png")
    labels_path = tmp_path / "train.tsv"
    labels_path.write_text("sample.png\tПример текста\nmissing.png\tНе включать\n", encoding="utf-8")

    samples = load_cyrillic_samples(labels_path, images_dir)

    assert len(samples) == 1
    assert samples[0].text == "Пример текста"
