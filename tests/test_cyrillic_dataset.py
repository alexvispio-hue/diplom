from pathlib import Path

from PIL import Image

from app.training.cyrillic_dataset import CyrillicSample, TrOCRTrainingDataset, load_cyrillic_samples


def test_load_cyrillic_samples_reads_tsv_and_existing_images(tmp_path: Path) -> None:
    images_dir = tmp_path / "train"
    images_dir.mkdir()
    Image.new("RGB", (10, 10), "white").save(images_dir / "sample.png")
    labels_path = tmp_path / "train.tsv"
    labels_path.write_text("sample.png\tПример текста\nmissing.png\tНе включать\n", encoding="utf-8")

    samples = load_cyrillic_samples(labels_path, images_dir)

    assert len(samples) == 1
    assert samples[0].text == "Пример текста"


def test_training_dataset_applies_transform(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (20, 10), "white").save(image_path)

    class Processor:
        class Tokenizer:
            pad_token_id = 0

            def __call__(self, *_args, **_kwargs):
                import torch

                return type("Tokens", (), {"input_ids": torch.tensor([[1, 0]])})

        tokenizer = Tokenizer()

        def __call__(self, images, **_kwargs):
            import torch

            assert images.size == (40, 20)
            return type("Pixels", (), {"pixel_values": torch.zeros((1, 3, 20, 40))})

    dataset = TrOCRTrainingDataset(
        [CyrillicSample(image_path=image_path, text="текст")],
        Processor(),
        transform=lambda image: image.resize((40, 20)),
    )

    sample = dataset[0]

    assert sample["pixel_values"].shape == (3, 20, 40)
    assert sample["labels"].tolist() == [1, -100]
