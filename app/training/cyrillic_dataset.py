import csv
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass(frozen=True)
class CyrillicSample:
    image_path: Path
    text: str


def load_cyrillic_samples(labels_path: Path, images_dir: Path, limit: int | None = None) -> list[CyrillicSample]:
    samples: list[CyrillicSample] = []
    with labels_path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.reader(source, delimiter="\t")
        for filename, text in reader:
            image_path = images_dir / filename
            if image_path.exists():
                samples.append(CyrillicSample(image_path=image_path, text=text))
            if limit is not None and len(samples) >= limit:
                break
    return samples


class TrOCRTrainingDataset:
    def __init__(self, samples: list[CyrillicSample], processor, max_target_length: int = 64) -> None:
        self.samples = samples
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        labels = self.processor.tokenizer(
            sample.text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels}
