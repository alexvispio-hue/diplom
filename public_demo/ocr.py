"""Small, stateless inference entry point for the public fragment demo."""
from __future__ import annotations

import io
import os
import re
from threading import Lock

from PIL import Image, ImageOps, UnidentifiedImageError

BASE_MODEL = "kazars24/trocr-base-handwritten-ru"
MAX_BYTES = 5 * 1024 * 1024
MAX_PIXELS = 12_000_000


def read_image(payload: bytes) -> Image.Image:
    if not payload or len(payload) > MAX_BYTES:
        raise ValueError("Выберите изображение размером до 5 МБ.")
    try:
        with Image.open(io.BytesIO(payload)) as source:
            if source.format not in {"PNG", "JPEG", "WEBP"}:
                raise ValueError("Поддерживаются только PNG, JPEG и WEBP.")
            if source.width * source.height > MAX_PIXELS:
                raise ValueError("Изображение слишком большое: максимум 12 мегапикселей.")
            source.load()
            oriented = ImageOps.exif_transpose(source)
            if oriented.mode in {"RGBA", "LA"} or "transparency" in oriented.info:
                rgba = oriented.convert("RGBA")
                background = Image.new("RGBA", rgba.size, "white")
                return Image.alpha_composite(background, rgba).convert("RGB")
            return oriented.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Не удалось прочитать изображение. Загрузите другой файл.") from exc


def crop_fragment(image: Image.Image, horizontal: tuple[int, int], vertical: tuple[int, int]) -> Image.Image:
    if not (0 <= horizontal[0] < horizontal[1] <= 100 and 0 <= vertical[0] < vertical[1] <= 100):
        raise ValueError("Выделите непустой фрагмент изображения.")
    left, right = (round(image.width * value / 100) for value in horizontal)
    top, bottom = (round(image.height * value / 100) for value in vertical)
    if right - left < 8 or bottom - top < 8:
        raise ValueError("Фрагмент слишком маленький. Выделите область не менее 8 × 8 пикселей.")
    return image.crop((left, top, right, bottom))


class FragmentRecognizer:
    """One model per server process; concurrent visitors do not share results."""

    def __init__(self, model_source: str = BASE_MODEL) -> None:
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        torch.set_num_threads(max(1, min(2, os.cpu_count() or 1)))
        self.torch = torch
        self.lock = Lock()
        self.processor = TrOCRProcessor.from_pretrained(model_source)
        self.model = VisionEncoderDecoderModel.from_pretrained(
            model_source,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).eval()

    def recognize(self, image: Image.Image) -> str:
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Демо сейчас обрабатывает другой запрос. Повторите попытку чуть позже.")
        try:
            pixels = self.processor(images=image, return_tensors="pt").pixel_values
            with self.torch.inference_mode():
                tokens = self.model.generate(pixels, max_new_tokens=96, num_beams=1)
            text = self.processor.batch_decode(tokens, skip_special_tokens=True)[0]
            return re.sub(r"[ \t]+", " ", text.replace("\r", "\n")).strip()
        finally:
            self.lock.release()
