import io
import random

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


class HandwritingPhotoAugmenter:
    """Adds lightweight camera artifacts while preserving the source text."""

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        original_size = image.size
        if random.random() < 0.45:
            image = add_notebook_background(image)
        if random.random() < 0.8:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.65, 1.25))
        if random.random() < 0.7:
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.75, 1.35))
        if random.random() < 0.55:
            image = image.rotate(random.uniform(-2.5, 2.5), resample=Image.Resampling.BICUBIC, expand=True, fillcolor="white")
        if random.random() < 0.35:
            image = add_perspective_distortion(image)
        if random.random() < 0.45:
            image = add_subtle_shadow(image)
        if random.random() < 0.35:
            image = add_paper_texture(image)
        if random.random() < 0.35:
            image = image.filter(ImageFilter.GaussianBlur(random.uniform(0.2, 1.0)))
        if random.random() < 0.4:
            image = self._add_noise(image)
        if random.random() < 0.4:
            image = self._compress_jpeg(image)
        return image.resize(original_size, Image.Resampling.BICUBIC)

    @staticmethod
    def _add_noise(image: Image.Image) -> Image.Image:
        array = np.asarray(image).astype(np.float32)
        noise = np.random.normal(0, random.uniform(2.0, 9.0), array.shape)
        return Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8))

    @staticmethod
    def _compress_jpeg(image: Image.Image) -> Image.Image:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=random.randint(50, 90))
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


def add_subtle_shadow(image: Image.Image) -> Image.Image:
    array = np.asarray(image.convert("RGB")).astype(np.float32)
    height, width = array.shape[:2]
    direction = random.choice(("horizontal", "vertical"))
    if direction == "horizontal":
        gradient = np.linspace(random.uniform(0.7, 1.0), 1.0, width, dtype=np.float32)[None, :, None]
    else:
        gradient = np.linspace(random.uniform(0.7, 1.0), 1.0, height, dtype=np.float32)[:, None, None]
    return Image.fromarray(np.clip(array * gradient, 0, 255).astype(np.uint8))


def add_paper_texture(image: Image.Image) -> Image.Image:
    array = np.asarray(image.convert("RGB")).astype(np.float32)
    texture = cv2.GaussianBlur(np.random.normal(0, 4.0, array.shape).astype(np.float32), (0, 0), 1.2)
    return Image.fromarray(np.clip(array + texture, 0, 255).astype(np.uint8))


def add_perspective_distortion(image: Image.Image, max_shift_ratio: float = 0.04) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    max_shift_x = max(1, int(width * max_shift_ratio))
    max_shift_y = max(1, int(height * max_shift_ratio))
    source = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    destination = np.array(
        [
            [random.randint(0, max_shift_x), random.randint(0, max_shift_y)],
            [width - 1 - random.randint(0, max_shift_x), random.randint(0, max_shift_y)],
            [width - 1 - random.randint(0, max_shift_x), height - 1 - random.randint(0, max_shift_y)],
            [random.randint(0, max_shift_x), height - 1 - random.randint(0, max_shift_y)],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(source, destination)
    array = np.asarray(image)
    warped = cv2.warpPerspective(array, matrix, (width, height), borderValue=(255, 255, 255))
    return Image.fromarray(warped)


def add_notebook_background(image: Image.Image) -> Image.Image:
    array = np.asarray(image.convert("RGB")).astype(np.float32)
    height, width = array.shape[:2]
    line_color = np.array([205, 220, 238], dtype=np.float32)
    mode = random.choice(("grid", "ruled"))
    step = random.randint(18, 34)
    thickness = 1

    for y in range(step, height, step):
        array[max(0, y - thickness) : min(height, y + thickness + 1), :, :] = (
            array[max(0, y - thickness) : min(height, y + thickness + 1), :, :] * 0.65 + line_color * 0.35
        )
    if mode == "grid":
        for x in range(step, width, step):
            array[:, max(0, x - thickness) : min(width, x + thickness + 1), :] = (
                array[:, max(0, x - thickness) : min(width, x + thickness + 1), :] * 0.65 + line_color * 0.35
            )
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
