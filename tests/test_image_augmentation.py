import random

import numpy as np
from PIL import Image

from app.training.image_augmentation import (
    HandwritingPhotoAugmenter,
    add_notebook_background,
    add_perspective_distortion,
    add_subtle_shadow,
)


def build_test_image() -> Image.Image:
    image = Image.new("RGB", (240, 80), "white")
    array = np.asarray(image).copy()
    array[30:45, 30:190] = (20, 20, 20)
    return Image.fromarray(array)


def test_perspective_distortion_preserves_size() -> None:
    random.seed(1)
    image = build_test_image()

    distorted = add_perspective_distortion(image)

    assert distorted.size == image.size
    assert distorted.mode == "RGB"


def test_notebook_background_changes_pixels() -> None:
    random.seed(1)
    image = Image.new("RGB", (120, 80), "white")

    augmented = add_notebook_background(image)

    assert augmented.size == image.size
    assert np.any(np.asarray(augmented) != np.asarray(image))


def test_shadow_changes_pixels() -> None:
    random.seed(1)
    image = Image.new("RGB", (120, 80), "white")

    shadowed = add_subtle_shadow(image)

    assert shadowed.size == image.size
    assert np.any(np.asarray(shadowed) != np.asarray(image))


def test_handwriting_photo_augmenter_preserves_original_size() -> None:
    random.seed(2)
    image = build_test_image()

    augmented = HandwritingPhotoAugmenter()(image)

    assert augmented.size == image.size
    assert augmented.mode == "RGB"
