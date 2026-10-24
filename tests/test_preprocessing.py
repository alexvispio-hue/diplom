from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.image_processing.preprocessing import DocumentNormalizer, ImagePreprocessor


def _save_image(path: Path, image: np.ndarray) -> None:
    success, encoded = cv2.imencode(".png", image)
    assert success
    path.write_bytes(encoded.tobytes())


def test_document_normalizer_corrects_perspective() -> None:
    page = np.full((420, 300, 3), 255, dtype=np.uint8)
    cv2.putText(page, "line one", (35, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    cv2.putText(page, "line two", (35, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    cv2.rectangle(page, (0, 0), (299, 419), (210, 210, 210), 4)

    photo = np.full((540, 620, 3), 35, dtype=np.uint8)
    source = np.array([[0, 0], [299, 0], [299, 419], [0, 419]], dtype=np.float32)
    target = np.array([[170, 55], [480, 95], [430, 500], [105, 455]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(source, target)
    warped = cv2.warpPerspective(page, matrix, (620, 540), borderValue=(35, 35, 35))
    mask = cv2.warpPerspective(np.full((420, 300), 255, dtype=np.uint8), matrix, (620, 540))
    photo[mask > 0] = warped[mask > 0]

    normalized = DocumentNormalizer().normalize(photo)

    assert normalized.shape[0] > 350
    assert normalized.shape[1] > 250
    assert normalized[5, 5].mean() > 180
    assert normalized[-6, -6].mean() > 180


def test_preprocessor_keeps_fallback_when_page_is_not_detected(tmp_path: Path) -> None:
    source = tmp_path / "fragment.png"
    destination = tmp_path / "processed.png"
    image = np.full((80, 260, 3), 255, dtype=np.uint8)
    cv2.putText(image, "word", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)
    _save_image(source, image)

    ImagePreprocessor().preprocess(source, destination)

    processed = Image.open(destination)
    assert processed.mode == "L"
    assert processed.size == (260, 80)
