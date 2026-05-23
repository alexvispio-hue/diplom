from pathlib import Path

import cv2
import numpy as np
from PIL import Image


class ImagePreprocessor:
    """Prepares document images for handwriting recognition."""

    def preprocess(self, source_path: Path, destination_path: Path) -> Path:
        image = cv2.imdecode(np.fromfile(str(source_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Не удалось прочитать изображение.")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
        normalized = self._normalize_contrast(denoised)

        destination_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(normalized).save(destination_path)
        return destination_path

    @staticmethod
    def _normalize_contrast(image: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)
