from pathlib import Path

import cv2
import numpy as np
from PIL import Image


class DocumentNormalizer:
    """Finds a photographed page and applies a perspective correction."""

    def normalize(self, image: np.ndarray) -> np.ndarray:
        page_corners = self.find_page_corners(image)
        if page_corners is None:
            return image
        return self._warp_perspective(image, page_corners)

    def find_page_corners(self, image: np.ndarray) -> np.ndarray | None:
        height, width = image.shape[:2]
        scale = 1.0
        max_side = max(height, width)
        working = image
        if max_side > 900:
            scale = 900 / max_side
            working = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        edges = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = working.shape[0] * working.shape[1] * 0.18
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            area = cv2.contourArea(contour)
            if area < min_area:
                break

            perimeter = cv2.arcLength(contour, True)
            approximated = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approximated) != 4 or not cv2.isContourConvex(approximated):
                continue

            corners = approximated.reshape(4, 2).astype(np.float32) / scale
            return self._order_points(corners)

        return None

    @staticmethod
    def _order_points(points: np.ndarray) -> np.ndarray:
        ordered = np.zeros((4, 2), dtype=np.float32)
        point_sum = points.sum(axis=1)
        point_diff = np.diff(points, axis=1).reshape(-1)
        ordered[0] = points[np.argmin(point_sum)]
        ordered[2] = points[np.argmax(point_sum)]
        ordered[1] = points[np.argmin(point_diff)]
        ordered[3] = points[np.argmax(point_diff)]
        return ordered

    @staticmethod
    def _warp_perspective(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        top_left, top_right, bottom_right, bottom_left = corners
        width_top = np.linalg.norm(top_right - top_left)
        width_bottom = np.linalg.norm(bottom_right - bottom_left)
        height_right = np.linalg.norm(bottom_right - top_right)
        height_left = np.linalg.norm(bottom_left - top_left)
        target_width = max(1, int(max(width_top, width_bottom)))
        target_height = max(1, int(max(height_right, height_left)))

        destination = np.array(
            [
                [0, 0],
                [target_width - 1, 0],
                [target_width - 1, target_height - 1],
                [0, target_height - 1],
            ],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(corners, destination)
        return cv2.warpPerspective(image, matrix, (target_width, target_height), borderValue=(255, 255, 255))


class ImagePreprocessor:
    """Prepares document images for handwriting recognition."""

    def __init__(self, document_normalizer: DocumentNormalizer | None = None) -> None:
        self.document_normalizer = document_normalizer or DocumentNormalizer()

    def preprocess(self, source_path: Path, destination_path: Path) -> Path:
        image = cv2.imdecode(np.fromfile(str(source_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not read image.")

        image = self.document_normalizer.normalize(image)
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
