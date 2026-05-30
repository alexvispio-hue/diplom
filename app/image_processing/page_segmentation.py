from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class TextLine:
    index: int
    image_path: Path
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class PageSegmentationResult:
    annotated_image_path: Path
    lines: list[TextLine]


class PageLineSegmenter:
    """Extracts text lines from a page image using projection profiles."""

    def segment(
        self,
        source_path: Path,
        output_dir: Path,
        annotated_image_path: Path,
    ) -> PageSegmentationResult:
        image = cv2.imdecode(np.fromfile(str(source_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Не удалось прочитать изображение страницы.")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        normalized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        binary = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        binary = self._remove_notebook_lines(binary)

        row_projection = np.count_nonzero(binary, axis=1)
        active_rows = row_projection >= max(3, int(binary.shape[1] * 0.003))
        bands = self._find_bands(active_rows, max_gap=max(3, binary.shape[0] // 180))

        output_dir.mkdir(parents=True, exist_ok=True)
        annotated = image.copy()
        lines: list[TextLine] = []
        for y_start, y_end in bands:
            if y_end - y_start < max(6, binary.shape[0] // 180):
                continue

            band = binary[y_start:y_end, :]
            active_columns = np.flatnonzero(np.count_nonzero(band, axis=0))
            if active_columns.size == 0:
                continue

            x_start = int(active_columns[0])
            x_end = int(active_columns[-1]) + 1
            padding_x = max(8, binary.shape[1] // 120)
            padding_y = max(5, binary.shape[0] // 250)
            x_start = max(0, x_start - padding_x)
            x_end = min(binary.shape[1], x_end + padding_x)
            y_start = max(0, y_start - padding_y)
            y_end = min(binary.shape[0], y_end + padding_y)

            crop = gray[y_start:y_end, x_start:x_end]
            if crop.shape[1] < 20 or crop.shape[0] < 8:
                continue

            index = len(lines)
            line_path = output_dir / f"line_{index:03d}.png"
            Image.fromarray(crop).save(line_path)
            lines.append(TextLine(index=index, image_path=line_path, bbox=(x_start, y_start, x_end, y_end)))
            cv2.rectangle(annotated, (x_start, y_start), (x_end, y_end), (36, 139, 92), 2)
            cv2.putText(
                annotated,
                str(index + 1),
                (x_start, max(18, y_start - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (36, 139, 92),
                2,
                cv2.LINE_AA,
            )

        if not lines:
            raise ValueError("Не удалось выделить строки текста на странице.")

        annotated_image_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imencode(".png", annotated)[1].tofile(str(annotated_image_path))
        return PageSegmentationResult(annotated_image_path=annotated_image_path, lines=lines)

    @staticmethod
    def _remove_notebook_lines(binary: np.ndarray) -> np.ndarray:
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, binary.shape[1] // 8), 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(40, binary.shape[0] // 8)))
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
        return cv2.subtract(binary, cv2.bitwise_or(horizontal, vertical))

    @staticmethod
    def _find_bands(active_rows: np.ndarray, max_gap: int) -> list[tuple[int, int]]:
        active_indices = np.flatnonzero(active_rows)
        if active_indices.size == 0:
            return []

        bands: list[tuple[int, int]] = []
        start = int(active_indices[0])
        previous = start
        for current_value in active_indices[1:]:
            current = int(current_value)
            if current - previous > max_gap + 1:
                bands.append((start, previous + 1))
                start = current
            previous = current
        bands.append((start, previous + 1))
        return bands
