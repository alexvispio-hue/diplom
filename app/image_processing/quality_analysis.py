from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class ImageQualityReport:
    score: float
    blur_variance: float
    brightness: float
    contrast: float
    width: int
    height: int
    issues: list[str]
    recommended_preprocessing: bool

    def to_dict(self) -> dict:
        return asdict(self)


class ImageQualityAnalyzer:
    def analyze(self, image_path: Path) -> ImageQualityReport:
        image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read image for quality analysis: {image_path}")
        return self.analyze_array(image)

    def analyze_bytes(self, data: bytes) -> ImageQualityReport:
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Could not read image bytes for quality analysis.")
        return self.analyze_array(image)

    def analyze_array(self, image: Any) -> ImageQualityReport:
        height, width = image.shape[:2]
        blur_variance = float(cv2.Laplacian(image, cv2.CV_64F).var())
        brightness = float(image.mean())
        contrast = float(image.std())

        issues = []
        if blur_variance < 70:
            issues.append("blur")
        if brightness < 55:
            issues.append("underexposed")
        elif brightness > 225:
            issues.append("overexposed")
        if contrast < 25:
            issues.append("low_contrast")
        if min(width, height) < 256:
            issues.append("low_resolution")

        blur_score = min(1.0, blur_variance / 180)
        exposure_score = max(0.0, 1.0 - abs(brightness - 150) / 150)
        contrast_score = min(1.0, contrast / 65)
        resolution_score = min(1.0, min(width, height) / 900)
        score = round(
            max(0.0, min(1.0, blur_score * 0.35 + exposure_score * 0.25 + contrast_score * 0.25 + resolution_score * 0.15)),
            4,
        )
        return ImageQualityReport(
            score=score,
            blur_variance=round(blur_variance, 3),
            brightness=round(brightness, 3),
            contrast=round(contrast, 3),
            width=width,
            height=height,
            issues=issues,
            recommended_preprocessing=bool(issues),
        )
