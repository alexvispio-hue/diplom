import cv2
import numpy as np

from app.image_processing.quality_analysis import ImageQualityAnalyzer


def test_quality_analyzer_flags_blurred_low_contrast_image() -> None:
    image = np.full((300, 500), 220, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok

    report = ImageQualityAnalyzer().analyze_bytes(encoded.tobytes())

    assert "blur" in report.issues
    assert "low_contrast" in report.issues
    assert report.recommended_preprocessing is True
    assert 0 <= report.score <= 1
