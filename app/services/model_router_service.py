from __future__ import annotations

from dataclasses import dataclass

from app.image_processing.quality_analysis import ImageQualityReport


@dataclass(frozen=True)
class RoutingDecision:
    requested_mode: str
    selected_mode: str
    reason: str


class ModelRouterService:
    def choose(
        self,
        requested_mode: str,
        quality: ImageQualityReport,
        *,
        qwen_available: bool,
    ) -> RoutingDecision:
        if requested_mode != "auto":
            return RoutingDecision(requested_mode, requested_mode, "User selected the recognition mode.")
        if not qwen_available:
            return RoutingDecision("auto", "trocr_lines", "Qwen runtime is unavailable; using TrOCR line mode.")
        if quality.score < 0.45:
            return RoutingDecision("auto", "qwen_page", "Low image quality; using full-page Qwen OCR.")
        if quality.score < 0.72:
            return RoutingDecision("auto", "hybrid", "Medium image quality; comparing TrOCR and Qwen by line.")
        return RoutingDecision("auto", "trocr_lines", "High image quality; using the faster TrOCR line mode.")
