from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RecognitionResult:
    text: str
    confidence: float | None
    model_name: str


class BaseRecognizer(ABC):
    @abstractmethod
    def recognize(self, image_path: Path) -> RecognitionResult:
        """Recognize text from an image."""
