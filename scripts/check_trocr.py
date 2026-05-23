from pathlib import Path
import sys

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.ocr.trocr_recognizer import TrOCRRecognizer


def build_sample_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (640, 160), "white")
    draw = ImageDraw.Draw(image)
    draw.text((40, 55), "hello world", fill="black")
    image.save(path)
    return path


def main() -> None:
    sample_path = build_sample_image(Path("data/samples/trocr_smoke.png"))
    recognizer = TrOCRRecognizer()
    result = recognizer.recognize(sample_path)
    print(f"model: {result.model_name}")
    print(f"text: {result.text}")


if __name__ == "__main__":
    main()
