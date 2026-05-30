import argparse
import csv
from pathlib import Path
import random
import re
import sys

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.training.image_augmentation import add_paper_texture, add_subtle_shadow


CYRILLIC_WORD = re.compile(r"[А-Яа-яЁё-]{2,32}")
CYRILLIC_PROBE = "Съешь ещё этих мягких французских булок"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic Russian handwriting word images.")
    parser.add_argument("--source-manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--fonts-dir", type=Path, default=Path("/mnt/c/Windows/Fonts"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/training/synthetic_ru_words"))
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_words(manifest_paths: list[Path]) -> list[str]:
    words: set[str] = set()
    for manifest_path in manifest_paths:
        with manifest_path.open("r", encoding="utf-8", newline="") as source:
            for row in csv.reader(source, delimiter="\t"):
                if len(row) < 2:
                    continue
                words.update(word.lower() for word in CYRILLIC_WORD.findall(row[1]))
    if not words:
        raise ValueError("В манифестах не найдены русские слова.")
    return sorted(words)


def find_cyrillic_fonts(fonts_dir: Path) -> list[Path]:
    fonts = []
    for font_path in fonts_dir.rglob("*"):
        if font_path.suffix.lower() not in {".ttf", ".otf"}:
            continue
        try:
            ImageFont.truetype(str(font_path), 36).getbbox(CYRILLIC_PROBE)
        except OSError:
            continue
        fonts.append(font_path)
    if not fonts:
        raise ValueError(f"Не найдены шрифты с кириллицей: {fonts_dir}")
    return fonts


def render_word(word: str, font_path: Path) -> Image.Image:
    font_size = random.randint(30, 64)
    font = ImageFont.truetype(str(font_path), font_size)
    left, top, right, bottom = font.getbbox(word)
    padding_x = random.randint(12, 30)
    padding_y = random.randint(8, 22)
    image = Image.new("RGB", (right - left + padding_x * 2, bottom - top + padding_y * 2), random.choice(("white", "#f7f4ed", "#f2f2f0")))
    draw = ImageDraw.Draw(image)
    ink = random.choice(("#111111", "#202020", "#182038", "#302725"))
    draw.text((padding_x - left, padding_y - top), word, font=font, fill=ink, stroke_width=random.choice((0, 0, 1)))
    if random.random() < 0.7:
        image = image.rotate(random.uniform(-3.0, 3.0), resample=Image.Resampling.BICUBIC, expand=True, fillcolor="white")
    if random.random() < 0.6:
        image = add_paper_texture(image)
    if random.random() < 0.5:
        image = add_subtle_shadow(image)
    return image


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    words = load_words(args.source_manifests)
    fonts = find_cyrillic_fonts(args.fonts_dir)
    train_dir = args.output_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "train.tsv"
    with manifest_path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target, delimiter="\t")
        for index in range(args.samples):
            word = random.choice(words)
            filename = f"{index:07d}.jpg"
            render_word(word, random.choice(fonts)).save(train_dir / filename, quality=random.randint(55, 95))
            writer.writerow((filename, word))
    print(f"generated: {args.samples}")
    print(f"words: {len(words)}")
    print(f"fonts: {len(fonts)}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
