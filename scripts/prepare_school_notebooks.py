import argparse
from collections import OrderedDict
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop Russian handwritten words from school_notebooks_RU.")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/training/school_notebooks_words"))
    parser.add_argument("--categories", nargs="+", default=["pupil_text", "pupil_comment", "teacher_comment"])
    parser.add_argument("--padding", type=int, default=8)
    parser.add_argument("--max-samples-per-split", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def find_image(images_dir: Path, filename: str) -> Path | None:
    direct = images_dir / filename
    if direct.exists():
        return direct
    matches = list(images_dir.rglob(filename))
    return matches[0] if matches else None


def polygon_bbox(segmentation: list[list[float]], image_width: int, image_height: int, padding: int) -> tuple[int, int, int, int]:
    points = np.asarray(segmentation[0], dtype=np.float32).reshape(-1, 2)
    x, y, width, height = cv2.boundingRect(points.astype(np.int32))
    return (
        max(0, x - padding),
        max(0, y - padding),
        min(image_width, x + width + padding),
        min(image_height, y + height + padding),
    )


def prepare_split(args: argparse.Namespace, split: str) -> int:
    data = json.loads((args.source_dir / f"annotations_{split}.json").read_text(encoding="utf-8"))
    category_names = {category["id"]: category["name"] for category in data["categories"]}
    image_names = {image["id"]: image["file_name"] for image in data["images"]}
    split_dir = args.output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / f"{split}.tsv"
    image_cache: OrderedDict[int, Image.Image] = OrderedDict()
    path_cache: dict[int, Path | None] = {}
    written = 0
    if args.resume and manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8", newline="") as source:
            written = sum(1 for _ in source)

    mode = "a" if args.resume else "w"
    with manifest_path.open(mode, encoding="utf-8", newline="") as target:
        writer = csv.writer(target, delimiter="\t")
        eligible_index = 0
        for annotation in data["annotations"]:
            if category_names.get(annotation["category_id"]) not in args.categories:
                continue
            text = (annotation.get("attributes", {}).get("translation") or "").strip()
            if not text or not annotation.get("segmentation"):
                continue
            if eligible_index < written:
                eligible_index += 1
                continue
            if args.max_samples_per_split is not None and written >= args.max_samples_per_split:
                break

            image_id = annotation["image_id"]
            if image_id not in path_cache:
                path_cache[image_id] = find_image(args.source_dir / "images", image_names[image_id])
            image_path = path_cache[image_id]
            if image_path is None:
                eligible_index += 1
                continue
            if image_id not in image_cache:
                image_cache[image_id] = Image.open(image_path).convert("RGB")
                if len(image_cache) > 4:
                    _, old_image = image_cache.popitem(last=False)
                    old_image.close()
            else:
                image_cache.move_to_end(image_id)
            image = image_cache[image_id]

            x_start, y_start, x_end, y_end = polygon_bbox(annotation["segmentation"], image.width, image.height, args.padding)
            if x_end - x_start < 8 or y_end - y_start < 8:
                eligible_index += 1
                continue
            filename = f"{written:07d}.jpg"
            image.crop((x_start, y_start, x_end, y_end)).save(split_dir / filename, quality=95)
            writer.writerow((filename, text))
            written += 1
            eligible_index += 1

    for image in image_cache.values():
        image.close()
    return written


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        print(f"{split}: {prepare_split(args, split)} samples")


if __name__ == "__main__":
    main()
