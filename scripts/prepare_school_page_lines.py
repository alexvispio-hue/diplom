import argparse
from collections import OrderedDict, defaultdict
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


DEFAULT_TEXT_CATEGORIES = ["pupil_text", "pupil_comment", "teacher_comment"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare page/line OCR manifests from school_notebooks_RU COCO data.")
    parser.add_argument("--source-dir", type=Path, default=Path("data/training/schooldataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/training/page_lines"))
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--categories", nargs="+", default=DEFAULT_TEXT_CATEGORIES)
    parser.add_argument("--padding", type=int, default=12)
    parser.add_argument("--max-groups-per-split", type=int)
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


def union_bbox(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def normalize_line_text(tokens: list[str]) -> str:
    text = " ".join(token.strip() for token in tokens if token and token.strip())
    for punctuation in (".", ",", ":", ";", "!", "?", ")", "]", "}", "»"):
        text = text.replace(f" {punctuation}", punctuation)
    for punctuation in ("(", "[", "{", "«"):
        text = text.replace(f"{punctuation} ", punctuation)
    text = text.replace(" - ", "-")
    return " ".join(text.split())


def build_groups(data: dict, selected_categories: set[str], padding: int):
    category_names = {category["id"]: category["name"] for category in data["categories"]}
    image_info = {image["id"]: image for image in data["images"]}
    groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for annotation in data["annotations"]:
        if category_names.get(annotation["category_id"]) not in selected_categories:
            continue
        if "group_id" not in annotation or not annotation.get("segmentation"):
            continue
        text = (annotation.get("attributes", {}).get("translation") or "").strip()
        if not text:
            continue
        image = image_info[annotation["image_id"]]
        bbox = polygon_bbox(annotation["segmentation"], image["width"], image["height"], padding)
        groups[(annotation["image_id"], annotation["group_id"])].append(
            {
                "text": text,
                "bbox": bbox,
                "x_center": (bbox[0] + bbox[2]) / 2,
                "y_center": (bbox[1] + bbox[3]) / 2,
            }
        )
    return image_info, groups


def prepare_split(args: argparse.Namespace, split: str) -> tuple[int, int]:
    annotation_path = args.source_dir / f"annotations_{split}.json"
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    image_info, groups = build_groups(data, set(args.categories), args.padding)

    output_lines_dir = args.output_dir / "lines" / split
    output_texts_dir = args.output_dir / "pages_text" / split
    output_lines_dir.mkdir(parents=True, exist_ok=True)
    output_texts_dir.mkdir(parents=True, exist_ok=True)
    line_manifest_path = args.output_dir / f"{split}_lines.csv"
    page_manifest_path = args.output_dir / f"{split}_pages.csv"

    image_cache: OrderedDict[int, Image.Image] = OrderedDict()
    path_cache: dict[int, Path | None] = {}
    pages: dict[int, list[dict]] = defaultdict(list)
    line_count = 0

    sorted_groups = sorted(
        groups.items(),
        key=lambda item: (
            item[0][0],
            min(part["y_center"] for part in item[1]),
            min(part["x_center"] for part in item[1]),
        ),
    )
    if args.max_groups_per_split is not None:
        sorted_groups = sorted_groups[: args.max_groups_per_split]

    with line_manifest_path.open("w", encoding="utf-8", newline="") as line_target:
        writer = csv.writer(line_target)
        writer.writerow(["image_path", "text", "page_id", "line_number", "x1", "y1", "x2", "y2", "source_image"])
        for (image_id, group_id), parts in sorted_groups:
            image = image_info[image_id]
            if image_id not in path_cache:
                path_cache[image_id] = find_image(args.source_dir / "images", image["file_name"])
            image_path = path_cache[image_id]
            if image_path is None:
                continue

            sorted_parts = sorted(parts, key=lambda part: (part["x_center"], part["y_center"]))
            text = normalize_line_text([part["text"] for part in sorted_parts])
            if not text:
                continue
            x1, y1, x2, y2 = union_bbox([part["bbox"] for part in sorted_parts])
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue

            if image_id not in image_cache:
                image_cache[image_id] = Image.open(image_path).convert("RGB")
                if len(image_cache) > 4:
                    _, old_image = image_cache.popitem(last=False)
                    old_image.close()
            else:
                image_cache.move_to_end(image_id)

            page_id = Path(image["file_name"]).stem
            line_number = len(pages[image_id]) + 1
            line_filename = f"{page_id}_line_{line_number:03d}.jpg"
            line_path = output_lines_dir / line_filename
            image_cache[image_id].crop((x1, y1, x2, y2)).save(line_path, quality=95)
            relative_line_path = line_path.as_posix()
            writer.writerow([relative_line_path, text, page_id, line_number, x1, y1, x2, y2, image_path.as_posix()])
            pages[image_id].append(
                {
                    "text": text,
                    "line_number": line_number,
                    "bbox": (x1, y1, x2, y2),
                    "source_image": image_path.as_posix(),
                }
            )
            line_count += 1

    with page_manifest_path.open("w", encoding="utf-8", newline="") as page_target:
        writer = csv.writer(page_target)
        writer.writerow(["image_path", "text_path", "page_id"])
        for image_id, lines in sorted(pages.items(), key=lambda item: image_info[item[0]]["file_name"]):
            image = image_info[image_id]
            page_id = Path(image["file_name"]).stem
            lines = sorted(lines, key=lambda line: (line["bbox"][1], line["bbox"][0]))
            text_path = output_texts_dir / f"{page_id}.txt"
            text_path.write_text("\n".join(line["text"] for line in lines), encoding="utf-8")
            writer.writerow([lines[0]["source_image"], text_path.as_posix(), page_id])

    for image in image_cache.values():
        image.close()
    return line_count, len(pages)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        line_count, page_count = prepare_split(args, split)
        print(f"{split}: {line_count} lines, {page_count} pages")


if __name__ == "__main__":
    main()
