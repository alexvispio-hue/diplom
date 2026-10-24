import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable


DEFAULT_TEXT_CATEGORIES = ("pupil_text", "pupil_comment", "teacher_comment")
PAGE_PROMPT = (
    "Recognize all Russian handwritten text on this notebook page. "
    "Preserve the reading order and line breaks. Return only the final text."
)
LAYOUT_PROMPT = (
    "Find all handwritten Russian text lines on this notebook page. "
    "Return only a valid JSON array. Each item must contain line_number, text, and box "
    "where box is [x1, y1, x2, y2] in image pixel coordinates."
)
LINE_PROMPT = "Recognize Russian handwritten text in the image. Return only the text."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert school_notebooks_RU COCO annotations into Qwen page/layout SFT data."
    )
    parser.add_argument("--source-dir", type=Path, default=Path("data/training/schooldataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/training/school_qwen_pages"))
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--categories", nargs="+", default=list(DEFAULT_TEXT_CATEGORIES))
    parser.add_argument("--padding", type=int, default=0)
    parser.add_argument("--max-pages-per-split", type=int)
    parser.add_argument("--include-line-examples", action="store_true")
    return parser.parse_args()


def find_image(images_dir: Path, filename: str) -> Path | None:
    direct = images_dir / filename
    if direct.exists():
        return direct
    matches = list(images_dir.rglob(filename))
    return matches[0] if matches else None


def normalize_line_text(tokens: Iterable[str]) -> str:
    text = " ".join(token.strip() for token in tokens if token and token.strip())
    for punctuation in (".", ",", ":", ";", "!", "?", ")", "]", "}"):
        text = text.replace(f" {punctuation}", punctuation)
    for punctuation in ("(", "[", "{"):
        text = text.replace(f"{punctuation} ", punctuation)
    return " ".join(text.replace(" - ", "-").split())


def polygon_points(segmentation: list) -> list[tuple[float, float]]:
    if not segmentation:
        return []
    raw = segmentation[0]
    if raw and isinstance(raw[0], list):
        raw = raw[0]
    return [(float(raw[index]), float(raw[index + 1])) for index in range(0, len(raw), 2)]


def bbox_from_points(
    points: list[tuple[float, float]],
    *,
    width: int,
    height: int,
    padding: int,
) -> tuple[int, int, int, int]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (
        max(0, int(min(xs)) - padding),
        max(0, int(min(ys)) - padding),
        min(width, int(max(xs)) + padding),
        min(height, int(max(ys)) + padding),
    )


def union_bbox(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def build_message(image_path: str, prompt: str, answer: str) -> dict:
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            },
            {"role": "assistant", "content": answer},
        ]
    }


def load_coco(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_page_lines(data: dict, categories: set[str], padding: int) -> dict[int, list[dict]]:
    category_names = {category["id"]: category["name"] for category in data["categories"]}
    image_info = {image["id"]: image for image in data["images"]}
    grouped_parts: dict[tuple[int, int], list[dict]] = defaultdict(list)

    for annotation in data["annotations"]:
        category_name = category_names.get(annotation.get("category_id"))
        if category_name not in categories:
            continue
        text = (annotation.get("attributes", {}).get("translation") or "").strip()
        if not text or "group_id" not in annotation:
            continue
        image = image_info.get(annotation["image_id"])
        if not image:
            continue
        points = polygon_points(annotation.get("segmentation") or [])
        if not points:
            continue
        bbox = bbox_from_points(points, width=image["width"], height=image["height"], padding=padding)
        grouped_parts[(annotation["image_id"], annotation["group_id"])].append(
            {
                "text": text,
                "bbox": bbox,
                "polygon": [[round(x, 2), round(y, 2)] for x, y in points],
                "category": category_name,
                "x_center": (bbox[0] + bbox[2]) / 2,
                "y_center": (bbox[1] + bbox[3]) / 2,
            }
        )

    pages: dict[int, list[dict]] = defaultdict(list)
    for (image_id, _group_id), parts in grouped_parts.items():
        sorted_parts = sorted(parts, key=lambda part: (part["x_center"], part["y_center"]))
        text = normalize_line_text(part["text"] for part in sorted_parts)
        if not text:
            continue
        bbox = union_bbox([part["bbox"] for part in sorted_parts])
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        pages[image_id].append(
            {
                "text": text,
                "box": list(bbox),
                "polygon": [point for part in sorted_parts for point in part["polygon"]],
                "categories": sorted({part["category"] for part in sorted_parts}),
            }
        )

    for image_id, lines in pages.items():
        lines.sort(key=lambda line: (line["box"][1], line["box"][0]))
        for index, line in enumerate(lines, start=1):
            line["line_number"] = index
    return pages


def convert_split(
    *,
    source_dir: Path,
    output_dir: Path,
    split: str,
    categories: set[str],
    padding: int,
    max_pages: int | None,
    include_line_examples: bool,
) -> dict:
    data = load_coco(source_dir / f"annotations_{split}.json")
    image_info = {image["id"]: image for image in data["images"]}
    pages = collect_page_lines(data, categories, padding)

    split_pages = sorted(
        pages.items(),
        key=lambda item: image_info[item[0]]["file_name"],
    )
    if max_pages is not None:
        split_pages = split_pages[:max_pages]

    text_dir = output_dir / "pages_text" / split
    text_dir.mkdir(parents=True, exist_ok=True)
    sft_dir = output_dir / "qwen_sft"
    sft_dir.mkdir(parents=True, exist_ok=True)

    pages_csv = output_dir / f"{split}_pages.csv"
    lines_csv = output_dir / f"{split}_lines.csv"
    sft_jsonl = sft_dir / f"{split}.jsonl"

    image_missing = 0
    page_count = 0
    line_count = 0
    example_count = 0

    with pages_csv.open("w", encoding="utf-8", newline="") as pages_file, lines_csv.open(
        "w", encoding="utf-8", newline=""
    ) as lines_file, sft_jsonl.open("w", encoding="utf-8", newline="\n") as sft_file:
        page_writer = csv.writer(pages_file)
        line_writer = csv.writer(lines_file)
        page_writer.writerow(["image_path", "text_path", "page_id", "width", "height", "line_count"])
        line_writer.writerow(
            ["image_path", "page_id", "line_number", "text", "x1", "y1", "x2", "y2", "polygon_json", "categories"]
        )

        for image_id, lines in split_pages:
            image = image_info[image_id]
            image_path = find_image(source_dir / "images", image["file_name"])
            if image_path is None:
                image_missing += 1
                continue

            page_id = Path(image["file_name"]).stem
            page_text = "\n".join(line["text"] for line in lines)
            text_path = text_dir / f"{page_id}.txt"
            text_path.write_text(page_text, encoding="utf-8")
            image_path_text = image_path.as_posix()
            text_path_text = text_path.as_posix()
            page_writer.writerow([image_path_text, text_path_text, page_id, image["width"], image["height"], len(lines)])

            layout = []
            for line in lines:
                x1, y1, x2, y2 = line["box"]
                line_writer.writerow(
                    [
                        image_path_text,
                        page_id,
                        line["line_number"],
                        line["text"],
                        x1,
                        y1,
                        x2,
                        y2,
                        json.dumps(line["polygon"], ensure_ascii=False),
                        "|".join(line["categories"]),
                    ]
                )
                layout.append({"line_number": line["line_number"], "text": line["text"], "box": line["box"]})
                if include_line_examples:
                    sft_file.write(json.dumps(build_message(image_path_text, LINE_PROMPT, line["text"]), ensure_ascii=False) + "\n")
                    example_count += 1

            sft_file.write(json.dumps(build_message(image_path_text, PAGE_PROMPT, page_text), ensure_ascii=False) + "\n")
            sft_file.write(
                json.dumps(build_message(image_path_text, LAYOUT_PROMPT, json.dumps(layout, ensure_ascii=False)), ensure_ascii=False)
                + "\n"
            )
            page_count += 1
            line_count += len(lines)
            example_count += 2

    return {
        "split": split,
        "pages": page_count,
        "lines": line_count,
        "examples": example_count,
        "missing_images": image_missing,
        "pages_csv": str(pages_csv),
        "lines_csv": str(lines_csv),
        "sft_jsonl": str(sft_jsonl),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats = [
        convert_split(
            source_dir=args.source_dir,
            output_dir=args.output_dir,
            split=split,
            categories=set(args.categories),
            padding=args.padding,
            max_pages=args.max_pages_per_split,
            include_line_examples=args.include_line_examples,
        )
        for split in args.splits
    ]
    stats_path = args.output_dir / "conversion_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in stats:
        print(
            f"{row['split']}: pages={row['pages']}, lines={row['lines']}, "
            f"examples={row['examples']}, missing_images={row['missing_images']}"
        )
    print(f"stats: {stats_path}")


if __name__ == "__main__":
    main()
