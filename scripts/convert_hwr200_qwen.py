import argparse
import csv
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path


PAGE_PROMPT = (
    "Recognize all Russian handwritten text on this page. "
    "Preserve the reading order and line breaks. Return only the final text."
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
ARCHIVE_PATTERN = re.compile(r"hwr200_(\d+)_(\d+)\.zip$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build page-level Qwen SFT data from single-page HWR200 documents."
    )
    parser.add_argument("--source-dir", type=Path, default=Path("data/training/hwr200_hf"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/training/hwr200_qwen_pages"))
    parser.add_argument("--conditions", nargs="+", default=["Сканы", "ФотоСветлое", "ФотоТемное"])
    parser.add_argument("--train-authors", type=int, default=160)
    parser.add_argument("--val-authors", type=int, default=20)
    parser.add_argument("--max-examples-per-split", type=int)
    return parser.parse_args()


def build_message(image_path: str, answer: str) -> dict:
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": PAGE_PROMPT},
                ],
            },
            {"role": "assistant", "content": answer},
        ]
    }


def archive_for_author(source_dir: Path, author_id: int) -> Path:
    for archive in source_dir.glob("hwr200_*.zip"):
        match = ARCHIVE_PATTERN.match(archive.name)
        if match and int(match.group(1)) <= author_id <= int(match.group(2)):
            return archive
    raise FileNotFoundError(f"No HWR200 image archive found for author {author_id}.")


def load_annotations(path: Path) -> dict[tuple[int, str], str]:
    annotations: dict[tuple[int, str], str] = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            match = re.fullmatch(r"annotations/(?:Originals|Reuse|FPR)/(\d+)/([^/]+)\.json", name)
            if not match:
                continue
            payload = json.loads(archive.read(name))
            text = "\n".join(line.rstrip() for line in payload.get("full_text", "").splitlines()).strip()
            if text:
                annotations[(int(match.group(1)), match.group(2))] = text
    return annotations


def collect_single_page_members(archive_path: Path, author_id: int, conditions: set[str]) -> dict[str, list[str]]:
    prefix = f"hw_dataset/{author_id}/"
    grouped: dict[str, list[str]] = defaultdict(list)
    with zipfile.ZipFile(archive_path) as archive:
        for name in archive.namelist():
            if not name.startswith(prefix):
                continue
            parts = name.split("/")
            if len(parts) != 5 or parts[3] not in conditions or Path(parts[-1]).suffix.lower() not in IMAGE_SUFFIXES:
                continue
            grouped[parts[2]].append(name)
    return grouped


def split_for_author(author_id: int, train_authors: int, val_authors: int) -> str | None:
    if author_id < train_authors:
        return "train"
    if author_id < train_authors + val_authors:
        return "val"
    return "test"


def convert(
    source_dir: Path,
    output_dir: Path,
    conditions: set[str],
    train_authors: int,
    val_authors: int,
    max_examples_per_split: int | None,
) -> dict[str, int]:
    annotations = load_annotations(source_dir / "annotations.zip")
    counters = {"train": 0, "val": 0, "test": 0, "skipped_multipage": 0, "skipped_unannotated": 0}
    writers = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    for split in ("train", "val", "test"):
        writers[split] = (output_dir / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n")

    try:
        author_ids = sorted({author_id for author_id, _ in annotations})
        for author_id in author_ids:
            split = split_for_author(author_id, train_authors, val_authors)
            if max_examples_per_split is not None and counters[split] >= max_examples_per_split:
                continue
            groups = collect_single_page_members(archive_for_author(source_dir, author_id), author_id, conditions)
            for document_name, members in sorted(groups.items()):
                text = annotations.get((author_id, document_name))
                if not text:
                    counters["skipped_unannotated"] += 1
                    continue
                if len(members) != len(conditions):
                    counters["skipped_multipage"] += 1
                    continue
                for member in sorted(members):
                    if max_examples_per_split is not None and counters[split] >= max_examples_per_split:
                        break
                    condition = member.split("/")[3]
                    suffix = Path(member).suffix.lower()
                    image_name = f"author_{author_id}_{document_name}_{condition}_{Path(member).stem}{suffix}"
                    image_path = images_dir / split / image_name
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    with zipfile.ZipFile(archive_for_author(source_dir, author_id)) as archive:
                        image_path.write_bytes(archive.read(member))
                    writers[split].write(json.dumps(build_message(image_path.as_posix(), text), ensure_ascii=False) + "\n")
                    counters[split] += 1
    finally:
        for writer in writers.values():
            writer.close()

    with (output_dir / "conversion_stats.json").open("w", encoding="utf-8") as stream:
        json.dump(counters, stream, ensure_ascii=False, indent=2)
    with (output_dir / "README.md").open("w", encoding="utf-8") as stream:
        stream.write(
            "# HWR200 Qwen page data\n\n"
            "Only documents with exactly one image per requested condition are included. "
            "This prevents assigning a multi-page document transcription to a single page.\n"
        )
    return counters


def main() -> None:
    args = parse_args()
    stats = convert(
        args.source_dir,
        args.output_dir,
        set(args.conditions),
        args.train_authors,
        args.val_authors,
        args.max_examples_per_split,
    )
    print(" ".join(f"{key}={value}" for key, value in stats.items()))


if __name__ == "__main__":
    main()
