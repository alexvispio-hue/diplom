import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_hwr200_full_qwen import normalize_alignment_text, read_jsonl


MAX_REASONABLE_LINE_COUNT = 80


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the complete page-level HWR200 Qwen dataset.")
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path("data/training/hwr200_page_alignment/inventory.jsonl"),
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data/training/hwr200_page_alignment/page_labels.jsonl"),
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/training/hwr200_qwen_pages_full"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/training/hwr200_qwen_pages_full/validation_report.json"),
    )
    parser.add_argument("--max-authors", type=int)
    parser.add_argument("--author-ids", type=int, nargs="+")
    return parser.parse_args()


def assistant_text(example: dict) -> str:
    return example["messages"][1]["content"]


def image_path(example: dict) -> Path:
    for part in example["messages"][0]["content"]:
        if part.get("type") == "image":
            return Path(part["image"])
    raise ValueError("Qwen example has no image part.")


def validate(
    inventory_path: Path,
    labels_path: Path,
    dataset_dir: Path,
    max_authors: int | None = None,
    author_ids: list[int] | None = None,
) -> dict:
    inventory = read_jsonl(inventory_path)
    if author_ids is not None:
        allowed_authors = set(author_ids)
        inventory = [row for row in inventory if row["author_id"] in allowed_authors]
    if max_authors is not None:
        allowed_authors = set(sorted({row["author_id"] for row in inventory})[:max_authors])
        inventory = [row for row in inventory if row["author_id"] in allowed_authors]
    labels = read_jsonl(labels_path)
    metadata = read_jsonl(dataset_dir / "page_metadata.jsonl")
    examples_by_split = {
        split: read_jsonl(dataset_dir / f"{split}.jsonl")
        for split in ("train", "val", "test")
    }

    expected_pages = sum(len(row["conditions"]["Сканы"]) for row in inventory if row["has_annotation"])
    expected_images = expected_pages * 3
    label_keys = {
        (row["author_id"], row["document_name"], row["page_index"])
        for row in labels
    }
    expected_label_keys = {
        (row["author_id"], row["document_name"], page_index)
        for row in inventory
        if row["has_annotation"]
        for page_index in range(1, len(row["conditions"]["Сканы"]) + 1)
    }

    labels_by_document = defaultdict(list)
    for row in labels:
        labels_by_document[(row["author_id"], row["document_name"])].append(row)
    reconstructed_mismatches = []
    for document in inventory:
        key = (document["author_id"], document["document_name"])
        pages = sorted(labels_by_document.get(key, []), key=lambda row: row["page_index"])
        reconstructed = " ".join(page["target_text"] for page in pages)
        if normalize_alignment_text(reconstructed) != normalize_alignment_text(document["full_text"]):
            reconstructed_mismatches.append(key)

    examples = [example for split in ("train", "val", "test") for example in examples_by_split[split]]
    paths = [image_path(example) for example in examples]
    missing_images = [path.as_posix() for path in paths if not path.is_file()]
    empty_targets = sum(not assistant_text(example).strip() for example in examples)
    metadata_by_path = {Path(row["image_path"]).as_posix(): row for row in metadata}
    variant_targets = defaultdict(set)
    missing_metadata = []
    for example, path in zip(examples, paths):
        row = metadata_by_path.get(path.as_posix())
        if not row:
            missing_metadata.append(path.as_posix())
            continue
        key = (row["author_id"], row["document_name"], row["page_index"])
        variant_targets[key].add(assistant_text(example))
    variant_mismatches = [key for key, values in variant_targets.items() if len(values) != 1]

    split_authors = {
        split: {row["author_id"] for row in metadata if row["split"] == split}
        for split in ("train", "val", "test")
    }
    author_overlap = bool(
        split_authors["train"] & split_authors["val"]
        or split_authors["train"] & split_authors["test"]
        or split_authors["val"] & split_authors["test"]
    )

    checks = {
        "all_documents_annotated": all(row["has_annotation"] for row in inventory),
        "all_condition_counts_match": all(row["condition_page_counts_match"] for row in inventory),
        "all_page_labels_present": label_keys == expected_label_keys,
        "all_documents_reconstruct_reference": not reconstructed_mismatches,
        "all_three_variants_exported": len(examples) == expected_images,
        "all_images_exist": not missing_images,
        "all_targets_nonempty": empty_targets == 0,
        "all_examples_have_metadata": not missing_metadata and len(metadata) == len(examples),
        "all_variants_share_target": not variant_mismatches and len(variant_targets) == expected_pages,
        "author_splits_disjoint": not author_overlap,
        "image_paths_unique": len(paths) == len({path.as_posix() for path in paths}),
        "no_ocr_draft_errors": all(not row.get("error") for row in labels),
        "reasonable_line_counts": all(
            1 <= int(row.get("line_count", 0)) <= MAX_REASONABLE_LINE_COUNT
            for row in labels
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "documents": len(inventory),
        "pages": len(labels),
        "expected_pages": expected_pages,
        "images": len(examples),
        "expected_images": expected_images,
        "split_examples": {split: len(rows) for split, rows in examples_by_split.items()},
        "review_pages": sum(row.get("needs_review", False) for row in labels),
        "maximum_line_count": max((int(row.get("line_count", 0)) for row in labels), default=0),
        "minimum_line_count": min((int(row.get("line_count", 0)) for row in labels), default=0),
        "line_count_outliers": [
            [row["author_id"], row["document_name"], row["page_index"], row.get("line_count", 0)]
            for row in labels
            if not 1 <= int(row.get("line_count", 0)) <= MAX_REASONABLE_LINE_COUNT
        ][:20],
        "reconstructed_mismatches": reconstructed_mismatches[:20],
        "missing_images": missing_images[:20],
        "variant_mismatches": variant_mismatches[:20],
    }


def main() -> None:
    args = parse_args()
    report = validate(args.inventory, args.labels, args.dataset_dir, args.max_authors, args.author_ids)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
