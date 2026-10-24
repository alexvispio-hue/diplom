import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from app.image_processing.page_segmentation import PageLineSegmenter


BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class SegmentationEvaluationItem:
    page_id: str
    image_path: Path
    reference_lines: int
    predicted_lines: int
    matched_lines: int
    missed_lines: int
    extra_lines: int
    merged_lines: int
    split_lines: int
    found_line_rate: float
    mean_iou: float
    mean_coordinate_deviation: float


@dataclass(frozen=True)
class SegmentationEvaluationSummary:
    items: list[SegmentationEvaluationItem]
    total_reference_lines: int
    total_predicted_lines: int
    total_matched_lines: int
    average_found_line_rate: float
    average_iou: float
    average_coordinate_deviation: float


class SegmentationEvaluationService:
    def __init__(self, page_segmenter: PageLineSegmenter | None = None) -> None:
        self.page_segmenter = page_segmenter or PageLineSegmenter()

    def evaluate_lines_manifest(
        self,
        lines_csv_path: Path,
        output_csv_path: Path,
        processed_dir: Path,
        iou_threshold: float = 0.3,
        limit_pages: int | None = None,
    ) -> SegmentationEvaluationSummary:
        pages = self._load_reference_pages(lines_csv_path)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)
        items: list[SegmentationEvaluationItem] = []

        for page_index, (page_id, page) in enumerate(pages.items()):
            if limit_pages is not None and page_index >= limit_pages:
                break
            segmentation = self.page_segmenter.segment(
                page["image_path"],
                processed_dir / page_id / "lines",
                processed_dir / page_id / "annotated.png",
            )
            items.append(
                self._evaluate_page(
                    page_id=page_id,
                    image_path=page["image_path"],
                    reference_boxes=page["boxes"],
                    predicted_boxes=[line.bbox for line in segmentation.lines],
                    iou_threshold=iou_threshold,
                )
            )

        self._write_results(output_csv_path, items)
        return self._build_summary(items)

    @staticmethod
    def _load_reference_pages(lines_csv_path: Path) -> dict[str, dict]:
        pages: dict[str, dict] = {}
        with lines_csv_path.open("r", encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            for row in reader:
                page_id = row["page_id"]
                if page_id not in pages:
                    image_path = Path(row.get("source_image") or row["image_path"])
                    if not image_path.is_absolute():
                        image_path = image_path.resolve()
                    pages[page_id] = {"image_path": image_path, "boxes": []}
                pages[page_id]["boxes"].append(
                    (int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"]))
                )
        return pages

    @classmethod
    def _evaluate_page(
        cls,
        page_id: str,
        image_path: Path,
        reference_boxes: list[BBox],
        predicted_boxes: list[BBox],
        iou_threshold: float,
    ) -> SegmentationEvaluationItem:
        matches = cls._match_boxes(reference_boxes, predicted_boxes, iou_threshold)
        matched_reference = {reference_index for reference_index, _, _ in matches}
        matched_predicted = {predicted_index for _, predicted_index, _ in matches}
        ious = [iou for _, _, iou in matches]
        deviations = [
            cls._coordinate_deviation(reference_boxes[reference_index], predicted_boxes[predicted_index])
            for reference_index, predicted_index, _ in matches
        ]
        merged_lines = cls._count_many_to_one(reference_boxes, predicted_boxes, iou_threshold)
        split_lines = cls._count_many_to_one(predicted_boxes, reference_boxes, iou_threshold)

        reference_count = len(reference_boxes)
        predicted_count = len(predicted_boxes)
        matched_count = len(matches)
        return SegmentationEvaluationItem(
            page_id=page_id,
            image_path=image_path,
            reference_lines=reference_count,
            predicted_lines=predicted_count,
            matched_lines=matched_count,
            missed_lines=reference_count - len(matched_reference),
            extra_lines=predicted_count - len(matched_predicted),
            merged_lines=merged_lines,
            split_lines=split_lines,
            found_line_rate=matched_count / reference_count if reference_count else 0.0,
            mean_iou=sum(ious) / len(ious) if ious else 0.0,
            mean_coordinate_deviation=sum(deviations) / len(deviations) if deviations else 0.0,
        )

    @staticmethod
    def _match_boxes(reference_boxes: list[BBox], predicted_boxes: list[BBox], iou_threshold: float) -> list[tuple[int, int, float]]:
        candidates = []
        for reference_index, reference_box in enumerate(reference_boxes):
            for predicted_index, predicted_box in enumerate(predicted_boxes):
                score = box_iou(reference_box, predicted_box)
                if score >= iou_threshold:
                    candidates.append((score, reference_index, predicted_index))

        matches = []
        used_reference = set()
        used_predicted = set()
        for score, reference_index, predicted_index in sorted(candidates, reverse=True):
            if reference_index in used_reference or predicted_index in used_predicted:
                continue
            used_reference.add(reference_index)
            used_predicted.add(predicted_index)
            matches.append((reference_index, predicted_index, score))
        return matches

    @staticmethod
    def _count_many_to_one(source_boxes: list[BBox], target_boxes: list[BBox], iou_threshold: float) -> int:
        count = 0
        for target_box in target_boxes:
            overlaps = sum(1 for source_box in source_boxes if box_iou(source_box, target_box) >= iou_threshold)
            if overlaps > 1:
                count += 1
        return count

    @staticmethod
    def _coordinate_deviation(reference_box: BBox, predicted_box: BBox) -> float:
        return sum(abs(reference_box[index] - predicted_box[index]) for index in range(4)) / 4

    @staticmethod
    def _write_results(output_csv_path: Path, items: list[SegmentationEvaluationItem]) -> None:
        with output_csv_path.open("w", encoding="utf-8", newline="") as target:
            fieldnames = [
                "page_id",
                "image_path",
                "reference_lines",
                "predicted_lines",
                "matched_lines",
                "missed_lines",
                "extra_lines",
                "merged_lines",
                "split_lines",
                "found_line_rate",
                "mean_iou",
                "mean_coordinate_deviation",
            ]
            writer = csv.DictWriter(target, fieldnames=fieldnames)
            writer.writeheader()
            for item in items:
                writer.writerow(
                    {
                        "page_id": item.page_id,
                        "image_path": str(item.image_path),
                        "reference_lines": item.reference_lines,
                        "predicted_lines": item.predicted_lines,
                        "matched_lines": item.matched_lines,
                        "missed_lines": item.missed_lines,
                        "extra_lines": item.extra_lines,
                        "merged_lines": item.merged_lines,
                        "split_lines": item.split_lines,
                        "found_line_rate": f"{item.found_line_rate:.6f}",
                        "mean_iou": f"{item.mean_iou:.6f}",
                        "mean_coordinate_deviation": f"{item.mean_coordinate_deviation:.2f}",
                    }
                )

    @staticmethod
    def _build_summary(items: list[SegmentationEvaluationItem]) -> SegmentationEvaluationSummary:
        total_reference = sum(item.reference_lines for item in items)
        total_predicted = sum(item.predicted_lines for item in items)
        total_matched = sum(item.matched_lines for item in items)
        return SegmentationEvaluationSummary(
            items=items,
            total_reference_lines=total_reference,
            total_predicted_lines=total_predicted,
            total_matched_lines=total_matched,
            average_found_line_rate=total_matched / total_reference if total_reference else 0.0,
            average_iou=_average([item.mean_iou for item in items]),
            average_coordinate_deviation=_average([item.mean_coordinate_deviation for item in items]),
        )


def box_iou(left: BBox, right: BBox) -> float:
    x_left = max(left[0], right[0])
    y_top = max(left[1], right[1])
    x_right = min(left[2], right[2])
    y_bottom = min(left[3], right[3])
    intersection_width = max(0, x_right - x_left)
    intersection_height = max(0, y_bottom - y_top)
    intersection = intersection_width * intersection_height
    if intersection == 0:
        return 0.0
    left_area = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
    right_area = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
