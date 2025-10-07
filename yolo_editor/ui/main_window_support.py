"""Support utilities used by the main window presenter."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

from PySide6.QtCore import QObject, Signal

from ..core.dataset_resolver import DatasetModel, resolve_dataset
from ..core.yolo_io import Box, imread_unicode, labels_for_image, read_yolo_txt


_MIN_BOX_NORM = 1e-4
_MIN_BOX_PIXELS = 2


def sanitize_boxes_by_size(boxes: List[Box], img_w: int, img_h: int) -> tuple[List[Box], bool]:
    """Ensure boxes remain within valid normalized bounds for a given image size."""
    if img_w <= 0 or img_h <= 0:
        return list(boxes), False

    min_w = max(_MIN_BOX_PIXELS / img_w, _MIN_BOX_NORM)
    min_h = max(_MIN_BOX_PIXELS / img_h, _MIN_BOX_NORM)
    sanitized: List[Box] = []
    changed = False
    for box in boxes:
        new_w = max(min(box.w, 1.0), min_w)
        new_h = max(min(box.h, 1.0), min_h)
        half_w = min(new_w / 2, 0.5)
        half_h = min(new_h / 2, 0.5)
        if new_w >= 1.0:
            cx = 0.5
        else:
            cx = min(max(box.cx, half_w), 1.0 - half_w)
        if new_h >= 1.0:
            cy = 0.5
        else:
            cy = min(max(box.cy, half_h), 1.0 - half_h)
        if (
            abs(new_w - box.w) > 1e-6
            or abs(new_h - box.h) > 1e-6
            or abs(cx - box.cx) > 1e-6
            or abs(cy - box.cy) > 1e-6
        ):
            changed = True
        sanitized.append(Box(box.cls, cx, cy, new_w, new_h))
    return sanitized, changed



class MergeDatasetStatsWorker(QObject):
    """Background worker that calculates per-class stats for merge datasets."""

    finished = Signal(str, str, list)
    failed = Signal(str, str, str)

    def __init__(self, dataset_id: str, dataset_name: str, dataset_model, image_cache: dict[Path, list[Box]]):
        super().__init__()
        self._dataset_id = dataset_id
        self._dataset_name = dataset_name
        self._dataset_model = dataset_model
        self._image_cache = dict(image_cache)

    def run(self):
        try:
            per_imgs = defaultdict(int)
            per_boxes = defaultdict(int)
            name_map = list(self._dataset_model.names or [])
            for split in self._dataset_model.ordered_splits():
                info = self._dataset_model.splits.get(split)
                if not info:
                    continue
                labels_dir = info.get("labels_dir")
                images_dir = info.get("images_dir")
                for img_path in info.get("images", []):
                    boxes = self._image_cache.get(img_path)
                    if boxes is None:
                        boxes = read_yolo_txt(labels_for_image(img_path, labels_dir, images_dir))
                        self._image_cache[img_path] = boxes
                    seen: set[int] = set()
                    for box in boxes:
                        per_boxes[box.cls] += 1
                        seen.add(box.cls)
                    for cls in seen:
                        per_imgs[cls] += 1
            items = []
            all_ids = sorted(set(per_imgs.keys()) | set(per_boxes.keys()))
            for cid in all_ids:
                label = name_map[cid] if 0 <= cid < len(name_map) else str(cid)
                items.append({
                    "class_id": cid,
                    "class_name": label,
                    "images": per_imgs.get(cid, 0),
                    "boxes": per_boxes.get(cid, 0),
                })
            self.finished.emit(self._dataset_id, self._dataset_name, items)
        except Exception as exc:  # pragma: no cover - defensive
            self.failed.emit(self._dataset_id, self._dataset_name, str(exc))
class ManifestWriter:
    """Incrementally writes export manifest entries and metadata."""

    def __init__(self, dest_root: Path, dry_run: bool):
        self._dest_root = dest_root
        self._dry_run = dry_run
        self._count = 0
        self.entries_path = dest_root / ("dry_run_manifest.jsonl" if dry_run else "export_manifest.jsonl")
        self.meta_path = dest_root / ("dry_run_manifest_meta.json" if dry_run else "export_manifest_meta.json")
        self._entries_file = self.entries_path.open("w", encoding="utf-8")
        self._meta = {
            "dry_run": dry_run,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    def append(self, record: dict):
        self._entries_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._count += 1

    def finalize(self):
        self._entries_file.close()
        self._meta["entries"] = self._count
        self.meta_path.write_text(json.dumps(self._meta, indent=2, ensure_ascii=False), encoding="utf-8")
        return self.entries_path

    @property
    def count(self) -> int:
        return self._count

    def abort(self):
        try:
            self._entries_file.close()
        finally:
            if self.entries_path.exists():
                self.entries_path.unlink()
            if self.meta_path.exists():
                self.meta_path.unlink()


class DatasetLoaderWorker(QObject):
    """Background worker that resolves dataset metadata."""

    finished = Signal(str, object, bool)
    failed = Signal(str, str)

    def __init__(
        self,
        source: Path,
        dataset_name: str,
        failure_title: str,
        missing_split_message: str,
        merge_mode: bool,
    ):
        super().__init__()
        self._source = source
        self._dataset_name = dataset_name
        self._failure_title = failure_title
        self._missing_split_message = missing_split_message
        self._merge_mode = merge_mode

    def run(self):
        try:
            dataset = resolve_dataset(self._source)
        except Exception as exc:  # pragma: no cover - defensive guard
            self.failed.emit(self._failure_title, f"Failed to load dataset:\n{exc}")
            return
        if not dataset.splits:
            self.failed.emit(self._failure_title, self._missing_split_message)
            return
        self.finished.emit(self._dataset_name, dataset, self._merge_mode)


class StatsWorker(QObject):
    """Background worker that calculates dataset statistics for the current image list."""

    finished = Signal(dict, dict, int, bool, dict, list)
    failed = Signal(str)
    progress = Signal(int, int)

    def __init__(
        self,
        images: List[Path],
        labels_dir: Optional[Path],
        images_dir: Optional[Path],
        image_sizes: Dict[Path, Tuple[int, int]],
    ):
        super().__init__()
        self._images = list(images)
        self._labels_dir = labels_dir
        self._images_dir = images_dir
        self._cancelled = False
        self._image_sizes = dict(image_sizes)

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            total = len(self._images)
            per_imgs = defaultdict(int)
            per_boxes = defaultdict(int)
            max_cls = -1
            folder_data: dict[str, dict] = {}
            images_no_labels = []
            for idx, img_path in enumerate(self._images, start=1):
                if self._cancelled:
                    self.finished.emit({}, {}, max_cls, True, {}, images_no_labels)
                    return
                boxes = read_yolo_txt(labels_for_image(img_path, self._labels_dir, self._images_dir))
                if not boxes:
                    images_no_labels.append(img_path)
                size = self._image_sizes.get(img_path)
                if size is None:
                    img = imread_unicode(img_path)
                    if img is not None:
                        size = (img.shape[1], img.shape[0])
                        self._image_sizes[img_path] = size
                if size:
                    boxes, _ = sanitize_boxes_by_size(boxes, size[0], size[1])
                seen: set[int] = set()
                rel_folder = "."
                if self._images_dir:
                    try:
                        rel = img_path.relative_to(self._images_dir)
                        rel_folder = rel.parent.as_posix() or "."
                    except ValueError:
                        rel_folder = img_path.parent.as_posix()
                folder_entry = folder_data.setdefault(rel_folder, {"images": 0, "per_class": defaultdict(int)})
                folder_entry["images"] += 1
                for box in boxes:
                    per_boxes[box.cls] += 1
                    seen.add(box.cls)
                    folder_entry["per_class"][box.cls] += 1
                    if box.cls > max_cls:
                        max_cls = box.cls
                for cls in seen:
                    per_imgs[cls] += 1
                self.progress.emit(idx, total)
            if self._cancelled:
                self.finished.emit({}, {}, max_cls, True, {}, images_no_labels)
                return
            folder_payload = {}
            for folder, info in folder_data.items():
                folder_payload[folder] = {
                    "images": info["images"],
                    "per_class": dict(info["per_class"]),
                }
            self.finished.emit(dict(per_imgs), dict(per_boxes), max_cls, False, folder_payload, images_no_labels)
        except Exception as exc:  # pragma: no cover - defensive guard
            self.failed.emit(str(exc))
