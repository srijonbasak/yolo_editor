from __future__ import annotations
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import shutil
import json
from datetime import datetime, timezone
import yaml
from PySide6.QtCore import Qt, QObject, Signal, QThread
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QStatusBar, QTabWidget, QWidget, QSplitter, QVBoxLayout,
    QHBoxLayout, QLabel, QComboBox, QPushButton, QTreeWidget, QTreeWidgetItem, QProgressDialog,
    QTableWidget, QTableWidgetItem, QHeaderView, QListWidget, QMessageBox, QInputDialog
)

# keep your existing image view
from .image_view import ImageView, Box as ViewBox

# NEW helpers
from ..core.dataset_resolver import resolve_dataset, DatasetModel
from ..core.yolo_io import Box, read_yolo_txt, write_yolo_txt, labels_for_image, imread_unicode

_MIN_BOX_NORM = 1e-4
_MIN_BOX_PIXELS = 2


def _sanitize_boxes_by_size(boxes: List[Box], img_w: int, img_h: int) -> tuple[List[Box], bool]:
    if img_w <= 0 or img_h <= 0:
        return list(boxes), False

    min_w = max(_MIN_BOX_PIXELS / img_w, _MIN_BOX_NORM)
    min_h = max(_MIN_BOX_PIXELS / img_h, _MIN_BOX_NORM)
    sanitized: List[Box] = []
    changed = False
    for b in boxes:
        new_w = max(min(b.w, 1.0), min_w)
        new_h = max(min(b.h, 1.0), min_h)
        half_w = min(new_w / 2, 0.5)
        half_h = min(new_h / 2, 0.5)
        if new_w >= 1.0:
            cx = 0.5
        else:
            cx = min(max(b.cx, half_w), 1.0 - half_w)
        if new_h >= 1.0:
            cy = 0.5
        else:
            cy = min(max(b.cy, half_h), 1.0 - half_h)
        if (abs(new_w - b.w) > 1e-6 or abs(new_h - b.h) > 1e-6 or abs(cx - b.cx) > 1e-6 or abs(cy - b.cy) > 1e-6):
            changed = True
        sanitized.append(Box(b.cls, cx, cy, new_w, new_h))
    return sanitized, changed


class ManifestWriter:
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

class _DatasetLoaderWorker(QObject):
    finished = Signal(str, object, bool)
    failed = Signal(str, str)

    def __init__(self, source: Path, dataset_name: str, failure_title: str, missing_split_message: str, merge_mode: bool):
        super().__init__()
        self._source = source
        self._dataset_name = dataset_name
        self._failure_title = failure_title
        self._missing_split_message = missing_split_message
        self._merge_mode = merge_mode

    def run(self):
        try:
            dm = resolve_dataset(self._source)
        except Exception as exc:
            self.failed.emit(self._failure_title, f"Failed to load dataset:\n{exc}")
            return
        if not dm.splits:
            self.failed.emit(self._failure_title, self._missing_split_message)
            return
        self.finished.emit(self._dataset_name, dm, self._merge_mode)


class _StatsWorker(QObject):
    finished = Signal(dict, dict, int, bool, dict, list)
    failed = Signal(str)
    progress = Signal(int, int)

    def __init__(self, images: List[Path], labels_dir: Path | None, images_dir: Path | None, image_sizes: Dict[Path, Tuple[int, int]]):
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
                    boxes, _ = _sanitize_boxes_by_size(boxes, size[0], size[1])
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
                for b in boxes:
                    per_boxes[b.cls] += 1
                    seen.add(b.cls)
                    folder_entry["per_class"][b.cls] += 1
                    if b.cls > max_cls:
                        max_cls = b.cls
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
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Editor - Offline")
        self.resize(1380, 860)
        self.setStatusBar(QStatusBar(self))

        self.dm: DatasetModel | None = None
        self.ds_name: str = "-"
        self.split: str | None = None
        self.images: List[Path] = []
        self.labels_dir: Path | None = None
        self.images_dir: Path | None = None
        self.names: List[str] = []
        self.idx: int = -1
        self._loader_thread: QThread | None = None
        self._loader_worker: _DatasetLoaderWorker | None = None
        self._loader_dialog: QProgressDialog | None = None
        self._label_cache: dict[Path, List[Box]] = {}
        self._image_class_sets: dict[Path, set[int]] = {}
        self._max_class_id: int = -1
        self._stats_thread: QThread | None = None
        self._stats_worker: _StatsWorker | None = None
        self._stats_dialog: QProgressDialog | None = None
        self._stats_pending: bool = False
        self._stats_active: bool = False
        self._image_sizes: Dict[Path, Tuple[int, int]] = {}
        self._adjustment_notices: set[Path] = set()
        self._merge_loaded_datasets: List[str] = []
        self._merge_datasets: Dict[str, DatasetModel] = {}

        self.tabs = QTabWidget(self)
        self.setCentralWidget(self.tabs)

        # --- Label Editor tab ---
        self.editor_root = QWidget()
        self._build_editor_ui(self.editor_root)
        self.tabs.addTab(self.editor_root, "Label Editor")

        # (optional) keep your Merge tab - only if imports are available
        try:
            from .merge_designer.canvas import MergeCanvas
            from .merge_designer.controller import MergeController
            from .merge_designer.palette import MergePalette
            self.merge_ctrl = MergeController()
            self.merge_canvas = MergeCanvas(self.merge_ctrl)
            self.merge_palette = MergePalette(
                on_spawn_dataset=self._spawn_dataset_node,
                on_spawn_target_class=self._spawn_target_node
            )
            self.merge_palette.populate([])
            self.merge_palette.requestLoadDataset.connect(self._load_dataset_for_merge)
            self.merge_palette.requestExportMerged.connect(self._export_merged_dataset)
            sp = QSplitter(Qt.Orientation.Horizontal)
            sp.addWidget(self.merge_palette); sp.addWidget(self.merge_canvas); sp.setStretchFactor(1, 1)
            self.tabs.addTab(sp, "Merge Designer")
        except Exception:
            pass  # don't block editor if merge ui has issues

        self._build_menu()

    # ---------------- UI ----------------

    def _build_editor_ui(self, parent: QWidget):
        splitter = QSplitter(Qt.Orientation.Horizontal, parent)

        # left
        left = QWidget(); lv = QVBoxLayout(left); lv.setContentsMargins(6,6,6,6)
        self.split_combo = QComboBox(); self.split_combo.currentTextChanged.connect(self._on_split_changed)
        self.file_tree = QTreeWidget(); self.file_tree.setHeaderLabels(["Image Files"])
        self.file_tree.itemClicked.connect(self._on_file_clicked)
        lv.addWidget(QLabel("Split:")); lv.addWidget(self.split_combo); lv.addWidget(self.file_tree, 1)

        # center
        center = QWidget(); cv = QVBoxLayout(center); cv.setContentsMargins(6,6,6,6)
        bar = QHBoxLayout()
        self.lbl_ds = QLabel("Dataset: -")
        self.class_combo = QComboBox(); self.class_combo.currentIndexChanged.connect(self._on_class_changed)
        self.btn_save = QPushButton("Save"); self.btn_save.clicked.connect(self._save_labels)
        bar.addWidget(self.lbl_ds, 1); bar.addWidget(QLabel("Class:")); bar.addWidget(self.class_combo); bar.addStretch(1); bar.addWidget(self.btn_save)
        self.view = ImageView()
        self.view.set_status_sink(lambda msg: self.statusBar().showMessage(msg, 3000))
        self.view.boxesChanged.connect(self._on_boxes_changed)
        self.view.requestPrev.connect(lambda: self._open_index(max(0, self.idx-1)))
        self.view.requestNext.connect(lambda: self._open_index(min(len(self.images)-1, self.idx+1)))
        cv.addLayout(bar); cv.addWidget(self.view, 1)

        # right
        right = QWidget(); rv = QVBoxLayout(right); rv.setContentsMargins(6,6,6,6)
        rv.addWidget(QLabel("Labels (YOLO)"))
        self.tbl = QTableWidget(0, 6)
        self.tbl.setHorizontalHeaderLabels(["name","id","cx","cy","w","h"])
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tbl.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        rv.addWidget(self.tbl, 3)
        rv.addWidget(QLabel("Dataset Stats"))
        self.stats = QListWidget(); rv.addWidget(self.stats, 2)

        splitter.addWidget(left); splitter.addWidget(center); splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)

        lay = QVBoxLayout(parent); lay.setContentsMargins(0,0,0,0); lay.addWidget(splitter)

    def _build_menu(self):
        m = self.menuBar().addMenu("&File")

        act_root = QAction("Open Dataset &Root...", self)
        act_yaml = QAction("Open Dataset &YAML...", self)
        act_quit = QAction("&Quit", self)

        act_root.triggered.connect(self._open_root)
        act_yaml.triggered.connect(self._open_yaml)
        act_quit.triggered.connect(self.close)

        m.addAction(act_root); m.addAction(act_yaml); m.addSeparator(); m.addAction(act_quit)

        e = self.menuBar().addMenu("&Edit")
        a_save = QAction("&Save Labels", self); a_save.setShortcut("S")
        a_prev = QAction("&Prev", self); a_prev.setShortcut(Qt.Key_Left)
        a_next = QAction("&Next", self); a_next.setShortcut(Qt.Key_Right)
        a_save.triggered.connect(self._save_labels)
        a_prev.triggered.connect(lambda: self._open_index(max(0, self.idx-1)))
        a_next.triggered.connect(lambda: self._open_index(min(len(self.images)-1, self.idx+1)))
        for a in (a_save, a_prev, a_next): e.addAction(a)

        tools = self.menuBar().addMenu("&Tools")
        act_diag = QAction("Show &Diagnostics...", self)
        act_diag.triggered.connect(self._show_diagnostics)
        tools.addAction(act_diag)

    # ---------------- Open handlers ----------------

    def _open_root(self):
        d = QFileDialog.getExistingDirectory(self, "Open Dataset Root")
        if not d:
            return
        path = Path(d)
        self._start_dataset_load(path, path.name, "Not a dataset", "Could not detect train/val/test/valid/eval in this folder.", merge_mode=False)

    def _open_yaml(self):
        f, _ = QFileDialog.getOpenFileName(self, "Open data.yaml", "", "YAML (*.yaml *.yml)")
        if not f:
            return
        path = Path(f)
        dataset_name = path.parent.name or path.stem
        self._start_dataset_load(path, dataset_name, "Invalid YAML", "Could not resolve any split paths from this YAML.", merge_mode=False)

    def _start_dataset_load(self, source: Path, dataset_name: str, failure_title: str, missing_split_message: str, merge_mode: bool = False):
        if self._loader_thread is not None:
            QMessageBox.information(self, "Loading", "A dataset load is already in progress. Please wait.")
            return
        self._loader_dialog = QProgressDialog("Loading dataset...", None, 0, 0, self)
        self._loader_dialog.setWindowModality(Qt.WindowModal)
        self._loader_dialog.setCancelButton(None)
        self._loader_dialog.setMinimumDuration(0)
        self._loader_dialog.setRange(0, 0)
        self._loader_dialog.setLabelText("Loading dataset...")
        self._loader_dialog.show()

        self._loader_thread = QThread(self)
        self._loader_worker = _DatasetLoaderWorker(source, dataset_name, failure_title, missing_split_message, merge_mode)
        self._loader_worker.moveToThread(self._loader_thread)
        self._loader_thread.started.connect(self._loader_worker.run)
        self._loader_worker.finished.connect(self._on_dataset_loaded)
        self._loader_worker.failed.connect(self._on_dataset_failed)
        self._loader_thread.start()

    def _on_dataset_loaded(self, dataset_name: str, model_obj, merge_mode: bool):
        self._teardown_loader_thread()
        if not isinstance(model_obj, DatasetModel):
            QMessageBox.warning(self, "Dataset load failed", "Unexpected dataset payload.")
            return
        self._load_dataset(dataset_name, model_obj, merge_mode)

    def _on_dataset_failed(self, title: str, message: str):
        self._teardown_loader_thread()
        QMessageBox.warning(self, title or "Dataset load failed", message or "Failed to load dataset.")

    def _teardown_loader_thread(self):
        if self._loader_dialog is not None:
            try:
                self._loader_dialog.close()
            finally:
                self._loader_dialog.deleteLater()
                self._loader_dialog = None
        if self._loader_thread is not None:
            self._loader_thread.quit()
            self._loader_thread.wait()
            self._loader_thread.deleteLater()
            self._loader_thread = None
        if self._loader_worker is not None:
            self._loader_worker.deleteLater()
            self._loader_worker = None
        self._cancel_stats_job()

    def _reset_cached_labels(self):
        self._label_cache.clear()
        self._image_class_sets.clear()
        self._max_class_id = -1

    def _ensure_image_size(self, img_path: Path, img=None) -> Optional[Tuple[int, int]]:
        size = self._image_sizes.get(img_path)
        if size:
            return size
        if img is None:
            img = imread_unicode(img_path)
        if img is None:
            return None
        size = (img.shape[1], img.shape[0])
        self._image_sizes[img_path] = size
        return size

    def _sanitize_boxes_for_image(self, img_path: Path, boxes: List[Box]) -> tuple[List[Box], bool]:
        size = self._ensure_image_size(img_path)
        if size is None:
            return list(boxes), False
        return _sanitize_boxes_by_size(boxes, size[0], size[1])

    def _notify_box_adjustment(self, img_path: Path):
        if img_path in self._adjustment_notices:
            return
        self._adjustment_notices.add(img_path)
        self.statusBar().showMessage(f"Adjusted extremely small box in {img_path.name}", 5000)

    def _get_or_load_boxes(self, img_path: Path, labels_dir: Path | None, images_dir: Path | None, *, notify: bool = False) -> List[Box]:
        boxes = self._label_cache.get(img_path)
        if boxes is None:
            boxes = read_yolo_txt(labels_for_image(img_path, labels_dir, images_dir))
            boxes, changed = self._sanitize_boxes_for_image(img_path, boxes)
            if changed and notify:
                self._notify_box_adjustment(img_path)
            self._label_cache[img_path] = boxes
            self._image_class_sets[img_path] = {b.cls for b in boxes}
        return boxes

    def _cancel_stats_job(self, restart: bool = False):
        if self._stats_worker:
            self._stats_pending = restart
            self._stats_worker.cancel()
            if self._stats_dialog:
                self._stats_dialog.setLabelText("Stopping class statistics...")
        else:
            if restart:
                self._stats_pending = False

    def _teardown_stats_thread(self):
        if self._stats_worker is not None:
            try:
                self._stats_worker.progress.disconnect(self._on_stats_progress)
            except (TypeError, RuntimeError):
                pass
            self._stats_worker.deleteLater()
            self._stats_worker = None
        if self._stats_dialog is not None:
            try:
                self._stats_dialog.close()
            finally:
                self._stats_dialog.deleteLater()
                self._stats_dialog = None
        if self._stats_thread is not None:
            if self._stats_thread.isRunning():
                self._stats_thread.quit()
                self._stats_thread.wait()
            self._stats_thread.deleteLater()
            self._stats_thread = None
        self._stats_active = False

    def _on_stats_dialog_canceled(self):
        self._cancel_stats_job(restart=False)

    def _on_stats_progress(self, current: int, total: int):
        if not self._stats_active:
            return
        try:
            if self._stats_dialog:
                if total:
                    self._stats_dialog.setMaximum(total)
                self._stats_dialog.setValue(current)
                self._stats_dialog.setLabelText(f"Computing class statistics... ({current}/{total})")
        except AttributeError:
            pass  # dialog was deleted

    def _on_stats_finished(self, per_imgs: dict, per_boxes: dict, max_cls: int, cancelled: bool, folder_stats: dict, images_no_labels: list):
        self._teardown_stats_thread()
        if cancelled:
            if self._stats_pending:
                self._stats_pending = False
                self._compute_stats_and_show()
            return
        self._stats_pending = False
        self.stats.clear()
        total_images = len(self.images)
        self.stats.addItem(f"Total images: {total_images}")
        total_boxes = sum(per_boxes.values())
        self.stats.addItem(f"Total boxes: {total_boxes}")
        images_with_labels = total_images - len(images_no_labels)
        self.stats.addItem(f"Images with labels: {images_with_labels}")
        self.stats.addItem(f"Images with no labels: {len(images_no_labels)}")

        ids = sorted(set(per_imgs.keys()) | set(per_boxes.keys()))
        all_classes = set(range(len(self.names))) if self.names else set()
        zero_classes = all_classes - set(ids)

        self.stats.addItem("Classes:")
        for cid in sorted(ids):
            nm = self.names[cid] if 0 <= cid < len(self.names) else str(cid)
            self.stats.addItem(f"  [{cid}] {nm}: {per_imgs.get(cid, 0)} imgs / {per_boxes.get(cid, 0)} boxes")
        for cid in sorted(zero_classes):
            nm = self.names[cid] if 0 <= cid < len(self.names) else str(cid)
            self.stats.addItem(f"  [{cid}] {nm}: 0 imgs / 0 boxes")

        if images_no_labels:
            self.stats.addItem("Images with no labels:")
            for img in images_no_labels[:10]:
                self.stats.addItem(f"  {img.name}")
            if len(images_no_labels) > 10:
                self.stats.addItem(f"  ...and {len(images_no_labels) - 10} more")
        if max_cls is not None and max_cls > self._max_class_id:
            self._max_class_id = max_cls
            self._ensure_class_combo_capacity(max_cls)
        elif ids:
            max_in_results = max(ids)
            if max_in_results > self._max_class_id:
                self._max_class_id = max_in_results
                self._ensure_class_combo_capacity(max_in_results)

        if folder_stats:
            self.stats.addItem("")
            self.stats.addItem("Folders:")
            for folder in sorted(folder_stats.keys()):
                info = folder_stats[folder]
                self.stats.addItem(f"  {folder}: {info['images']} images")
                per_class = info.get("per_class", {})
                if per_class:
                    class_parts = []
                    for cid, count in sorted(per_class.items()):
                        nm = self.names[cid] if 0 <= cid < len(self.names) else str(cid)
                        class_parts.append(f"[{cid}] {nm}={count}")
                    self.stats.addItem("    " + ", ".join(class_parts))

    def _on_stats_failed(self, message: str):
        self._teardown_stats_thread()
        self._stats_pending = False
        QMessageBox.warning(self, "Stats failed", f"Could not compute stats:\n{message}")

    def _init_class_combo(self, max_cls: int):
        self.class_combo.blockSignals(True)
        self.class_combo.clear()
        if self.names:
            target_len = max(len(self.names), (max_cls + 1) if max_cls >= 0 else len(self.names))
            while len(self.names) < target_len:
                self.names.append(f"class_{len(self.names)}")
            for idx, label in enumerate(self.names):
                display = (label or f"class_{idx}").strip() or f"class_{idx}"
                self.class_combo.addItem(display)
        else:
            fallback_max = max(max_cls, 19)
            for i in range(fallback_max + 1):
                self.class_combo.addItem(str(i))
        self.class_combo.blockSignals(False)
        if self.class_combo.count():
            self.class_combo.setCurrentIndex(0)
        self._max_class_id = max(self._max_class_id, self.class_combo.count() - 1)
        self.view.set_current_class(self.class_combo.currentIndex(), self.names)

    def _ensure_class_combo_capacity(self, cls_id: int):
        if cls_id < 0:
            return
        if self.names:
            updated = False
            while len(self.names) <= cls_id:
                self.names.append(f"class_{len(self.names)}")
                updated = True
            if updated:
                current_idx = self.class_combo.currentIndex()
                self.class_combo.blockSignals(True)
                self.class_combo.clear()
                for idx, label in enumerate(self.names):
                    display = (label or f"class_{idx}").strip() or f"class_{idx}"
                    self.class_combo.addItem(display)
                if current_idx >= 0 and self.class_combo.count():
                    self.class_combo.setCurrentIndex(min(current_idx, self.class_combo.count() - 1))
                self.class_combo.blockSignals(False)
                self.view.set_current_class(self.class_combo.currentIndex(), self.names)
        else:
            target = max(cls_id, 19)
            while self.class_combo.count() <= target:
                idx = self.class_combo.count()
                self.class_combo.addItem(str(idx))
            self.view.set_current_class(self.class_combo.currentIndex(), self.names)
        self._max_class_id = max(self._max_class_id, cls_id)

    def _load_dataset(self, name: str, dm: DatasetModel, merge_mode: bool = False):
        if merge_mode:
            # For merge mode, set the dataset as current for merge operations
            self.dm = dm
            self.ds_name = name
            self.names = list(dm.names) if dm.names else []
            self._reset_cached_labels()
            self.lbl_ds.setText(f"Dataset: {name}")
            # splits
            self.split_combo.blockSignals(True)
            self.split_combo.clear()
            for s in dm.ordered_splits(): self.split_combo.addItem(s)
            self.split_combo.blockSignals(False)
            if dm.ordered_splits():
                self._on_split_changed(dm.ordered_splits()[0])
            # Update merge palette
            self._merge_datasets[name] = dm
            if name not in self._merge_loaded_datasets:
                self._merge_loaded_datasets.append(name)
            self.merge_palette.populate(self._merge_loaded_datasets)
            self.merge_palette.update()
        else:
            # Normal loading
            self.dm = dm; self.ds_name = name
            self.lbl_ds.setText(f"Dataset: {name}")
            self.names = list(dm.names) if dm.names else []
            self._reset_cached_labels()
            initial_max = len(self.names) - 1 if self.names else -1
            self._max_class_id = initial_max
            self._init_class_combo(initial_max)

            # splits
            self.split_combo.blockSignals(True)
            self.split_combo.clear()
            for s in dm.ordered_splits(): self.split_combo.addItem(s)
            self.split_combo.blockSignals(False)
            
            # default split
            if dm.ordered_splits():
                self._on_split_changed(dm.ordered_splits()[0])
                
            # Update merge palette if available
            if hasattr(self, 'merge_palette'):
                if name not in self._merge_loaded_datasets:
                    self._merge_loaded_datasets.append(name)
                    self._merge_datasets[name] = dm
                self.merge_palette.populate(self._merge_loaded_datasets)

    # ---------------- Split / file tree ----------------

    def _on_split_changed(self, split: str):
        if not self.dm or split not in self.dm.splits: return
        sp = self.dm.splits[split]
        self.split = split
        self.images = sp["images"]
        self.labels_dir = sp["labels_dir"]
        self.images_dir = sp.get("images_dir")
        self._populate_file_tree()
        self._compute_stats_and_show()
        if self.images:
            self._open_index(0)

    def _populate_file_tree(self):
        self.file_tree.clear()
        root = QTreeWidgetItem(["(images)"])
        for p in self.images: QTreeWidgetItem(root, [p.name])
        self.file_tree.addTopLevelItem(root)
        self.file_tree.expandAll()

    def _on_file_clicked(self, item: QTreeWidgetItem):
        if not item.parent(): return
        idx = item.parent().indexOfChild(item)
        self._open_index(idx)

    # ---------------- Image / labels ----------------

    def _open_index(self, i: int):
        if i < 0 or i >= len(self.images): return
        img_path = self.images[i]
        img = imread_unicode(img_path)
        if img is None:
            self.statusBar().showMessage(f"Failed to load: {img_path.name}", 4000)
            return
        self._ensure_image_size(img_path, img)
        self.view.show_image_bgr(img_path, img)
        core_boxes = self._get_or_load_boxes(img_path, self.labels_dir, self.images_dir, notify=True)
        if core_boxes:
            new_max = max(b.cls for b in core_boxes)
            if new_max > self._max_class_id:
                self._max_class_id = new_max
                self._ensure_class_combo_capacity(new_max)
        self.view.clear_boxes()
        view_boxes = [ViewBox(cls=b.cls, cx=b.cx, cy=b.cy, w=b.w, h=b.h) for b in core_boxes]
        for vb in view_boxes:
            self.view.add_box_norm(vb)
        self._fill_table(view_boxes)
        self.idx = i
        self._highlight_tree_row(i)
        self.statusBar().showMessage(f"{img_path.name}  ({i+1}/{len(self.images)})", 4000)

    def _save_labels(self):
        if self.idx < 0 or not self.images: return
        img_path = self.images[self.idx]
        txt = labels_for_image(img_path, self.labels_dir, self.images_dir)
        view_boxes = self.view.get_boxes_as_norm()
        core_boxes = [Box(int(b.cls), b.cx, b.cy, b.w, b.h) for b in view_boxes]
        sanitized, changed = self._sanitize_boxes_for_image(img_path, core_boxes)
        if changed:
            self._notify_box_adjustment(img_path)
        write_yolo_txt(txt, sanitized)
        self._label_cache[img_path] = sanitized
        class_set = {b.cls for b in sanitized}
        self._image_class_sets[img_path] = class_set
        if class_set:
            new_max = max(class_set)
            if new_max > self._max_class_id:
                self._max_class_id = new_max
                self._ensure_class_combo_capacity(new_max)
        self.view.clear_boxes()
        sanitized_view = [ViewBox(cls=b.cls, cx=b.cx, cy=b.cy, w=b.w, h=b.h) for b in sanitized]
        for vb in sanitized_view:
            self.view.add_box_norm(vb)
        self._fill_table(sanitized_view)
        self.statusBar().showMessage(f"Saved: {txt}", 4000)
        self._compute_stats_and_show()

    def _fill_table(self, boxes: List[ViewBox]):
        self.tbl.setRowCount(0)
        for b in boxes:
            r = self.tbl.rowCount(); self.tbl.insertRow(r)
            name = self.names[b.cls] if 0 <= b.cls < len(self.names) else str(b.cls)
            self.tbl.setItem(r, 0, QTableWidgetItem(name))
            self.tbl.setItem(r, 1, QTableWidgetItem(str(b.cls)))
            self.tbl.setItem(r, 2, QTableWidgetItem(f"{b.cx:.4f}"))
            self.tbl.setItem(r, 3, QTableWidgetItem(f"{b.cy:.4f}"))
            self.tbl.setItem(r, 4, QTableWidgetItem(f"{b.w:.4f}"))
            self.tbl.setItem(r, 5, QTableWidgetItem(f"{b.h:.4f}"))

    def _on_boxes_changed(self):
        if self.idx < 0:
            return
        boxes = self.view.get_boxes_as_norm()
        view_boxes = [ViewBox(cls=b.cls, cx=b.cx, cy=b.cy, w=b.w, h=b.h) for b in boxes]
        self._fill_table(view_boxes)

    def _highlight_tree_row(self, idx: int):
        root = self.file_tree.topLevelItem(0)
        if not root: return
        for i in range(root.childCount()):
            ch = root.child(i); ch.setSelected(i == idx)
        self.file_tree.scrollToItem(root.child(idx))

    def _on_class_changed(self, idx: int):
        self.view.set_current_class(idx, self.names)
        nm = self.names[idx] if 0 <= idx < len(self.names) else str(idx)
        self.statusBar().showMessage(f"Current class -> {nm} [{idx}]", 3000)

    # ---------------- Stats ----------------

    def _compute_stats_and_show(self):
        if not self.images:
            self.stats.clear()
            self.stats.addItem("Total images: 0")
            return

        if self._stats_thread is not None:
            self._cancel_stats_job(restart=True)
            return

        self.stats.clear()
        self.stats.addItem(f"Total images: {len(self.images)}")
        self.stats.addItem("Computing class statistics...")

        self._stats_dialog = QProgressDialog("Computing class statistics...", "Cancel", 0, len(self.images), self)
        self._stats_dialog.setWindowModality(Qt.WindowModal)
        self._stats_dialog.setMinimumDuration(0)
        self._stats_dialog.setAutoReset(False)
        self._stats_dialog.setAutoClose(False)
        self._stats_dialog.setValue(0)
        self._stats_dialog.canceled.connect(self._on_stats_dialog_canceled)

        self._stats_active = True

        self._stats_pending = False
        self._stats_worker = _StatsWorker(self.images.copy(), self.labels_dir, self.images_dir, self._image_sizes)
        self._stats_thread = QThread(self)
        self._stats_worker.moveToThread(self._stats_thread)
        self._stats_thread.started.connect(self._stats_worker.run)
        self._stats_worker.progress.connect(self._on_stats_progress)
        self._stats_worker.finished.connect(self._on_stats_finished)
        self._stats_worker.failed.connect(self._on_stats_failed)
        self._stats_worker.finished.connect(lambda *_: self._stats_thread.quit())
        self._stats_worker.failed.connect(lambda *_: self._stats_thread.quit())
        self._stats_thread.finished.connect(self._stats_thread.deleteLater)
        self._stats_thread.start()

    def _show_diagnostics(self):
        if not self.dm:
            QMessageBox.information(self, "No dataset", "Load a dataset first.")
            return
        lines = [f"Dataset: {self.ds_name}"]
        if getattr(self.dm, "yaml_path", None):
            lines.append(f"YAML: {self.dm.yaml_path}")
        if not self.dm.splits:
            lines.append("No splits resolved.")
        else:
            for split in self.dm.ordered_splits():
                info = self.dm.splits[split]
                img_dir = info.get("images_dir")
                labels_dir = info.get("labels_dir")
                count = len(info.get("images", []))
                lines.append(f"[{split}] images: {img_dir}")
                label_text = labels_dir if labels_dir else '(next to images)'
                lines.append(f"    labels: {label_text}")
                lines.append(f"    image count: {count}")
        QMessageBox.information(self, "Dataset diagnostics", "\n".join(lines))

    # ---------------- Merge Designer ----------------
    
    def _load_dataset_for_merge(self):
        d = QFileDialog.getExistingDirectory(self, "Load Dataset for Merge")
        if not d:
            return
        path = Path(d)
        dataset_name = path.name
        if dataset_name in self._merge_datasets:
            QMessageBox.information(self, "Already loaded", f"Dataset '{dataset_name}' is already loaded for merge.")
            return
        self._start_dataset_load(path, dataset_name, "Load failed", "Could not load dataset for merge.", merge_mode=True)

    def _spawn_dataset_node(self, dataset_name: str):
        """Spawn a dataset node on the merge canvas"""
        if not hasattr(self, 'merge_canvas') or not hasattr(self, 'merge_ctrl'):
            return

        if dataset_name not in self._merge_loaded_datasets:
            QMessageBox.warning(self, "Dataset not loaded", f"Dataset '{dataset_name}' is not loaded for merge.")
            return

        # Get the specific dataset
        dm = self._merge_datasets.get(dataset_name)
        if dm is None:
            QMessageBox.warning(self, "Dataset not found", f"Dataset '{dataset_name}' data not available.")
            return

        # Create unique dataset name for multiple instances
        unique_name = f"{dataset_name}_{len(self.merge_ctrl.model.sources)}"

        # Create source classes from the dataset
        per_imgs = defaultdict(int)
        per_boxes = defaultdict(int)

        for info in dm.splits.values():
            labels_dir = info.get("labels_dir")
            images_dir = info.get("images_dir")
            for img in info.get("images", []):
                boxes = self._label_cache.get(img)
                if boxes is None:
                    boxes = read_yolo_txt(labels_for_image(img, labels_dir, images_dir))
                    boxes, changed = self._sanitize_boxes_for_image(img, boxes)
                    if changed:
                        self._notify_box_adjustment(img)
                    self._label_cache[img] = boxes
                    self._image_class_sets[img] = {b.cls for b in boxes}
                if boxes:
                    new_max = max(b.cls for b in boxes)
                    if new_max > self._max_class_id:
                        self._max_class_id = new_max
                        self._ensure_class_combo_capacity(new_max)
                seen = self._image_class_sets.get(img)
                if seen is None:
                    seen = {b.cls for b in boxes}
                    self._image_class_sets[img] = seen
                for b in boxes:
                    per_boxes[b.cls] += 1
                for c in seen:
                    per_imgs[c] += 1
        
        # Create SourceClass objects
        from .merge_designer.controller import SourceClass
        source_classes = []
        for cls_id in sorted(set(per_imgs.keys()) | set(per_boxes.keys())):
            class_names = dm.names or []
            name = class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)
            source_classes.append(SourceClass(
                dataset_id=unique_name,
                class_id=cls_id,
                class_name=name,
                images=per_imgs.get(cls_id, 0),
                boxes=per_boxes.get(cls_id, 0)
            ))

        # Add to controller and spawn on canvas
        self.merge_ctrl.upsert_dataset(unique_name, source_classes)
        self.merge_canvas.spawn_dataset_node(unique_name, source_classes)
    
    def _spawn_target_node(self, name: str, quota: int = None):
        """Spawn a target node on the merge canvas"""
        if not hasattr(self, 'merge_canvas') or not hasattr(self, 'merge_ctrl'):
            return
        
        target_id = self.merge_ctrl.add_target_class(name, quota)
        self.merge_canvas.spawn_target_node(target_id, name, quota)


    def _write_export_yaml(self, dest_root: Path, names: List[str]):
        yaml_content = {
            "path": str(dest_root.resolve()),
            "nc": len(names),
            "names": names,
        }
        for split in self.dm.ordered_splits():
            yaml_content[split] = f"{split}/images"
        yaml_path = dest_root / "data.yaml"
        yaml_path.write_text(yaml.safe_dump(yaml_content, sort_keys=False), encoding="utf-8")

    def _resolve_target_names(self, targets) -> tuple[List[str], Dict[int, int]]:
        ordered_ids = sorted(targets.keys())
        names: List[str] = []
        remap: Dict[int, int] = {}
        for new_tid, old_tid in enumerate(ordered_ids):
            tgt = targets[old_tid]
            label = (tgt.class_name or f"class_{old_tid}").strip() or f"class_{old_tid}"
            names.append(label)
            remap[old_tid] = new_tid
        return names, remap

    def _prompt_export_mode(self) -> bool | None:
        dlg = QMessageBox(self)
        dlg.setWindowTitle('Export Options')
        dlg.setText('Choose how to run the export.')
        dry_button = dlg.addButton('Dry Run', QMessageBox.ActionRole)
        export_button = dlg.addButton('Export', QMessageBox.AcceptRole)
        cancel_button = dlg.addButton(QMessageBox.Cancel)
        dlg.setDefaultButton(export_button)
        dlg.exec()
        clicked = dlg.clickedButton()
        if clicked is cancel_button:
            return None
        return clicked is dry_button

    def _choose_fallback_target(self, targets: dict[int, str]) -> tuple[bool | None, int | None]:
        options = ['Drop unmapped classes']
        mapping: list[int | None] = [None]
        for tid in sorted(targets.keys()):
            label = targets[tid] or f'class_{tid}'
            options.append(f'[{tid}] {label}')
            mapping.append(tid)
        choice, ok = QInputDialog.getItem(self, 'Fallback mapping', 'Send unmapped boxes to:', options, 0, False)
        if not ok:
            return None, None
        idx = options.index(choice)
        return (idx != 0), mapping[idx]


    def _export_merged_dataset(self):
        if not hasattr(self, "merge_ctrl"):
            return
        if not self.dm or not self.dm.has_any():
            QMessageBox.warning(self, "No dataset", "Load a dataset before exporting.")
            return

        model = self.merge_ctrl.model
        if not model.targets:
            QMessageBox.warning(self, "No targets", "Define at least one target class in the merge designer.")
            return

        mapping_by_dataset: dict[str, dict[int, int]] = {}
        for edge in model.edges:
            ds, cid = edge.source_key
            mapping_by_dataset.setdefault(ds, {})[cid] = edge.target_id

        dataset_id = self.ds_name

        dest_dir = QFileDialog.getExistingDirectory(self, "Select export folder")
        if not dest_dir:
            return

        mode_choice = self._prompt_export_mode()
        if mode_choice is None:
            return
        dry_run = mode_choice

        dest_root = Path(dest_dir)
        try:
            dest_root.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", f"Could not access output folder:\n{exc}")
            return

        names, tid_remap = self._resolve_target_names(model.targets)
        if not names:
            QMessageBox.warning(self, "No targets", "Create at least one target class before exporting.")
            return

        dataset_mapping_old = {}
        for ds, mapping in mapping_by_dataset.items():
            if ds.startswith(dataset_id):
                dataset_mapping_old.update(mapping)
        if not dataset_mapping_old:
            QMessageBox.warning(self, "No mappings", f"No class mappings defined for dataset '{dataset_id}'.")
            return

        dataset_mapping = {}
        for src_cls, old_tid in dataset_mapping_old.items():
            new_tid = tid_remap.get(old_tid)
            if new_tid is not None:
                dataset_mapping[src_cls] = new_tid
        if not dataset_mapping:
            QMessageBox.warning(self, "No mappings", f"No valid class mappings defined for dataset '{dataset_id}'.")
            return

        target_labels_for_prompt = {old_tid: names[new_tid] for old_tid, new_tid in tid_remap.items()}

        fallback_target_id: int | None = None
        fallback_label = ""
        fallback_enabled = False
        dataset_sources = self.merge_ctrl.model.sources.get(dataset_id, [])
        mapped_source_ids = set(dataset_mapping.keys())
        unmapped_sources = [src for src in dataset_sources if src.class_id not in mapped_source_ids]
        if unmapped_sources:
            msg = "The following source classes are not mapped and will be dropped:\n" + "\n".join(f"  {src.class_name} ({src.class_id})" for src in unmapped_sources)
            QMessageBox.warning(self, "Unmapped classes", msg)
        if unmapped_sources:
            choice_enabled, choice_tid = self._choose_fallback_target(target_labels_for_prompt)
            if choice_enabled is None:
                return
            if choice_enabled and choice_tid is not None:
                remapped_tid = tid_remap.get(choice_tid)
                if remapped_tid is None:
                    QMessageBox.warning(self, "Fallback unavailable", "Selected fallback target is no longer available.")
                    return
                fallback_target_id = remapped_tid
                fallback_enabled = True
                fallback_label = names[fallback_target_id] if 0 <= fallback_target_id < len(names) else f"class_{fallback_target_id}"
            else:
                fallback_target_id = None
                fallback_enabled = False
                fallback_label = ""

        quota_map = {new_tid: self.merge_ctrl.get_target_quota(old_tid) for old_tid, new_tid in tid_remap.items()}
        images_per_target = {new_tid: 0 for new_tid in tid_remap.values()}
        if fallback_target_id is not None and fallback_target_id not in images_per_target:
            images_per_target[fallback_target_id] = 0
        edge_limits = dict(self.merge_ctrl.model.edge_limits)
        edge_usage = defaultdict(int)

        total_images = sum(len(info["images"]) for info in self.dm.splits.values())
        if total_images <= 0:
            QMessageBox.warning(self, "No images", "No images available to export.")
            return

        manifest_writer = ManifestWriter(dest_root, dry_run)
        manifest_writer._meta["dataset_id"] = dataset_id
        manifest_writer._meta["target_names"] = names
        manifest_writer._meta["export_mode"] = "dry_run" if dry_run else "export"
        manifest_writer._meta["fallback_target"] = fallback_target_id
        manifest_writer._meta["fallback_label"] = fallback_label
        manifest_writer._meta["fallback_enabled"] = fallback_enabled

        scan_progress = QProgressDialog("Analyzing dataset...", "Cancel", 0, total_images, self)
        scan_progress.setWindowModality(Qt.WindowModal)
        scan_progress.setMinimumDuration(0)
        scan_progress.setAutoReset(False)
        scan_progress.setAutoClose(False)

        scan_count = 0
        scan_cancelled = False
        per_split_target_counts: dict[str, defaultdict[int]] = {}
        manifest_map: dict[tuple[str, str], dict] = {}
        image_boxes_cache: Dict[Path, List[Box]] = {}
        copied = 0
        written = 0
        dropped_unmapped = 0
        dropped_with_unmapped_image = 0
        dropped_by_quota = 0
        dropped_edge_limit = 0
        fallback_boxes_relabelled = 0
        skipped_images_no_mapped = 0
        skipped_images_unmapped = 0
        skipped_images_quota = 0
        errors: List[str] = []

        def _relative_image_path(img_path: Path, images_dir: Path) -> Path:
            try:
                return img_path.relative_to(images_dir)
            except ValueError:
                return Path(img_path.name)

        def _label_relative_path(src_txt: Path, labels_dir: Path | None, rel_img: Path) -> Path:
            try:
                return src_txt.relative_to(labels_dir) if labels_dir else rel_img.with_suffix(".txt")
            except ValueError:
                return rel_img.with_suffix(".txt")

        try:
            for split in self.dm.ordered_splits():
                info = self.dm.splits.get(split)
                if not info:
                    continue
                images_dir: Path = info["images_dir"]
                labels_dir = info.get("labels_dir")
                per_split_target_counts.setdefault(split, defaultdict(int))
                sorted_images = sorted(info["images"], key=lambda p: _relative_image_path(p, images_dir).as_posix())
                for img_path in sorted_images:
                    if scan_cancelled:
                        break

                    scan_count += 1
                    scan_progress.setLabelText(f"Analyzing {split} ({scan_count}/{total_images})")
                    scan_progress.setValue(scan_count)
                    QApplication.processEvents()
                    if scan_progress.wasCanceled():
                        scan_cancelled = True
                        break

                    rel_img = _relative_image_path(img_path, images_dir)
                    rel_key = (split, rel_img.as_posix())

                    src_txt = labels_for_image(img_path, labels_dir, images_dir)
                    boxes = read_yolo_txt(src_txt)
                    boxes, changed = self._sanitize_boxes_for_image(img_path, boxes)
                    if changed:
                        record["notes"].append("Adjusted extremely small boxes during analysis")
                    image_boxes_cache[img_path] = boxes
                    self._label_cache[img_path] = boxes
                    self._image_class_sets[img_path] = {b.cls for b in boxes}
                    label_rel = _label_relative_path(src_txt, labels_dir, rel_img)

                    record = {
                        "split": split,
                        "image": rel_img.as_posix(),
                        "source_image": str(img_path),
                        "label_rel": label_rel.as_posix(),
                        "status": "scanned",
                        "boxes_total": len(boxes),
                        "mapped_targets": [],
                        "included_targets": [],
                        "skipped_targets": {},
                        "fallback_boxes": 0,
                        "fallback_target": fallback_target_id,
                        "source_classes": [],
                        "notes": [],
                    }

                    mapped_targets: set[int] = set()
                    source_classes_present: dict[int, set[int]] = {}
                    mapped_count = 0
                    fallback_count_local = 0
                    has_unmapped = False

                    for b in boxes:
                        tgt_id = dataset_mapping.get(b.cls)
                        if tgt_id is None:
                            if fallback_target_id is not None:
                                tgt_id = fallback_target_id
                                fallback_count_local += 1
                            else:
                                dropped_unmapped += 1
                                has_unmapped = True
                                continue
                        mapped_targets.add(tgt_id)
                        source_classes_present.setdefault(tgt_id, set()).add(b.cls)
                        mapped_count += 1

                    record["fallback_boxes"] = fallback_count_local
                    record["mapped_targets"] = sorted(mapped_targets)
                    record["source_classes"] = sorted({cls for cls_set in source_classes_present.values() for cls in cls_set})

                    if fallback_count_local:
                        fallback_boxes_relabelled += fallback_count_local

                    if has_unmapped:
                        skipped_images_unmapped += 1
                        dropped_with_unmapped_image += mapped_count
                        record["status"] = "skipped_unmapped"
                        record["notes"].append("Dropped due to unmapped classes")
                        manifest_writer.append(record)
                        continue

                    if not mapped_targets:
                        skipped_images_no_mapped += 1
                        record["status"] = "skipped_no_mapping"
                        record["notes"].append("No mapped boxes for this image")
                        manifest_writer.append(record)
                        continue

                    record["status"] = "queued"
                    manifest_map[rel_key] = record
                    for tid in mapped_targets:
                        per_split_target_counts[split][tid] += 1
        finally:
            scan_progress.close()

        if scan_cancelled:
            for key, record in list(manifest_map.items()):
                if record.get("status") == "queued":
                    record["status"] = "not_processed"
                    record["notes"].append("Export canceled during analysis")
                manifest_writer.append(record)
                manifest_map.pop(key, None)
            try:
                manifest_file = manifest_writer.finalize()
            except Exception as exc:
                QMessageBox.warning(self, "Export canceled", f"Analysis canceled and manifest failed: {exc}")
                return
            QMessageBox.information(
                self,
                "Export canceled",
                f"Export was canceled during analysis.\nManifest written: {manifest_file}"
            )
            return

        quota_per_split: dict[str, dict[int, int]] = {}
        for tid, quota in quota_map.items():
            if quota is None:
                continue
            available_by_split: dict[str, int] = {}
            for split in self.dm.ordered_splits():
                split_counts = per_split_target_counts.get(split)
                available_by_split[split] = split_counts.get(tid, 0) if split_counts else 0
            total_available = sum(available_by_split.values())
            if total_available <= 0:
                continue
            if quota >= total_available:
                for split, count in available_by_split.items():
                    if count:
                        quota_per_split.setdefault(split, {})[tid] = count
                continue
            alloc: dict[str, int] = {}
            remainders: list[tuple[float, str, int]] = []
            for split, count in available_by_split.items():
                if count <= 0:
                    continue
                exact = quota * count / total_available
                base = min(int(exact), count)
                alloc[split] = base
                remainders.append((exact - base, split, count))
            assigned = sum(alloc.values())
            remaining = min(quota, total_available) - assigned
            remainders.sort(key=lambda x: x[0], reverse=True)
            for _, split, count in remainders:
                if remaining <= 0:
                    break
                if alloc.get(split, 0) < count:
                    alloc[split] = alloc.get(split, 0) + 1
                    remaining -= 1
            for split, count in alloc.items():
                if count > 0:
                    quota_per_split.setdefault(split, {})[tid] = min(count, available_by_split[split])

        manifest_writer._meta["queued_images"] = len(manifest_map)

        images_per_split_target = defaultdict(int)
        export_keys = sorted(manifest_map.keys(), key=lambda key: (key[0], key[1]))
        export_progress = None
        if export_keys:
            export_progress = QProgressDialog("Exporting merged dataset...", "Cancel", 0, len(export_keys), self)
            export_progress.setWindowModality(Qt.WindowModal)
            export_progress.setMinimumDuration(0)
            export_progress.setAutoReset(False)
            export_progress.setAutoClose(False)

        export_cancelled = False

        for index, key in enumerate(export_keys, start=1):
            if export_cancelled:
                break
            record = manifest_map[key]

            if export_progress:
                export_progress.setLabelText(f"Exporting {record['split']} ({index}/{len(export_keys)})")
                export_progress.setValue(index)
                QApplication.processEvents()
                if export_progress.wasCanceled():
                    export_cancelled = True
                    break

            split = record["split"]
            rel_img = Path(record["image"])
            label_rel = Path(record["label_rel"])
            img_path = Path(record["source_image"])
            info = self.dm.splits.get(split)
            labels_dir = info.get("labels_dir") if info else None
            images_dir = info.get("images_dir") if info else None

            boxes = image_boxes_cache.get(img_path)
            if boxes is None:
                src_txt = labels_for_image(img_path, labels_dir, images_dir)
                boxes = read_yolo_txt(src_txt)
            boxes, changed = self._sanitize_boxes_for_image(img_path, boxes)
            if changed:
                record.setdefault("notes", []).append("Adjusted extremely small boxes before export")
            image_boxes_cache[img_path] = boxes
            self._label_cache[img_path] = boxes
            self._image_class_sets[img_path] = {b.cls for b in boxes}

            boxes_by_target: dict[int, List[Box]] = {}
            per_target_sources: dict[int, set[int]] = {}
            fallback_count_local = 0
            has_unmapped = False
            mapped_count = 0

            for b in boxes:
                tgt_id = dataset_mapping.get(b.cls)
                if tgt_id is None:
                    if fallback_target_id is not None:
                        tgt_id = fallback_target_id
                        fallback_count_local += 1
                    else:
                        dropped_unmapped += 1
                        has_unmapped = True
                        continue
                boxes_by_target.setdefault(tgt_id, []).append(Box(tgt_id, b.cx, b.cy, b.w, b.h))
                per_target_sources.setdefault(tgt_id, set()).add(b.cls)
                mapped_count += 1

            record["fallback_boxes"] = fallback_count_local
            record["mapped_targets"] = sorted(boxes_by_target.keys())
            record["source_classes"] = sorted({cls for cls_set in per_target_sources.values() for cls in cls_set})
            record["skipped_targets"] = {}

            if has_unmapped and fallback_target_id is None:
                skipped_images_unmapped += 1
                dropped_with_unmapped_image += mapped_count
                record["status"] = "skipped_unmapped"
                record["notes"].append("Dropped due to unmapped classes (post-analysis)")
                manifest_writer.append(record)
                manifest_map.pop(key, None)
                continue

            filtered_boxes: List[Box] = []
            included_targets_info: List[tuple[int, set[int]]] = []
            for tgt_id, box_list in boxes_by_target.items():
                reason = None
                quota_total = quota_map.get(tgt_id)
                if quota_total is not None and images_per_target.get(tgt_id, 0) >= quota_total:
                    reason = "target quota reached"
                else:
                    quota_split = quota_per_split.get(split, {}).get(tgt_id)
                    if quota_split is not None and images_per_split_target[(split, tgt_id)] >= quota_split:
                        reason = "split quota reached"
                if reason is None:
                    for src_cls in per_target_sources.get(tgt_id, set()):
                        limit = edge_limits.get((dataset_id, src_cls))
                        if limit is not None and edge_usage[(dataset_id, src_cls)] >= limit:
                            reason = "edge limit reached"
                            break
                if reason:
                    if reason == "edge limit reached":
                        dropped_edge_limit += len(box_list)
                    else:
                        dropped_by_quota += len(box_list)
                    record["skipped_targets"][str(tgt_id)] = reason
                    continue
                filtered_boxes.extend(box_list)
                included_targets_info.append((tgt_id, set(per_target_sources.get(tgt_id, set()))))

            if not filtered_boxes:
                skipped_images_quota += 1
                record["status"] = "skipped_quota"
                record["notes"].append("All mapped targets exceeded quotas or limits")
                manifest_writer.append(record)
                manifest_map.pop(key, None)
                continue

            record["included_targets"] = sorted(tid for tid, _ in included_targets_info)

            if dry_run:
                record["status"] = "dry_run"
                record["notes"].append("Dry run only (no files written)")
                copied += 1
                written += 1
                manifest_writer.append(record)
                manifest_map.pop(key, None)
                continue

            dest_images_dir = dest_root / split / "images"
            dest_labels_dir = dest_root / split / "labels"
            out_img = dest_images_dir / rel_img
            out_img.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(img_path, out_img)
            except Exception as exc:
                errors.append(f"Image copy failed ({img_path.name}): {exc}")
                record["status"] = "copy_failed"
                record["notes"].append(str(exc))
                manifest_writer.append(record)
                manifest_map.pop(key, None)
                continue
            copied += 1

            out_label = dest_labels_dir / label_rel
            out_label.parent.mkdir(parents=True, exist_ok=True)
            try:
                write_yolo_txt(out_label, filtered_boxes)
            except Exception as exc:
                errors.append(f"Label write failed ({out_label.name}): {exc}")
                record["status"] = "label_failed"
                record["notes"].append(str(exc))
                manifest_writer.append(record)
                manifest_map.pop(key, None)
                continue
            written += 1
            record["status"] = "exported"

            for tgt_id, src_cls_set in included_targets_info:
                images_per_target[tgt_id] = images_per_target.get(tgt_id, 0) + 1
                images_per_split_target[(split, tgt_id)] += 1
                for src_cls in src_cls_set:
                    edge_usage[(dataset_id, src_cls)] += 1
            manifest_writer.append(record)
            manifest_map.pop(key, None)

        if export_progress:
            export_progress.close()

        if export_cancelled:
            for key, record in list(manifest_map.items()):
                if record.get("status") == "queued":
                    record["status"] = "not_processed"
                    record["notes"].append("Export canceled before processing")
                manifest_writer.append(record)
                manifest_map.pop(key, None)
            try:
                manifest_file = manifest_writer.finalize()
            except Exception as exc:
                QMessageBox.warning(self, "Export canceled", f"Export canceled and manifest failed: {exc}")
                return
            QMessageBox.information(
                self,
                "Export canceled",
                f"Export was canceled. Some files may have been written.\nManifest written: {manifest_file}"
            )
            return

        for key, record in list(manifest_map.items()):
            if record.get("status") == "queued":
                record["status"] = "not_processed"
                record["notes"].append("Image was queued but not processed")
            manifest_writer.append(record)
            manifest_map.pop(key, None)

        if not dry_run:
            try:
                self._write_export_yaml(dest_root, names)
            except Exception as exc:
                manifest_writer.abort()
                QMessageBox.critical(self, "Export failed", f"Could not write data.yaml:\n{exc}")
                return

        manifest_writer._meta["stats"] = {
            "copied_images": copied,
            "written_labels": written,
            "skipped_unmapped": skipped_images_unmapped,
            "skipped_no_mapping": skipped_images_no_mapped,
            "skipped_quota": skipped_images_quota,
            "dropped_unmapped_boxes": dropped_unmapped,
            "dropped_with_unmapped_image": dropped_with_unmapped_image,
            "dropped_by_quota": dropped_by_quota,
            "dropped_edge_limit": dropped_edge_limit,
            "fallback_boxes": fallback_boxes_relabelled,
        }
        manifest_writer._meta["quota_per_split"] = {split: dict(values) for split, values in quota_per_split.items()}

        try:
            manifest_file = manifest_writer.finalize()
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", f"Could not write manifest:\n{exc}")
            return

        summary = [f"Images copied: {copied}", f"Label files written: {written}"]
        if dry_run:
            summary.insert(0, "Dry run only: no files were written.")
        if skipped_images_unmapped:
            summary.append(f"Images skipped (unmapped classes present): {skipped_images_unmapped}")
        if skipped_images_no_mapped:
            summary.append(f"Images skipped (no mapped boxes): {skipped_images_no_mapped}")
        if skipped_images_quota:
            summary.append(f"Images skipped (quota exhausted): {skipped_images_quota}")
        if dropped_unmapped:
            summary.append(f"Boxes skipped (no mapping): {dropped_unmapped}")
        if dropped_with_unmapped_image:
            summary.append(f"Boxes skipped (discarded with unmapped image): {dropped_with_unmapped_image}")
        if dropped_by_quota:
            summary.append(f"Boxes skipped (quota reached): {dropped_by_quota}")
        if dropped_edge_limit:
            summary.append(f"Boxes skipped (edge limit reached): {dropped_edge_limit}")
        if fallback_target_id is not None and fallback_boxes_relabelled:
            summary.append(f"Boxes remapped to fallback [{fallback_target_id}] {fallback_label or ''}: {fallback_boxes_relabelled}")
        quota_lines = []
        for tid, count in sorted(images_per_target.items()):
            quota = quota_map.get(tid)
            if quota is None:
                continue
            label = names[tid] if 0 <= tid < len(names) and names[tid] else f"class_{tid}"
            quota_lines.append(f"[{tid}] {label}: {count}/{quota} images")
        if quota_lines:
            summary.append("Quota usage:")
            summary.extend(quota_lines)
        summary.append(f"Manifest written: {manifest_file}")
        summary.append(f"Manifest metadata: {manifest_writer.meta_path}")
        summary.append(f"Manifest entries: {manifest_writer.count}")
        if errors:
            shown = "\n    ".join(errors[:5])
            summary.append("Warnings:\n    " + shown)
            if len(errors) > 5:
                summary.append(f"...and {len(errors) - 5} more issues.")
        QMessageBox.information(
            self,
            "Export complete" if not dry_run else "Dry run complete",
            "\n    ".join(summary)
        )

