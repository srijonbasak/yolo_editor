"""Presentation layer logic extracted from the legacy main window."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from collections import defaultdict
import shutil

from PySide6.QtCore import QObject, Qt, QThread
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QInputDialog,
    QListWidget,
    QMessageBox,
    QProgressDialog,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QComboBox,
    QPushButton,
    QLabel,
    QTabWidget,
)

from ..core.dataset_resolver import DatasetModel
from ..core.yolo_io import Box, labels_for_image, read_yolo_txt, write_yolo_txt, imread_unicode
from .image_view import ImageView, Box as ViewBox
from .main_window_support import (
    DatasetLoaderWorker,
    ManifestWriter,
    StatsWorker,
    sanitize_boxes_by_size,
    MergeDatasetStatsWorker,
)

if TYPE_CHECKING:  # pragma: no cover - typing-only imports
    from .merge_designer.controller import MergeController
    from .merge_designer.canvas import MergeCanvas
    from .merge_designer.palette import MergePalette


@dataclass
class EditorWidgets:
    """UI elements belonging to the dataset editor tab."""

    tabs: QTabWidget
    dataset_label: QLabel
    split_combo: QComboBox
    file_tree: QTreeWidget
    class_combo: QComboBox
    save_button: QPushButton
    image_view: ImageView
    labels_table: QTableWidget
    stats_list: QListWidget
    status_bar: QStatusBar


@dataclass
class MergeWidgets:
    """UI elements related to the optional merge designer tab."""

    controller: "MergeController"
    canvas: "MergeCanvas"
    palette: "MergePalette"


class MainWindowPresenter(QObject):
    """Encapsulates the non-UI logic previously embedded in the main window."""

    def __init__(
        self,
        window,
        widgets: EditorWidgets,
        merge_widgets: Optional[MergeWidgets] = None,
    ) -> None:
        super().__init__(window)
        self._window = window
        self._widgets = widgets
        self._merge = merge_widgets

        # Dataset state
        self.dm: Optional[DatasetModel] = None
        self.ds_name: str = "-"
        self.split: Optional[str] = None
        self.images: List[Path] = []
        self.labels_dir: Optional[Path] = None
        self.images_dir: Optional[Path] = None
        self.names: List[str] = []
        self.idx: int = -1

        # Background workers
        self._loader_thread: Optional[QThread] = None
        self._loader_worker: Optional[DatasetLoaderWorker] = None
        self._loader_dialog: Optional[QProgressDialog] = None
        self._stats_thread: Optional[QThread] = None
        self._stats_worker: Optional[StatsWorker] = None
        self._stats_dialog: Optional[QProgressDialog] = None

        # Caches and flags
        self._label_cache: Dict[Path, List[Box]] = {}
        self._image_class_sets: Dict[Path, set[int]] = {}
        self._max_class_id: int = -1
        self._stats_pending: bool = False
        self._stats_active: bool = False
        self._image_sizes: Dict[Path, Tuple[int, int]] = {}
        self._adjustment_notices: set[Path] = set()

        # Merge designer bookkeeping
        self._merge_loaded_datasets: List[str] = []
        self._merge_datasets: Dict[str, DatasetModel] = {}
        self._merge_dataset_stats_cache: Dict[str, list[dict]] = {}
        self._merge_dataset_instance_count: defaultdict[str, int] = defaultdict(int)
        self._merge_stats_threads: Dict[str, QThread] = {}
        self._merge_stats_workers: Dict[str, MergeDatasetStatsWorker] = {}
        self._merge_dataset_base: Dict[str, str] = {}
    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    @property
    def window(self):
        return self._window

    @property
    def tabs(self) -> QTabWidget:
        return self._widgets.tabs

    @property
    def dataset_label(self) -> QLabel:
        return self._widgets.dataset_label

    @property
    def split_combo(self) -> QComboBox:
        return self._widgets.split_combo

    @property
    def file_tree(self) -> QTreeWidget:
        return self._widgets.file_tree

    @property
    def class_combo(self) -> QComboBox:
        return self._widgets.class_combo

    @property
    def save_button(self) -> QPushButton:
        return self._widgets.save_button

    @property
    def view(self) -> ImageView:
        return self._widgets.image_view

    @property
    def labels_table(self) -> QTableWidget:
        return self._widgets.labels_table

    @property
    def stats_list(self) -> QListWidget:
        return self._widgets.stats_list

    @property
    def status_bar(self) -> QStatusBar:
        return self._widgets.status_bar

    @property
    def merge_controller(self):
        return self._merge.controller if self._merge else None

    @property
    def merge_canvas(self):
        return self._merge.canvas if self._merge else None

    @property
    def merge_palette(self):
        return self._merge.palette if self._merge else None

    # ------------------------------------------------------------------
    # Public API used by the main window shell
    # ------------------------------------------------------------------
    def bind(self) -> None:
        self.split_combo.currentTextChanged.connect(self._on_split_changed)
        self.file_tree.itemClicked.connect(self._on_file_clicked)
        self.class_combo.currentIndexChanged.connect(self._on_class_changed)
        self.save_button.clicked.connect(self._save_labels)
        self.view.boxesChanged.connect(self._on_boxes_changed)
        self.view.requestPrev.connect(self.go_previous_image)
        self.view.requestNext.connect(self.go_next_image)

    def open_dataset_root(self) -> None:
        self._open_root()

    def open_data_yaml(self) -> None:
        self._open_yaml()

    def show_diagnostics(self) -> None:
        self._show_diagnostics()

    def load_dataset_for_merge(self) -> None:
        self._load_dataset_for_merge()

    def export_merged_dataset(self) -> None:
        self._export_merged_dataset()

    def spawn_dataset_node(self, dataset_name: str) -> None:
        self._spawn_dataset_node(dataset_name)

    def spawn_target_node(self, name: str, quota: int | None = None) -> None:
        self._spawn_target_node(name, quota)

    def save_current_labels(self) -> None:
        self._save_labels()

    def go_previous_image(self) -> None:
        if not self.images:
            return
        self._open_index(max(0, self.idx - 1))

    def go_next_image(self) -> None:
        if not self.images:
            return
        self._open_index(min(len(self.images) - 1, self.idx + 1))
    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------
    def _open_root(self) -> None:
        directory = QFileDialog.getExistingDirectory(self.window, "Open Dataset Root")
        if not directory:
            return
        path = Path(directory)
        self._start_dataset_load(
            path,
            path.name,
            "Not a dataset",
            "Could not detect train/val/test/valid/eval in this folder.",
            merge_mode=False,
        )

    def _open_yaml(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self.window, "Open data.yaml", "", "YAML (*.yaml *.yml)"
        )
        if not file_path:
            return
        path = Path(file_path)
        dataset_name = path.parent.name or path.stem
        self._start_dataset_load(
            path,
            dataset_name,
            "Invalid YAML",
            "Could not resolve any split paths from this YAML.",
            merge_mode=False,
        )

    def _start_dataset_load(
        self,
        source: Path,
        dataset_name: str,
        failure_title: str,
        missing_split_message: str,
        merge_mode: bool = False,
    ) -> None:
        if self._loader_thread is not None:
            QMessageBox.information(
                self.window,
                "Loading",
                "A dataset load is already in progress. Please wait.",
            )
            return
        self._loader_dialog = QProgressDialog("Loading dataset...", None, 0, 0, self.window)
        self._loader_dialog.setWindowModality(Qt.WindowModal)
        self._loader_dialog.setCancelButton(None)
        self._loader_dialog.setMinimumDuration(0)
        self._loader_dialog.setRange(0, 0)
        self._loader_dialog.setLabelText("Loading dataset...")
        self._loader_dialog.show()

        self._loader_thread = QThread(self.window)
        self._loader_worker = DatasetLoaderWorker(
            source,
            dataset_name,
            failure_title,
            missing_split_message,
            merge_mode,
        )
        self._loader_worker.moveToThread(self._loader_thread)
        self._loader_thread.started.connect(self._loader_worker.run)
        self._loader_worker.finished.connect(self._on_dataset_loaded)
        self._loader_worker.failed.connect(self._on_dataset_failed)
        self._loader_thread.start()

    def _on_dataset_loaded(self, dataset_name: str, model_obj, merge_mode: bool) -> None:
        self._teardown_loader_thread()
        if not isinstance(model_obj, DatasetModel):
            QMessageBox.warning(
                self.window,
                "Dataset load failed",
                "Unexpected dataset payload.",
            )
            return
        self._load_dataset(dataset_name, model_obj, merge_mode)

    def _on_dataset_failed(self, title: str, message: str) -> None:
        self._teardown_loader_thread()
        QMessageBox.warning(
            self.window,
            title or "Dataset load failed",
            message or "Failed to load dataset.",
        )

    def _teardown_loader_thread(self) -> None:
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

    def _reset_cached_labels(self) -> None:
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
        return sanitize_boxes_by_size(boxes, size[0], size[1])

    def _notify_box_adjustment(self, img_path: Path) -> None:
        if img_path in self._adjustment_notices:
            return
        self._adjustment_notices.add(img_path)
        self.status_bar.showMessage(f"Adjusted extremely small box in {img_path.name}", 5000)

    def _get_or_load_boxes(
        self,
        img_path: Path,
        labels_dir: Optional[Path],
        images_dir: Optional[Path],
        *,
        notify: bool = False,
    ) -> List[Box]:
        boxes = self._label_cache.get(img_path)
        if boxes is None:
            boxes = read_yolo_txt(labels_for_image(img_path, labels_dir, images_dir))
            boxes, changed = self._sanitize_boxes_for_image(img_path, boxes)
            if changed and notify:
                self._notify_box_adjustment(img_path)
            self._label_cache[img_path] = boxes
            self._image_class_sets[img_path] = {b.cls for b in boxes}
        return boxes
    # ------------------------------------------------------------------
    # Statistics handling
    # ------------------------------------------------------------------
    def _cancel_stats_job(self, restart: bool = False) -> None:
        if self._stats_worker:
            self._stats_pending = restart
            self._stats_worker.cancel()
            if self._stats_dialog:
                self._stats_dialog.setLabelText("Stopping class statistics...")
        else:
            if restart:
                self._stats_pending = False

    def _teardown_stats_thread(self) -> None:
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

    def _on_stats_dialog_canceled(self) -> None:
        self._cancel_stats_job(restart=False)

    def _on_stats_progress(self, current: int, total: int) -> None:
        if not self._stats_active:
            return
        try:
            if self._stats_dialog:
                if total:
                    self._stats_dialog.setMaximum(total)
                self._stats_dialog.setValue(current)
                self._stats_dialog.setLabelText(
                    f"Computing class statistics... ({current}/{total})"
                )
        except AttributeError:
            pass  # dialog was deleted

    def _on_stats_finished(
        self,
        per_imgs: dict,
        per_boxes: dict,
        max_cls: int,
        cancelled: bool,
        folder_stats: dict,
        images_no_labels: list,
    ) -> None:
        self._teardown_stats_thread()
        if cancelled:
            if self._stats_pending:
                self._stats_pending = False
                self._compute_stats_and_show()
            return
        self._stats_pending = False
        self.stats_list.clear()
        total_images = len(self.images)
        self.stats_list.addItem(f"Total images: {total_images}")
        total_boxes = sum(per_boxes.values())
        self.stats_list.addItem(f"Total boxes: {total_boxes}")
        images_with_labels = total_images - len(images_no_labels)
        self.stats_list.addItem(f"Images with labels: {images_with_labels}")
        self.stats_list.addItem(f"Images with no labels: {len(images_no_labels)}")

        ids = sorted(set(per_imgs.keys()) | set(per_boxes.keys()))
        all_classes = set(range(len(self.names))) if self.names else set()
        zero_classes = all_classes - set(ids)

        self.stats_list.addItem("Classes:")
        for cid in sorted(ids):
            nm = self.names[cid] if 0 <= cid < len(self.names) else str(cid)
            self.stats_list.addItem(
                f"  [{cid}] {nm}: {per_imgs.get(cid, 0)} imgs / {per_boxes.get(cid, 0)} boxes"
            )
        for cid in sorted(zero_classes):
            nm = self.names[cid] if 0 <= cid < len(self.names) else str(cid)
            self.stats_list.addItem(f"  [{cid}] {nm}: 0 imgs / 0 boxes")

        if images_no_labels:
            self.stats_list.addItem("Images with no labels:")
            for img in images_no_labels[:10]:
                self.stats_list.addItem(f"  {img.name}")
            if len(images_no_labels) > 10:
                self.stats_list.addItem(f"  ...and {len(images_no_labels) - 10} more")
        if max_cls is not None and max_cls > self._max_class_id:
            self._max_class_id = max_cls
            self._ensure_class_combo_capacity(max_cls)
        elif ids:
            max_in_results = max(ids)
            if max_in_results > self._max_class_id:
                self._max_class_id = max_in_results
                self._ensure_class_combo_capacity(max_in_results)

        if folder_stats:
            self.stats_list.addItem("")
            self.stats_list.addItem("Folders:")
            for folder in sorted(folder_stats.keys()):
                info = folder_stats[folder]
                self.stats_list.addItem(f"  {folder}: {info['images']} images")
                per_class = info.get("per_class", {})
                if per_class:
                    class_parts = []
                    for cid, count in sorted(per_class.items()):
                        nm = self.names[cid] if 0 <= cid < len(self.names) else str(cid)
                        class_parts.append(f"[{cid}] {nm}={count}")
                    self.stats_list.addItem("    " + ", ".join(class_parts))

    def _on_stats_failed(self, message: str) -> None:
        self._teardown_stats_thread()
        self._stats_pending = False
        QMessageBox.warning(
            self.window,
            "Stats failed",
            f"Could not compute stats:\n{message}",
        )

    def _init_class_combo(self, max_cls: int) -> None:
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

    def _ensure_class_combo_capacity(self, cls_id: int) -> None:
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

    def _load_dataset(self, name: str, dm: DatasetModel, merge_mode: bool = False) -> None:
        if merge_mode:
            self.dm = dm
            self.ds_name = name
            self.names = list(dm.names) if dm.names else []
            self._reset_cached_labels()
            self.dataset_label.setText(f"Dataset: {name}")
            self.split_combo.blockSignals(True)
            self.split_combo.clear()
            for split_name in dm.ordered_splits():
                self.split_combo.addItem(split_name)
            self.split_combo.blockSignals(False)
            if dm.ordered_splits():
                self._on_split_changed(dm.ordered_splits()[0])
            self._merge_datasets[name] = dm
            if name not in self._merge_loaded_datasets:
                self._merge_loaded_datasets.append(name)
            if self.merge_palette:
                self.merge_palette.populate(self._merge_loaded_datasets)
                self.merge_palette.update()
        else:
            self.dm = dm
            self.ds_name = name
            self.dataset_label.setText(f"Dataset: {name}")
            self.names = list(dm.names) if dm.names else []
            self._reset_cached_labels()
            initial_max = len(self.names) - 1 if self.names else -1
            self._max_class_id = initial_max
            self._init_class_combo(initial_max)

            self.split_combo.blockSignals(True)
            self.split_combo.clear()
            for split_name in dm.ordered_splits():
                self.split_combo.addItem(split_name)
            self.split_combo.blockSignals(False)

            if dm.ordered_splits():
                self._on_split_changed(dm.ordered_splits()[0])

            if self.merge_palette:
                if name not in self._merge_loaded_datasets:
                    self._merge_loaded_datasets.append(name)
                    self._merge_datasets[name] = dm
                self.merge_palette.populate(self._merge_loaded_datasets)
    # ------------------------------------------------------------------
    # Split and file selectors
    # ------------------------------------------------------------------
    def _on_split_changed(self, split: str) -> None:
        if not self.dm or split not in self.dm.splits:
            return
        split_info = self.dm.splits[split]
        self.split = split
        self.images = split_info["images"]
        self.labels_dir = split_info["labels_dir"]
        self.images_dir = split_info.get("images_dir")
        self._populate_file_tree()
        self._compute_stats_and_show()
        if self.images:
            self._open_index(0)

    def _populate_file_tree(self) -> None:
        self.file_tree.clear()
        root = QTreeWidgetItem(["(images)"])
        for path in self.images:
            QTreeWidgetItem(root, [path.name])
        self.file_tree.addTopLevelItem(root)
        self.file_tree.expandAll()

    def _on_file_clicked(self, item: QTreeWidgetItem) -> None:
        if not item.parent():
            return
        index = item.parent().indexOfChild(item)
        self._open_index(index)

    # ------------------------------------------------------------------
    # Image interaction
    # ------------------------------------------------------------------
    def _open_index(self, index: int) -> None:
        if index < 0 or index >= len(self.images):
            return
        img_path = self.images[index]
        img = imread_unicode(img_path)
        if img is None:
            self.status_bar.showMessage(f"Failed to load: {img_path.name}", 4000)
            return
        self._ensure_image_size(img_path, img)
        self.view.show_image_bgr(img_path, img)
        core_boxes = self._get_or_load_boxes(
            img_path, self.labels_dir, self.images_dir, notify=True
        )
        if core_boxes:
            new_max = max(box.cls for box in core_boxes)
            if new_max > self._max_class_id:
                self._max_class_id = new_max
                self._ensure_class_combo_capacity(new_max)
        self.view.clear_boxes()
        view_boxes = [
            ViewBox(cls=box.cls, cx=box.cx, cy=box.cy, w=box.w, h=box.h)
            for box in core_boxes
        ]
        for view_box in view_boxes:
            self.view.add_box_norm(view_box)
        self._fill_table(view_boxes)
        self.idx = index
        self._highlight_tree_row(index)
        self.status_bar.showMessage(
            f"{img_path.name}  ({index + 1}/{len(self.images)})", 4000
        )

    def _save_labels(self) -> None:
        if self.idx < 0 or not self.images:
            return
        img_path = self.images[self.idx]
        txt_path = labels_for_image(img_path, self.labels_dir, self.images_dir)
        view_boxes = self.view.get_boxes_as_norm()
        core_boxes = [Box(int(box.cls), box.cx, box.cy, box.w, box.h) for box in view_boxes]
        sanitized, changed = self._sanitize_boxes_for_image(img_path, core_boxes)
        if changed:
            self._notify_box_adjustment(img_path)
        write_yolo_txt(txt_path, sanitized)
        self._label_cache[img_path] = sanitized
        class_set = {box.cls for box in sanitized}
        self._image_class_sets[img_path] = class_set
        if class_set:
            new_max = max(class_set)
            if new_max > self._max_class_id:
                self._max_class_id = new_max
                self._ensure_class_combo_capacity(new_max)
        self.view.clear_boxes()
        sanitized_view = [
            ViewBox(cls=box.cls, cx=box.cx, cy=box.cy, w=box.w, h=box.h)
            for box in sanitized
        ]
        for view_box in sanitized_view:
            self.view.add_box_norm(view_box)
        self._fill_table(sanitized_view)
        self.status_bar.showMessage(f"Saved: {txt_path}", 4000)
        self._compute_stats_and_show()

    def _fill_table(self, boxes: List[ViewBox]) -> None:
        self.labels_table.setRowCount(0)
        for box in boxes:
            row = self.labels_table.rowCount()
            self.labels_table.insertRow(row)
            name = self.names[box.cls] if 0 <= box.cls < len(self.names) else str(box.cls)
            self.labels_table.setItem(row, 0, QTableWidgetItem(name))
            self.labels_table.setItem(row, 1, QTableWidgetItem(str(box.cls)))
            self.labels_table.setItem(row, 2, QTableWidgetItem(f"{box.cx:.4f}"))
            self.labels_table.setItem(row, 3, QTableWidgetItem(f"{box.cy:.4f}"))
            self.labels_table.setItem(row, 4, QTableWidgetItem(f"{box.w:.4f}"))
            self.labels_table.setItem(row, 5, QTableWidgetItem(f"{box.h:.4f}"))

    def _on_boxes_changed(self) -> None:
        if self.idx < 0:
            return
        boxes = self.view.get_boxes_as_norm()
        view_boxes = [
            ViewBox(cls=box.cls, cx=box.cx, cy=box.cy, w=box.w, h=box.h)
            for box in boxes
        ]
        self._fill_table(view_boxes)

    def _highlight_tree_row(self, index: int) -> None:
        root = self.file_tree.topLevelItem(0)
        if not root:
            return
        for i in range(root.childCount()):
            child = root.child(i)
            child.setSelected(i == index)
        self.file_tree.scrollToItem(root.child(index))

    def _on_class_changed(self, index: int) -> None:
        self.view.set_current_class(index, self.names)
        nm = self.names[index] if 0 <= index < len(self.names) else str(index)
        self.status_bar.showMessage(f"Current class -> {nm} [{index}]", 3000)

    # ------------------------------------------------------------------
    # Statistics initiation
    # ------------------------------------------------------------------
    def _compute_stats_and_show(self) -> None:
        if not self.images:
            self.stats_list.clear()
            self.stats_list.addItem("Total images: 0")
            return

        if self._stats_thread is not None:
            self._cancel_stats_job(restart=True)
            return

        self.stats_list.clear()
        self.stats_list.addItem(f"Total images: {len(self.images)}")
        self.stats_list.addItem("Computing class statistics...")

        self._stats_dialog = QProgressDialog(
            "Computing class statistics...",
            "Cancel",
            0,
            len(self.images),
            self.window,
        )
        self._stats_dialog.setWindowModality(Qt.WindowModal)
        self._stats_dialog.setMinimumDuration(0)
        self._stats_dialog.setAutoReset(False)
        self._stats_dialog.setAutoClose(False)
        self._stats_dialog.setValue(0)
        self._stats_dialog.canceled.connect(self._on_stats_dialog_canceled)

        self._stats_active = True

        self._stats_pending = False
        self._stats_worker = StatsWorker(
            self.images.copy(), self.labels_dir, self.images_dir, self._image_sizes
        )
        self._stats_thread = QThread(self.window)
        self._stats_worker.moveToThread(self._stats_thread)
        self._stats_thread.started.connect(self._stats_worker.run)
        self._stats_worker.progress.connect(self._on_stats_progress)
        self._stats_worker.finished.connect(self._on_stats_finished)
        self._stats_worker.failed.connect(self._on_stats_failed)
        self._stats_worker.finished.connect(lambda *_: self._stats_thread.quit())
        self._stats_worker.failed.connect(lambda *_: self._stats_thread.quit())
        self._stats_thread.finished.connect(self._stats_thread.deleteLater)
        self._stats_thread.start()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def _show_diagnostics(self) -> None:
        if not self.dm:
            QMessageBox.information(self.window, "No dataset", "Load a dataset first.")
            return
        lines = [f"Dataset: {self.ds_name}"]
        if getattr(self.dm, "yaml_path", None):
            lines.append(f"YAML: {self.dm.yaml_path}")
        if not self.dm.splits:
            lines.append("No splits resolved.")
        else:
            for split_name in self.dm.ordered_splits():
                info = self.dm.splits[split_name]
                img_dir = info.get("images_dir")
                labels_dir = info.get("labels_dir")
                count = len(info.get("images", []))
                lines.append(f"[{split_name}] images: {img_dir}")
                label_text = labels_dir if labels_dir else "(next to images)"
                lines.append(f"    labels: {label_text}")
                lines.append(f"    image count: {count}")
        QMessageBox.information(
            self.window,
            "Dataset diagnostics",
            "\n".join(lines),
        )
    # ------------------------------------------------------------------
    # Merge designer integration
    # ------------------------------------------------------------------
    def _load_dataset_for_merge(self) -> None:
        directory = QFileDialog.getExistingDirectory(self.window, "Load Dataset for Merge")
        if not directory:
            return
        path = Path(directory)
        dataset_name = path.name
        if dataset_name in self._merge_datasets:
            QMessageBox.information(
                self.window,
                "Already loaded",
                f"Dataset '{dataset_name}' is already loaded for merge.",
            )
            return
        self._start_dataset_load(
            path,
            dataset_name,
            "Load failed",
            "Could not load dataset for merge.",
            merge_mode=True,
        )

    def _spawn_dataset_node(self, dataset_name: str):
        canvas = self.merge_canvas
        ctrl = self.merge_controller
        palette = self.merge_palette
        if not (canvas and ctrl and palette):
            return
        if dataset_name not in self._merge_loaded_datasets:
            QMessageBox.warning(self.window, "Dataset not loaded", f"Dataset '{dataset_name}' is not loaded for merge.")
            return

        dm = self._merge_datasets.get(dataset_name)
        if dm is None:
            QMessageBox.warning(self.window, "Dataset not found", f"Dataset '{dataset_name}' data not available.")
            return

        index = self._merge_dataset_instance_count[dataset_name]
        self._merge_dataset_instance_count[dataset_name] = index + 1
        unique_name = f"{dataset_name}_{index}"
        self._merge_dataset_base[unique_name] = dataset_name

        # Register empty dataset so edges can be created once stats arrive
        ctrl.upsert_dataset(unique_name, [])
        canvas.spawn_dataset_node(unique_name, [], loading=True)

        cached = self._merge_dataset_stats_cache.get(dataset_name)
        if cached is not None:
            self._apply_merge_dataset_stats(unique_name, dataset_name, cached)
            return

        self._start_merge_dataset_stats(unique_name, dataset_name, dm)
    def _start_merge_dataset_stats(self, dataset_id: str, dataset_name: str, dm: DatasetModel) -> None:
        if dataset_id in self._merge_stats_threads:
            return
        worker = MergeDatasetStatsWorker(dataset_id, dataset_name, dm, self._label_cache)
        thread = QThread(self.window)
        worker.moveToThread(thread)
        self._merge_stats_workers[dataset_id] = worker
        self._merge_stats_threads[dataset_id] = thread
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_merge_dataset_stats_ready)
        worker.failed.connect(self._on_merge_dataset_stats_failed)
        worker.finished.connect(lambda *_: thread.quit())
        worker.failed.connect(lambda *_: thread.quit())
        thread.finished.connect(lambda uname=dataset_id: self._cleanup_merge_stats_thread(uname))
        thread.start()

    def _apply_merge_dataset_stats(self, dataset_id: str, dataset_name: str, items: list[dict]) -> None:
        from .merge_designer.controller import SourceClass

        source_classes = [
            SourceClass(
                dataset_id=dataset_id,
                class_id=item["class_id"],
                class_name=item["class_name"],
                images=item["images"],
                boxes=item["boxes"],
            )
            for item in items
        ]

        for sc in source_classes:
            if sc.class_id > self._max_class_id:
                self._max_class_id = sc.class_id
                self._ensure_class_combo_capacity(sc.class_id)

        ctrl = self.merge_controller
        if ctrl:
            ctrl.upsert_dataset(dataset_id, source_classes)
        canvas = self.merge_canvas
        if canvas:
            canvas.update_dataset_stats(dataset_id, source_classes)
        if self.merge_palette:
            self.merge_palette.update()

    def _on_merge_dataset_stats_ready(self, dataset_id: str, dataset_name: str, items: list) -> None:
        clean_items = [dict(item) for item in items]
        self._merge_dataset_stats_cache[dataset_name] = clean_items
        ctrl = self.merge_controller
        if ctrl and dataset_id in ctrl.model.sources:
            self._apply_merge_dataset_stats(dataset_id, dataset_name, clean_items)

    def _on_merge_dataset_stats_failed(self, dataset_id: str, dataset_name: str, message: str) -> None:
        canvas = self.merge_canvas
        if canvas:
            canvas.set_dataset_error(dataset_id, message)
        QMessageBox.warning(self.window, "Dataset scan failed", message)
        ctrl = self.merge_controller
        if ctrl:
            ctrl.remove_dataset(dataset_id)
        self._merge_dataset_base.pop(dataset_id, None)

    def _cleanup_merge_stats_thread(self, dataset_id: str) -> None:
        thread = self._merge_stats_threads.pop(dataset_id, None)
        worker = self._merge_stats_workers.pop(dataset_id, None)
        if worker is not None:
            worker.deleteLater()
        if thread is not None:
            thread.deleteLater()

    def _spawn_target_node(self, name: str, quota: int = None) -> None:
        canvas = self.merge_canvas
        controller = self.merge_controller
        if not (canvas and controller):
            return
        target_id = controller.add_target_class(name, quota)
        canvas.spawn_target_node(target_id, name, quota)
    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------
    def _write_export_yaml(self, dest_root: Path, names: List[str]) -> None:
        yaml_content = {
            "path": str(dest_root.resolve()),
            "nc": len(names),
            "names": names,
        }
        for split_name in self.dm.ordered_splits():
            yaml_content[split_name] = f"{split_name}/images"
        yaml_path = dest_root / "data.yaml"
        import yaml  # Lazy import to avoid global dependency during tests

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

    def _prompt_export_mode(self) -> Optional[bool]:
        dlg = QMessageBox(self.window)
        dlg.setWindowTitle("Export Options")
        dlg.setText("Choose how to run the export.")
        dry_button = dlg.addButton("Dry Run", QMessageBox.ActionRole)
        export_button = dlg.addButton("Export", QMessageBox.AcceptRole)
        cancel_button = dlg.addButton(QMessageBox.Cancel)
        dlg.setDefaultButton(export_button)
        dlg.exec()
        clicked = dlg.clickedButton()
        if clicked is cancel_button:
            return None
        return clicked is dry_button

    def _choose_fallback_target(self, targets: dict[int, str]) -> tuple[Optional[bool], Optional[int]]:
        options = ["Drop unmapped classes"]
        mapping: List[Optional[int]] = [None]
        for tid in sorted(targets.keys()):
            label = targets[tid] or f"class_{tid}"
            options.append(f"[{tid}] {label}")
            mapping.append(tid)
        choice, ok = QInputDialog.getItem(
            self.window,
            "Fallback mapping",
            "Send unmapped boxes to:",
            options,
            0,
            False,
        )
        if not ok:
            return None, None
        idx = options.index(choice)
        return (idx != 0), mapping[idx]

    def _export_merged_dataset(self) -> None:
        if not self.merge_controller:
            return

        model = self.merge_controller.model
        if not model.targets:
            QMessageBox.warning(
                self.window,
                "No targets",
                "Define at least one target class in the merge designer.",
            )
            return

        mapping_by_dataset: Dict[str, Dict[int, int]] = {}
        for edge in model.edges:
            ds, cid = edge.source_key
            mapping_by_dataset.setdefault(ds, {})[cid] = edge.target_id

        if not mapping_by_dataset:
            QMessageBox.warning(
                self.window,
                "No mappings",
                "Map at least one source class to a target before exporting.",
            )
            return

        names, tid_remap = self._resolve_target_names(model.targets)
        if not names:
            QMessageBox.warning(
                self.window,
                "No targets",
                "Create at least one target class before exporting.",
            )
            return

        contexts: List[dict] = []
        missing_models: List[str] = []
        skipped_datasets: List[str] = []

        for dataset_id, source_mapping in mapping_by_dataset.items():
            base_name = self._merge_dataset_base.get(dataset_id, dataset_id)
            dm = self._merge_datasets.get(base_name)
            if dm is None:
                missing_models.append(dataset_id)
                continue
            dataset_mapping: Dict[int, int] = {}
            for src_cls, old_tid in source_mapping.items():
                new_tid = tid_remap.get(old_tid)
                if new_tid is not None:
                    dataset_mapping[src_cls] = new_tid
            if not dataset_mapping:
                skipped_datasets.append(dataset_id)
                continue
            if not any(len(info.get("images", [])) for info in dm.splits.values()):
                skipped_datasets.append(dataset_id)
                continue
            contexts.append(
                {
                    "dataset_id": dataset_id,
                    "base_name": base_name,
                    "model": dm,
                    "mapping": dataset_mapping,
                    "fallback_target_id": None,
                    "fallback_label": "",
                    "fallback_enabled": False,
                    "fallback_box_total": 0,
                }
            )

        if missing_models:
            missing_msg = "\n".join(sorted(missing_models))
            QMessageBox.warning(
                self.window,
                "Datasets unavailable",
                f"The following datasets are not available for export:\n{missing_msg}",
            )
            return

        if not contexts:
            QMessageBox.warning(
                self.window,
                "No datasets",
                "No mapped datasets are available for export. Verify your dataset nodes and mappings.",
            )
            return

        if skipped_datasets:
            skipped_labels = ", ".join(
                sorted({self._merge_dataset_base.get(ds, ds) for ds in skipped_datasets})
            )
            if skipped_labels:
                self.status_bar.showMessage(
                    f"Skipped datasets with no images or mappings: {skipped_labels}",
                    6000,
                )

        context_lookup = {ctx["dataset_id"]: ctx for ctx in contexts}

        dest_dir = QFileDialog.getExistingDirectory(self.window, "Select export folder")
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
            QMessageBox.critical(
                self.window,
                "Export failed",
                f"Could not access output folder:\n{exc}",
            )
            return

        target_labels_for_prompt = {old_tid: names[new_tid] for old_tid, new_tid in tid_remap.items()}
        for ctx in contexts:
            dataset_sources = model.sources.get(ctx["dataset_id"], [])
            mapped_source_ids = set(ctx["mapping"].keys())
            unmapped_sources = [src for src in dataset_sources if src.class_id not in mapped_source_ids]
            if unmapped_sources:
                msg = (
                    f"Dataset '{ctx['base_name']}' has unmapped classes that will be dropped:\n"
                    + "\n".join(f"  {src.class_name} ({src.class_id})" for src in unmapped_sources)
                )
                QMessageBox.warning(self.window, "Unmapped classes", msg)
                choice_enabled, choice_tid = self._choose_fallback_target(target_labels_for_prompt)
                if choice_enabled is None:
                    return
                if choice_enabled and choice_tid is not None:
                    remapped_tid = tid_remap.get(choice_tid)
                    if remapped_tid is None:
                        QMessageBox.warning(
                            self.window,
                            "Fallback unavailable",
                            "Selected fallback target is no longer available.",
                        )
                        return
                    label = names[remapped_tid] if 0 <= remapped_tid < len(names) else f"class_{remapped_tid}"
                    ctx["fallback_target_id"] = remapped_tid
                    ctx["fallback_label"] = label
                    ctx["fallback_enabled"] = True
                else:
                    ctx["fallback_target_id"] = None
                    ctx["fallback_label"] = ""
                    ctx["fallback_enabled"] = False
            else:
                ctx["fallback_target_id"] = None
                ctx["fallback_label"] = ""
                ctx["fallback_enabled"] = False

        quota_map = {new_tid: self.merge_controller.get_target_quota(old_tid) for old_tid, new_tid in tid_remap.items()}
        images_per_target = {new_tid: 0 for new_tid in tid_remap.values()}
        for ctx in contexts:
            ftid = ctx["fallback_target_id"]
            if ftid is not None and ftid not in images_per_target:
                images_per_target[ftid] = 0
        edge_limits = dict(self.merge_controller.model.edge_limits)
        edge_usage = defaultdict(int)

        total_images = 0
        for ctx in contexts:
            dm = ctx["model"]
            for info in dm.splits.values():
                images = info.get("images") if info else None
                if images:
                    total_images += len(images)
        if total_images <= 0:
            QMessageBox.warning(self.window, "No images", "No images available to export.")
            return

        use_dataset_subdirs = len(contexts) > 1

        manifest_writer = ManifestWriter(dest_root, dry_run)
        manifest_writer._meta["datasets"] = [
            {"id": ctx["dataset_id"], "name": ctx["base_name"]} for ctx in contexts
        ]
        manifest_writer._meta["target_names"] = names
        manifest_writer._meta["export_mode"] = "dry_run" if dry_run else "export"
        if len(contexts) == 1:
            ctx = contexts[0]
            manifest_writer._meta["dataset_id"] = ctx["dataset_id"]
            manifest_writer._meta["fallback_target"] = ctx["fallback_target_id"]
            manifest_writer._meta["fallback_label"] = ctx["fallback_label"]
            manifest_writer._meta["fallback_enabled"] = ctx["fallback_enabled"]
        else:
            manifest_writer._meta["dataset_id"] = "multiple"
            manifest_writer._meta["fallback_target"] = None
            manifest_writer._meta["fallback_label"] = None
            manifest_writer._meta["fallback_enabled"] = any(ctx["fallback_enabled"] for ctx in contexts)
            manifest_writer._meta["fallback_map"] = {
                ctx["dataset_id"]: {
                    "target": ctx["fallback_target_id"],
                    "label": ctx["fallback_label"],
                    "enabled": ctx["fallback_enabled"],
                }
                for ctx in contexts
            }

        scan_progress = QProgressDialog("Analyzing dataset...", "Cancel", 0, total_images, self.window)
        scan_progress.setWindowModality(Qt.WindowModal)
        scan_progress.setMinimumDuration(0)
        scan_progress.setAutoReset(False)
        scan_progress.setAutoClose(False)

        scan_count = 0
        scan_cancelled = False
        per_split_target_counts: Dict[str, defaultdict[int]] = {}
        manifest_map: Dict[tuple[str, str, str], dict] = {}
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

        def _label_relative_path(src_txt: Path, labels_dir: Optional[Path], rel_img: Path) -> Path:
            try:
                return src_txt.relative_to(labels_dir) if labels_dir else rel_img.with_suffix(".txt")
            except ValueError:
                return rel_img.with_suffix(".txt")

        try:
            for ctx in contexts:
                dataset_id = ctx["dataset_id"]
                base_name = ctx["base_name"]
                dm = ctx["model"]
                dataset_mapping = ctx["mapping"]
                fallback_target_id = ctx["fallback_target_id"]
                for split_name in dm.ordered_splits():
                    info = dm.splits.get(split_name)
                    if not info:
                        continue
                    images_dir: Path = info["images_dir"]
                    labels_dir = info.get("labels_dir")
                    per_split_target_counts.setdefault(split_name, defaultdict(int))
                    sorted_images = sorted(
                        info["images"],
                        key=lambda p: _relative_image_path(p, images_dir).as_posix(),
                    )
                    for img_path in sorted_images:
                        if scan_cancelled:
                            break

                        scan_count += 1
                        scan_progress.setLabelText(
                            f"Analyzing {dataset_id}:{split_name} ({scan_count}/{total_images})"
                        )
                        scan_progress.setValue(scan_count)
                        QApplication.processEvents()
                        if scan_progress.wasCanceled():
                            scan_cancelled = True
                            break

                        rel_img = _relative_image_path(img_path, images_dir)
                        rel_key = (dataset_id, split_name, rel_img.as_posix())

                        src_txt = labels_for_image(img_path, labels_dir, images_dir)
                        boxes = read_yolo_txt(src_txt)
                        boxes, changed = self._sanitize_boxes_for_image(img_path, boxes)
                        record = {
                            "dataset_id": dataset_id,
                            "dataset_name": base_name,
                            "split": split_name,
                            "image": rel_img.as_posix(),
                            "source_image": str(img_path),
                            "label_rel": _label_relative_path(src_txt, labels_dir, rel_img).as_posix(),
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
                        if changed:
                            record["notes"].append("Adjusted extremely small boxes during analysis")
                        image_boxes_cache[img_path] = boxes
                        self._label_cache[img_path] = boxes
                        self._image_class_sets[img_path] = {b.cls for b in boxes}

                        mapped_targets: set[int] = set()
                        source_classes_present: Dict[int, set[int]] = {}
                        mapped_count = 0
                        fallback_count_local = 0
                        has_unmapped = False

                        for box in boxes:
                            tgt_id = dataset_mapping.get(box.cls)
                            if tgt_id is None:
                                if fallback_target_id is not None:
                                    tgt_id = fallback_target_id
                                    fallback_count_local += 1
                                else:
                                    dropped_unmapped += 1
                                    has_unmapped = True
                                    continue
                            mapped_targets.add(tgt_id)
                            source_classes_present.setdefault(tgt_id, set()).add(box.cls)
                            mapped_count += 1

                        record["fallback_boxes"] = fallback_count_local
                        record["mapped_targets"] = sorted(mapped_targets)
                        record["source_classes"] = sorted(
                            {cls for cls_set in source_classes_present.values() for cls in cls_set}
                        )

                        if fallback_count_local:
                            fallback_boxes_relabelled += fallback_count_local
                            ctx["fallback_box_total"] += fallback_count_local

                        if has_unmapped and fallback_target_id is None:
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
                            per_split_target_counts[split_name][tid] += 1
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
                QMessageBox.warning(
                    self.window,
                    "Export canceled",
                    f"Analysis canceled and manifest failed: {exc}",
                )
                return
            QMessageBox.information(
                self.window,
                "Export canceled",
                f"Export was canceled during analysis.\nManifest written: {manifest_file}",
            )
            return

        all_splits = sorted({split for ctx in contexts for split in ctx["model"].ordered_splits()})
        quota_per_split: Dict[str, Dict[int, int]] = {}
        for tid, quota in quota_map.items():
            if quota is None:
                continue
            available_by_split: Dict[str, int] = {}
            for split_name in all_splits:
                split_counts = per_split_target_counts.get(split_name)
                available_by_split[split_name] = split_counts.get(tid, 0) if split_counts else 0
            total_available = sum(available_by_split.values())
            if total_available <= 0:
                continue
            if quota >= total_available:
                for split_name, count in available_by_split.items():
                    if count:
                        quota_per_split.setdefault(split_name, {})[tid] = count
                continue
            alloc: Dict[str, int] = {}
            remainders: List[tuple[float, str, int]] = []
            for split_name, count in available_by_split.items():
                if count <= 0:
                    continue
                exact = quota * count / total_available
                base = min(int(exact), count)
                alloc[split_name] = base
                remainders.append((exact - base, split_name, count))
            assigned = sum(alloc.values())
            remaining = min(quota, total_available) - assigned
            remainders.sort(key=lambda x: x[0], reverse=True)
            for _, split_name, count in remainders:
                if remaining <= 0:
                    break
                if alloc.get(split_name, 0) < count:
                    alloc[split_name] = alloc.get(split_name, 0) + 1
                    remaining -= 1
            for split_name, count in alloc.items():
                if count > 0:
                    quota_per_split.setdefault(split_name, {})[tid] = min(count, available_by_split[split_name])

        manifest_writer._meta["queued_images"] = len(manifest_map)

        images_per_split_target = defaultdict(int)
        export_keys = sorted(manifest_map.keys(), key=lambda key: (key[1], key[2], key[0]))
        export_progress = None
        if export_keys:
            export_progress = QProgressDialog(
                "Exporting merged dataset...",
                "Cancel",
                0,
                len(export_keys),
                self.window,
            )
            export_progress.setWindowModality(Qt.WindowModal)
            export_progress.setMinimumDuration(0)
            export_progress.setAutoReset(False)
            export_progress.setAutoClose(False)

        export_cancelled = False

        for index, key in enumerate(export_keys, start=1):
            if export_cancelled:
                break
            dataset_id, split_name, rel_img_str = key
            record = manifest_map[key]
            ctx = context_lookup[dataset_id]
            dataset_mapping = ctx["mapping"]
            fallback_target_id = ctx["fallback_target_id"]
            fallback_label = ctx["fallback_label"]

            if export_progress:
                export_progress.setLabelText(
                    f"Exporting {dataset_id}:{split_name} ({index}/{len(export_keys)})"
                )
                export_progress.setValue(index)
                QApplication.processEvents()
                if export_progress.wasCanceled():
                    export_cancelled = True
                    break

            rel_img = Path(rel_img_str)
            label_rel = Path(record["label_rel"])
            img_path = Path(record["source_image"])
            info = ctx["model"].splits.get(split_name)
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

            boxes_by_target: Dict[int, List[Box]] = {}
            per_target_sources: Dict[int, set[int]] = {}
            fallback_count_local = 0
            has_unmapped = False
            mapped_count = 0

            for box in boxes:
                tgt_id = dataset_mapping.get(box.cls)
                if tgt_id is None:
                    if fallback_target_id is not None:
                        tgt_id = fallback_target_id
                        fallback_count_local += 1
                    else:
                        dropped_unmapped += 1
                        has_unmapped = True
                        continue
                boxes_by_target.setdefault(tgt_id, []).append(
                    Box(tgt_id, box.cx, box.cy, box.w, box.h)
                )
                per_target_sources.setdefault(tgt_id, set()).add(box.cls)
                mapped_count += 1

            record["fallback_boxes"] = fallback_count_local
            record["mapped_targets"] = sorted(boxes_by_target.keys())
            record["source_classes"] = sorted(
                {cls for cls_set in per_target_sources.values() for cls in cls_set}
            )
            record["skipped_targets"] = {}

            if has_unmapped and fallback_target_id is None:
                skipped_images_unmapped += 1
                dropped_with_unmapped_image += mapped_count
                record["status"] = "skipped_unmapped"
                record.setdefault("notes", []).append("Dropped due to unmapped classes (post-analysis)")
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
                    quota_split = quota_per_split.get(split_name, {}).get(tgt_id)
                    if quota_split is not None and images_per_split_target[(split_name, tgt_id)] >= quota_split:
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
                record.setdefault("notes", []).append("All mapped targets exceeded quotas or limits")
                manifest_writer.append(record)
                manifest_map.pop(key, None)
                continue

            record["included_targets"] = sorted(tid for tid, _ in included_targets_info)

            def _apply_inclusion_counts():
                for tgt_id, src_cls_set in included_targets_info:
                    images_per_target[tgt_id] = images_per_target.get(tgt_id, 0) + 1
                    images_per_split_target[(split_name, tgt_id)] += 1
                    for src_cls in src_cls_set:
                        edge_usage[(dataset_id, src_cls)] += 1

            if dry_run:
                _apply_inclusion_counts()
                record["status"] = "dry_run"
                record.setdefault("notes", []).append("Dry run only (no files written)")
                copied += 1
                written += 1
                manifest_writer.append(record)
                manifest_map.pop(key, None)
                continue

            dest_images_dir = dest_root / split_name / "images"
            dest_labels_dir = dest_root / split_name / "labels"
            if use_dataset_subdirs:
                dest_images_dir = dest_images_dir / dataset_id
                dest_labels_dir = dest_labels_dir / dataset_id
            out_img = dest_images_dir / rel_img
            out_img.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(img_path, out_img)
            except Exception as exc:
                errors.append(f"Image copy failed ({img_path.name}): {exc}")
                record["status"] = "copy_failed"
                record.setdefault("notes", []).append(str(exc))
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
                record.setdefault("notes", []).append(str(exc))
                manifest_writer.append(record)
                manifest_map.pop(key, None)
                continue
            written += 1
            record["status"] = "exported"

            _apply_inclusion_counts()
            manifest_writer.append(record)
            manifest_map.pop(key, None)

        if export_progress:
            export_progress.close()

        if export_cancelled:
            for key, record in list(manifest_map.items()):
                if record.get("status") == "queued":
                    record["status"] = "not_processed"
                    record.setdefault("notes", []).append("Export canceled before processing")
                manifest_writer.append(record)
                manifest_map.pop(key, None)
            try:
                manifest_file = manifest_writer.finalize()
            except Exception as exc:
                QMessageBox.warning(
                    self.window,
                    "Export canceled",
                    f"Export canceled and manifest failed: {exc}",
                )
                return
            QMessageBox.information(
                self.window,
                "Export canceled",
                f"Export was canceled. Some files may have been written.\nManifest written: {manifest_file}",
            )
            return

        for key, record in list(manifest_map.items()):
            if record.get("status") == "queued":
                record["status"] = "not_processed"
                record.setdefault("notes", []).append("Image was queued but not processed")
            manifest_writer.append(record)
            manifest_map.pop(key, None)

        if not dry_run:
            try:
                self._write_export_yaml(dest_root, names)
            except Exception as exc:
                manifest_writer.abort()
                QMessageBox.critical(
                    self.window,
                    "Export failed",
                    f"Could not write data.yaml:\n{exc}",
                )
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
        manifest_writer._meta["quota_per_split"] = {
            split_name: dict(values) for split_name, values in quota_per_split.items()
        }

        try:
            manifest_file = manifest_writer.finalize()
        except Exception as exc:
            QMessageBox.critical(
                self.window,
                "Export failed",
                f"Could not write manifest:\n{exc}",
            )
            return

        summary = [f"Images copied: {copied}", f"Label files written: {written}"]
        if dry_run:
            summary.insert(0, "Dry run only: no files were written.")
        if skipped_images_unmapped:
            summary.append(
                f"Images skipped (unmapped classes present): {skipped_images_unmapped}"
            )
        if skipped_images_no_mapped:
            summary.append(
                f"Images skipped (no mapped boxes): {skipped_images_no_mapped}"
            )
        if skipped_images_quota:
            summary.append(
                f"Images skipped (quota exhausted): {skipped_images_quota}"
            )
        if dropped_unmapped:
            summary.append(f"Boxes skipped (no mapping): {dropped_unmapped}")
        if dropped_with_unmapped_image:
            summary.append(
                f"Boxes skipped (discarded with unmapped image): {dropped_with_unmapped_image}"
            )
        if dropped_by_quota:
            summary.append(f"Boxes skipped (quota reached): {dropped_by_quota}")
        if dropped_edge_limit:
            summary.append(f"Boxes skipped (edge limit reached): {dropped_edge_limit}")
        fallback_lines = []
        for ctx in contexts:
            ftid = ctx["fallback_target_id"]
            if ftid is not None and ctx["fallback_box_total"] > 0:
                label = ctx["fallback_label"] or ""
                fallback_lines.append(
                    f"Boxes remapped to fallback [{ftid}] {label} ({ctx['base_name']}): {ctx['fallback_box_total']}"
                )
        if fallback_lines:
            summary.extend(fallback_lines)
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
            self.window,
            "Export complete" if not dry_run else "Dry run complete",
            "\n    ".join(summary),
        )
