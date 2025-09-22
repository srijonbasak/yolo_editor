
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any
from PySide6.QtCore import Qt, QThread, QObject, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
    QFileDialog,
    QHeaderView,
)

from .image_view import ImageView, Box as ViewBox

from ..core.dataset_resolver import resolve_dataset, DatasetModel
from ..core.yolo_io import Box, read_yolo_txt, write_yolo_txt, labels_for_image, imread_unicode
from ..core.merge_model import MergePlan, TargetClass, CopyMode, CollisionPolicy, SplitStrategy, BalanceMode
from ..core.merge_selector import build_edge_index, select_with_quotas
from ..core.merger import merge_execute
from ..core.report import write_report
from ..core.multi_repo import MultiRepo


def _collect_source_classes(dataset_id: str, dm: DatasetModel, source_class_type):
    from collections import defaultdict

    per_imgs = defaultdict(int)
    per_boxes = defaultdict(int)
    for info in dm.splits.values():
        labels_dir = info.get("labels_dir")
        for img in info.get("images", []):
            seen = set()
            for b in read_yolo_txt(labels_for_image(img, labels_dir)):
                per_boxes[b.cls] += 1
                seen.add(b.cls)
            for c in seen:
                per_imgs[c] += 1

    cls_ids = sorted(set(per_imgs.keys()) | set(per_boxes.keys()))
    if not cls_ids and dm.names:
        cls_ids = list(range(len(dm.names)))

    source_classes = []
    for cls_id in cls_ids:
        name = dm.names[cls_id] if 0 <= cls_id < len(dm.names) else str(cls_id)
        source_classes.append(
            source_class_type(
                dataset_id=dataset_id,
                class_id=cls_id,
                class_name=name,
                images=per_imgs.get(cls_id, 0),
                boxes=per_boxes.get(cls_id, 0),
            )
        )
    return source_classes


class _DatasetSummaryWorker(QObject):
    finished = Signal(str, object)
    failed = Signal(str, str)

    def __init__(self, dataset_id: str, dm: DatasetModel, source_class_type):
        super().__init__()
        self._dataset_id = dataset_id
        self._dm = dm
        self._source_class_type = source_class_type

    def run(self):
        try:
            classes = _collect_source_classes(self._dataset_id, self._dm, self._source_class_type)
        except Exception as exc:
            self.failed.emit(self._dataset_id, str(exc))
        else:
            self.finished.emit(self._dataset_id, classes)


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
        self.names: List[str] = []
        self.idx: int = -1

        # Merge designer bookkeeping
        self.merge_dataset_store: Dict[str, Dict[str, Any]] = {}
        self.merge_dataset_order: List[str] = []
        self._source_class_type = None
        self._dataset_threads: Dict[str, QThread] = {}

        self.tabs = QTabWidget(self)
        self.setCentralWidget(self.tabs)

        # --- Label Editor tab ---
        self.editor_root = QWidget()
        self._build_editor_ui(self.editor_root)
        self.tabs.addTab(self.editor_root, "Label Editor")

        # (optional) Merge tab - only if imports are available
        try:
            from .merge_designer.canvas import MergeCanvas
            from .merge_designer.controller import MergeController, SourceClass
            from .merge_designer.palette import MergePalette

            self._source_class_type = SourceClass
            self.merge_ctrl = MergeController()
            self.merge_canvas = MergeCanvas(self.merge_ctrl)
            self.merge_palette = MergePalette(
                on_spawn_dataset=self._spawn_dataset_node,
                on_spawn_target_class=self._spawn_target_node,
            )
            self.merge_palette.requestLoadDataset.connect(self._on_merge_palette_load)
            self.merge_palette.requestExportMerged.connect(self._on_merge_palette_export)
            sp = QSplitter(Qt.Orientation.Horizontal)
            sp.addWidget(self.merge_palette)
            sp.addWidget(self.merge_canvas)
            sp.setStretchFactor(1, 1)
            self.tabs.addTab(sp, "Merge Designer")
        except Exception:
            # Keep attributes defined for type checking even if merge UI unavailable
            self.merge_ctrl = None
            self.merge_canvas = None
            self.merge_palette = None

        self._build_menu()

    # ---------------- UI ----------------

    def _build_editor_ui(self, parent: QWidget):
        splitter = QSplitter(Qt.Orientation.Horizontal, parent)

        # left
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(6, 6, 6, 6)
        self.split_combo = QComboBox()
        self.split_combo.currentTextChanged.connect(self._on_split_changed)
        self.file_tree = QTreeWidget()
        self.file_tree.setHeaderLabels(["Image Files"])
        self.file_tree.itemClicked.connect(self._on_file_clicked)
        lv.addWidget(QLabel("Split:"))
        lv.addWidget(self.split_combo)
        lv.addWidget(self.file_tree, 1)

        # center
        center = QWidget()
        cv = QVBoxLayout(center)
        cv.setContentsMargins(6, 6, 6, 6)
        bar = QHBoxLayout()
        self.lbl_ds = QLabel("Dataset: -")
        self.class_combo = QComboBox()
        self.class_combo.currentIndexChanged.connect(self._on_class_changed)
        self.btn_save = QPushButton("Save")
        self.btn_save.clicked.connect(self._save_labels)
        bar.addWidget(self.lbl_ds, 1)
        bar.addWidget(QLabel("Class:"))
        bar.addWidget(self.class_combo)
        bar.addStretch(1)
        bar.addWidget(self.btn_save)
        self.view = ImageView()
        self.view.set_status_sink(lambda msg: self.statusBar().showMessage(msg, 3000))
        self.view.boxesChanged.connect(self._on_boxes_changed)
        self.view.requestPrev.connect(lambda: self._open_index(max(0, self.idx - 1)))
        self.view.requestNext.connect(lambda: self._open_index(min(len(self.images) - 1, self.idx + 1)))
        cv.addLayout(bar)
        cv.addWidget(self.view, 1)

        # right
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(6, 6, 6, 6)
        rv.addWidget(QLabel("Labels (YOLO)"))
        self.tbl = QTableWidget(0, 6)
        self.tbl.setHorizontalHeaderLabels(["name", "id", "cx", "cy", "w", "h"])
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tbl.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        rv.addWidget(self.tbl, 3)
        rv.addWidget(QLabel("Dataset Stats"))
        self.stats = QListWidget()
        rv.addWidget(self.stats, 2)

        splitter.addWidget(left)
        splitter.addWidget(center)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)

        lay = QVBoxLayout(parent)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(splitter)

    def _build_menu(self):
        m = self.menuBar().addMenu("&File")

        act_root = QAction("Open Dataset &Root...", self)
        act_yaml = QAction("Open Dataset &YAML...", self)
        act_quit = QAction("&Quit", self)

        act_root.triggered.connect(self._open_root)
        act_yaml.triggered.connect(self._open_yaml)
        act_quit.triggered.connect(self.close)

        m.addAction(act_root)
        m.addAction(act_yaml)
        m.addSeparator()
        m.addAction(act_quit)

        e = self.menuBar().addMenu("&Edit")
        a_save = QAction("&Save Labels", self)
        a_save.setShortcut("S")
        a_prev = QAction("&Prev", self)
        a_prev.setShortcut(Qt.Key_Left)
        a_next = QAction("&Next", self)
        a_next.setShortcut(Qt.Key_Right)
        a_save.triggered.connect(self._save_labels)
        a_prev.triggered.connect(lambda: self._open_index(max(0, self.idx - 1)))
        a_next.triggered.connect(lambda: self._open_index(min(len(self.images) - 1, self.idx + 1)))
        for action in (a_save, a_prev, a_next):
            e.addAction(action)

        tools = self.menuBar().addMenu("&Tools")
        act_diag = QAction("Show &Diagnostics...", self)
        act_diag.triggered.connect(self._show_diagnostics)
        tools.addAction(act_diag)

    # ---------------- Open handlers ----------------

    def _open_root(self):
        d = QFileDialog.getExistingDirectory(self, "Open Dataset Root")
        if not d:
            return
        dm = resolve_dataset(Path(d))
        if not dm.splits:
            QMessageBox.warning(self, "Not a dataset", "Could not detect train/val/test/valid/eval in this folder.")
            return
        self._load_dataset(Path(d).name, dm, Path(d))

    def _open_yaml(self):
        f, _ = QFileDialog.getOpenFileName(self, "Open data.yaml", "", "YAML (*.yaml *.yml)")
        if not f:
            return
        dm = resolve_dataset(Path(f))
        if not dm.splits:
            QMessageBox.warning(self, "Invalid YAML", "Could not resolve any split paths from this YAML.")
            return
        self._load_dataset(Path(f).parent.name, dm, Path(f))

    def _load_dataset(self, name: str, dm: DatasetModel, source_path: Path):
        dataset_id = self._register_merge_dataset(name, dm, source_path)
        self.dm = dm
        self.ds_name = dataset_id
        self.lbl_ds.setText(f"Dataset: {dataset_id}")
        self.names = dm.names or []
        self.class_combo.blockSignals(True)
        self.class_combo.clear()
        if self.names:
            for n in self.names:
                self.class_combo.addItem(n)
        else:
            for i in range(100):
                self.class_combo.addItem(str(i))
        self.class_combo.blockSignals(False)
        self.view.set_current_class(self.class_combo.currentIndex(), self.names)

        # splits
        self.split_combo.blockSignals(True)
        self.split_combo.clear()
        for s in dm.ordered_splits():
            self.split_combo.addItem(s)
        self.split_combo.blockSignals(False)

        # default split
        if dm.ordered_splits():
            self._on_split_changed(dm.ordered_splits()[0])

        self._refresh_merge_palette_list()

    # ---------------- Split / file tree ----------------

    def _on_split_changed(self, split: str):
        if not self.dm or split not in self.dm.splits:
            return
        sp = self.dm.splits[split]
        self.split = split
        self.images = sp["images"]
        self.labels_dir = sp["labels_dir"]
        self._populate_file_tree()
        self._compute_stats_and_show()
        if self.images:
            self._open_index(0)

    def _populate_file_tree(self):
        self.file_tree.clear()
        root = QTreeWidgetItem(["(images)"])
        for p in self.images:
            QTreeWidgetItem(root, [p.name])
        self.file_tree.addTopLevelItem(root)
        self.file_tree.expandAll()

    def _on_file_clicked(self, item: QTreeWidgetItem):
        if not item.parent():
            return
        idx = item.parent().indexOfChild(item)
        self._open_index(idx)

    # ---------------- Image / labels ----------------

    def _open_index(self, i: int):
        if i < 0 or i >= len(self.images):
            return
        img_path = self.images[i]
        img = imread_unicode(img_path)
        if img is None:
            self.statusBar().showMessage(f"Failed to load: {img_path.name}", 4000)
            return
        self.view.show_image_bgr(img_path, img)
        core_boxes = read_yolo_txt(labels_for_image(img_path, self.labels_dir))
        self.view.clear_boxes()
        view_boxes = [ViewBox(cls=b.cls, cx=b.cx, cy=b.cy, w=b.w, h=b.h) for b in core_boxes]
        for vb in view_boxes:
            self.view.add_box_norm(vb)
        self._fill_table(view_boxes)
        self.idx = i
        self._highlight_tree_row(i)
        self.statusBar().showMessage(f"{img_path.name}  ({i + 1}/{len(self.images)})", 4000)

    def _save_labels(self):
        if self.idx < 0 or not self.images:
            return
        img_path = self.images[self.idx]
        txt = labels_for_image(img_path, self.labels_dir)
        boxes = self.view.get_boxes_as_norm()
        write_yolo_txt(txt, [Box(int(b.cls), b.cx, b.cy, b.w, b.h) for b in boxes])
        self._fill_table(boxes)
        self.statusBar().showMessage(f"Saved: {txt}", 4000)
        self._compute_stats_and_show()

    def _fill_table(self, boxes: List[ViewBox]):
        self.tbl.setRowCount(0)
        for b in boxes:
            r = self.tbl.rowCount()
            self.tbl.insertRow(r)
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
        if not root:
            return
        for i in range(root.childCount()):
            ch = root.child(i)
            ch.setSelected(i == idx)
        self.file_tree.scrollToItem(root.child(idx))

    def _on_class_changed(self, idx: int):
        self.view.set_current_class(idx, self.names)
        nm = self.names[idx] if 0 <= idx < len(self.names) else str(idx)
        self.statusBar().showMessage(f"Current class -> {nm} [{idx}]", 3000)

    # ---------------- Stats ----------------

    def _compute_stats_and_show(self):
        from collections import defaultdict

        per_imgs = defaultdict(int)
        per_boxes = defaultdict(int)
        for img in self.images:
            seen = set()
            for b in read_yolo_txt(labels_for_image(img, self.labels_dir)):
                per_boxes[b.cls] += 1
                seen.add(b.cls)
            for c in seen:
                per_imgs[c] += 1
        self.stats.clear()
        self.stats.addItem(f"Total images: {len(self.images)}")
        ids = sorted(set(per_imgs.keys()) | set(per_boxes.keys()))
        for cid in ids:
            nm = self.names[cid] if 0 <= cid < len(self.names) else str(cid)
            self.stats.addItem(f"[{cid}] {nm}: {per_imgs.get(cid, 0)} imgs / {per_boxes.get(cid, 0)} boxes")

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
        QMessageBox.information(self, "Dataset diagnostics", '\n'.join(lines))

    # ---------------- Merge Designer ----------------

    def _spawn_dataset_node(self, dataset_name: str):
        if not self.merge_canvas or not self.merge_ctrl:
            return
        entry = self.merge_dataset_store.get(dataset_name)
        if not entry:
            QMessageBox.warning(self, "Dataset not found", f"Dataset '{dataset_name}' is not registered.")
            return
        if dataset_name in self.merge_canvas.nodes:
            self.statusBar().showMessage(f"Dataset '{dataset_name}' is already on the canvas.", 3000)
            return

        classes = entry.get('source_classes') or []
        ready = bool(entry.get('summary_ready') and classes)
        if ready:
            self.merge_ctrl.upsert_dataset(dataset_name, classes)
        self.merge_canvas.spawn_dataset_node(dataset_name, classes, loading=not ready)
        self._ensure_dataset_summary_async(dataset_name)
    def _spawn_target_node(self, name: str, quota: int | None = None):
        if not self.merge_canvas or not self.merge_ctrl:
            return
        target_id = self.merge_ctrl.add_target_class(name, quota)
        self.merge_canvas.spawn_target_node(target_id, name, quota)

    def _on_merge_palette_load(self):
        if not self.merge_palette:
            return
        menu = QMenu(self.merge_palette.btn_load)
        act_root = menu.addAction("Add dataset folder...")
        act_yaml = menu.addAction("Add dataset YAML...")
        pos = self.merge_palette.btn_load.mapToGlobal(self.merge_palette.btn_load.rect().bottomLeft())
        chosen = menu.exec(pos)
        if chosen == act_root:
            self._open_root()
        elif chosen == act_yaml:
            self._open_yaml()

    def _on_merge_palette_export(self):
        self._export_merge()

    def _export_merge(self):
        if not self.merge_canvas or not self.merge_ctrl:
            QMessageBox.information(self, "Merge unavailable", "Merge designer components are not available on this platform.")
            return
        if not self.merge_ctrl.model.targets:
            QMessageBox.warning(self, "No targets", "Create at least one target class before exporting.")
            return
        if not self.merge_ctrl.model.edges:
            QMessageBox.warning(self, "No connections", "Wire source classes to targets before exporting.")
            return

        dataset_ids = list(self.merge_canvas.nodes.keys())
        if not dataset_ids:
            QMessageBox.warning(self, "No datasets", "Add at least one dataset to the canvas before exporting.")
            return
        for ds_id in dataset_ids:
            if ds_id not in self.merge_dataset_store:
                QMessageBox.warning(self, "Dataset missing", f"Dataset '{ds_id}' is no longer available. Reload it before exporting.")
                return

        out_dir = QFileDialog.getExistingDirectory(self, "Select output folder")
        if not out_dir:
            return
        output_path = Path(out_dir)

        target_classes = [TargetClass(index=tid, name=tgt.class_name) for tid, tgt in sorted(self.merge_ctrl.model.targets.items())]
        mapping = {edge.source_key: edge.target_id for edge in self.merge_ctrl.model.edges}
        if not mapping:
            QMessageBox.warning(self, "No mappings", "Define at least one mapping before exporting.")
            return
        target_quota = {tid: tgt.quota_images for tid, tgt in self.merge_ctrl.model.targets.items() if tgt.quota_images}
        edge_limit = dict(self.merge_ctrl.model.edge_limits)

        plan = MergePlan(
            name="merged",
            output_dir=output_path,
            target_classes=target_classes,
            mapping=mapping,
            target_quota=target_quota,
            edge_limit=edge_limit,
            balance_mode=BalanceMode.EQUAL,
            random_seed=1337,
            split_strategy=SplitStrategy.KEEP,
            copy_mode=CopyMode.HARDLINK,
            collision_policy=CollisionPolicy.RENAME,
            drop_empty_images=True,
            target_train_name="train",
            target_val_name="val",
            target_test_name="test",
        )

        repo = MultiRepo()
        for ds_id in dataset_ids:
            entry = self.merge_dataset_store[ds_id]
            repo.add(ds_id, entry['root'], yaml_path=entry.get('yaml_path'), display_name=ds_id)

        per_target = build_edge_index(plan, repo)
        selection = select_with_quotas(plan, per_target)
        if not selection.selected_images:
            QMessageBox.warning(self, "Nothing to export", "No images satisfy the current quotas/limits. Adjust settings and try again.")
            return

        warnings = selection.warnings[:]
        if warnings:
            QMessageBox.information(self, "Quota warnings", '\n'.join(warnings))

        progress = QProgressDialog("Merging datasets...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.show()

        cancelled = False

        def progress_cb(prog):
            progress.setMaximum(max(prog.total, 1))
            progress.setValue(prog.value)
            QApplication.processEvents()
            if progress.wasCanceled():
                raise RuntimeError("Cancelled")

        try:
            merge_execute(
                plan=plan,
                sources=repo,
                progress_cb=progress_cb,
                cancel=None,
                selection=selection.selected_images,
            )
            write_report(plan.output_dir, plan, selection)
            progress.setValue(progress.maximum())
        except RuntimeError:
            cancelled = True
        except Exception as exc:
            progress.cancel()
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        finally:
            progress.close()

        if cancelled:
            QMessageBox.information(self, "Merge cancelled", "Export cancelled by user.")
            return

        msg_lines = [
            "Merged dataset written to:",
            f"  {plan.output_dir}",
            "",
            "Report:",
            f"  {plan.output_dir / 'reports' / 'merge_report.json'}",
        ]
        if warnings:
            msg_lines.append("")
            msg_lines.append("Warnings:")
            msg_lines.extend(f"  - {w}" for w in warnings)
        QMessageBox.information(self, "Export complete", '\n'.join(msg_lines))

    # ---------------- Merge helpers ----------------

    def _register_merge_dataset(self, base_name: str, dm: DatasetModel, source_path: Path) -> str:
        resolved = source_path.resolve()
        yaml_path = resolved if resolved.is_file() else None
        root = resolved.parent if yaml_path else resolved
        dataset_id = base_name
        counter = 2
        while dataset_id in self.merge_dataset_store:
            existing = self.merge_dataset_store[dataset_id]
            if existing['root'] == root and existing.get('yaml_path') == yaml_path:
                break
            dataset_id = f"{base_name} ({counter})"
            counter += 1

        entry = {
            'dm': dm,
            'root': root,
            'yaml_path': yaml_path,
            'source_path': resolved,
            'source_classes': None,
            'summary_ready': False,
            'summary_loading': False,
        }
        self.merge_dataset_store[dataset_id] = entry

        if dataset_id not in self.merge_dataset_order:
            self.merge_dataset_order.append(dataset_id)
        else:
            stored = self.merge_dataset_store[dataset_id]
            stored['source_classes'] = None
            stored['summary_ready'] = False

        if self.merge_canvas and dataset_id in self.merge_canvas.nodes:
            if entry['summary_ready'] and entry['source_classes']:
                self.merge_ctrl.upsert_dataset(dataset_id, entry['source_classes'])
                self.merge_canvas.update_dataset_stats(dataset_id, entry['source_classes'])
            else:
                self.merge_canvas.update_dataset_stats(dataset_id, [])

        self._ensure_dataset_summary_async(dataset_id)
        return dataset_id
    def _refresh_merge_palette_list(self):
        if self.merge_palette:
            self.merge_palette.populate(self.merge_dataset_order)

    def _ensure_dataset_summary_async(self, dataset_id: str):
        if not self._source_class_type:
            return
        entry = self.merge_dataset_store.get(dataset_id)
        if not entry or entry.get('summary_ready') or entry.get('summary_loading'):
            return

        worker = _DatasetSummaryWorker(dataset_id, entry['dm'], self._source_class_type)
        thread = QThread(self)
        entry['summary_loading'] = True
        entry['worker'] = worker

        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_dataset_summary_ready)
        worker.failed.connect(self._on_dataset_summary_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda did=dataset_id: self._dataset_threads.pop(did, None))
        thread.finished.connect(lambda did=dataset_id: self.merge_dataset_store.get(did, {}).pop('worker', None))

        self._dataset_threads[dataset_id] = thread
        if self.statusBar():
            self.statusBar().showMessage(f"Scanning dataset '{dataset_id}'...", 2000)
        thread.start()

    def _on_dataset_summary_ready(self, dataset_id: str, classes: object):
        entry = self.merge_dataset_store.get(dataset_id)
        if not entry:
            return
        entry['summary_loading'] = False
        entry['summary_ready'] = True
        entry['source_classes'] = classes

        if self.merge_ctrl:
            self.merge_ctrl.upsert_dataset(dataset_id, classes)
        if self.merge_canvas:
            self.merge_canvas.update_dataset_stats(dataset_id, classes)
        self.statusBar().showMessage(f"Dataset '{dataset_id}' statistics ready", 3000)

    def _on_dataset_summary_failed(self, dataset_id: str, message: str):
        entry = self.merge_dataset_store.get(dataset_id)
        if entry:
            entry['summary_loading'] = False
            entry['summary_ready'] = False
        if self.merge_canvas:
            self.merge_canvas.set_dataset_error(dataset_id, message)
        self.statusBar().showMessage(f"Failed to summarize '{dataset_id}': {message}", 5000)
        QMessageBox.warning(self, "Dataset summary failed", message)
        if self.merge_palette:
            self.merge_palette.populate(self.merge_dataset_order)

