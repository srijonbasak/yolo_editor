from __future__ import annotations
from pathlib import Path
from typing import List
import cv2

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QMainWindow, QFileDialog, QMessageBox, QWidget, QSplitter, QVBoxLayout,
    QLabel, QComboBox, QHBoxLayout, QPushButton, QStatusBar
)

from ..core.repo import DatasetRepository
from ..core.yolo_io import parse_label_file, save_label_file
from .image_view import ImageView, Box
from .file_tree import FileTree
from .label_table import LabelTable
from .stats_panel import StatsPanel
from .merge_designer.dialog import MergeDesignerDialog

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Editor — by Srijon Basak")
        self.resize(1400, 900)

        self.repo: DatasetRepository | None = None
        self.class_names: List[str] = []
        self.image_list: List[Path] = []
        self.current_index: int = -1

        self.file_tree = FileTree()
        self.file_tree.on_open(self._open_image_from_tree)

        self.view = ImageView()
        self.view.requestPrev.connect(self.prev_image)
        self.view.requestNext.connect(self.next_image)

        tool = QWidget()
        tl = QHBoxLayout(tool)
        self.cmb_class = QComboBox()
        self.btn_to_cls = QPushButton("Set selection → class")
        self.btn_save = QPushButton("Save (S)")
        tl.addWidget(QLabel("Current class:"))
        tl.addWidget(self.cmb_class, 1)
        tl.addWidget(self.btn_to_cls)
        tl.addStretch(1)
        tl.addWidget(self.btn_save)

        center = QWidget()
        cl = QVBoxLayout(center)
        cl.addWidget(tool)
        cl.addWidget(self.view, 1)

        self.labels = LabelTable()
        self.stats = StatsPanel()
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.addWidget(self.labels, 2)
        rl.addWidget(self.stats, 1)

        split = QSplitter()
        split.addWidget(self.file_tree)
        split.addWidget(center)
        split.addWidget(right)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 3)
        split.setStretchFactor(2, 2)
        self.setCentralWidget(split)

        sb = QStatusBar(self); self.setStatusBar(sb)
        self.view.set_status_sink(lambda s: self.statusBar().showMessage(s, 2000))

        m = self.menuBar()
        filem = m.addMenu("File")
        act_open_root = QAction("Open Dataset Root…", self); act_open_root.triggered.connect(self.open_root)
        act_open_yaml = QAction("Open Dataset YAML…", self); act_open_yaml.triggered.connect(self.open_yaml)
        act_save = QAction("Save", self); act_save.setShortcut(QKeySequence.StandardKey.Save); act_save.triggered.connect(self.save_current)
        filem.addAction(act_open_root); filem.addAction(act_open_yaml); filem.addSeparator(); filem.addAction(act_save)

        toolsm = m.addMenu("Tools")
        act_merge = QAction("Merge Datasets…", self); act_merge.triggered.connect(self.open_merge)
        toolsm.addAction(act_merge)

        self.btn_to_cls.clicked.connect(self._set_selected_to_current_class)
        self.btn_save.clicked.connect(self.save_current)
        self.cmb_class.currentIndexChanged.connect(self._on_current_class_changed)
        self.addAction(act_save)

    def open_root(self):
        d = QFileDialog.getExistingDirectory(self, "Select dataset root (train/eval/test with Image/Labels)")
        if not d: return
        self._load_repo(Path(d), None)

    def open_yaml(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select data.yaml", filter="YAML files (*.yaml *.yml)")
        if not f: return
        yaml_path = Path(f)
        self._load_repo(yaml_path.parent, yaml_path)

    def _load_repo(self, root: Path, yaml_path: Path | None):
        try:
            self.repo = DatasetRepository(root=root, yaml_path=yaml_path)
            self.class_names = self.repo.names or []
            self._populate_classes()
            self._refresh_file_tree()
            self._update_stats()
            all_imgs = []
            for s in ("train","val","test"):
                all_imgs.extend(self.repo.splits_map.get(s, []))
            self.image_list = all_imgs
            self.current_index = -1
            if self.image_list:
                self.next_image()
            self.statusBar().showMessage("Dataset loaded.", 2000)
        except Exception as e:
            QMessageBox.critical(self, "Load failed", str(e))

    def _populate_classes(self):
        self.cmb_class.clear()
        for i, n in enumerate(self.class_names):
            self.cmb_class.addItem(f"{i}: {n}", i)
        self.view.set_current_class(0, self.class_names)
        self.labels.set_class_names(self.class_names)

    def _refresh_file_tree(self):
        if not self.repo: return
        self.file_tree.populate_from_splits(self.repo.splits_map)

    def _open_image_from_tree(self, path: Path):
        if not self.repo: return
        try:
            self.current_index = self.image_list.index(path)
        except ValueError:
            self.image_list = []
            for s in ("train","val","test"):
                self.image_list.extend(self.repo.splits_map.get(s, []))
            self.current_index = self.image_list.index(path) if path in self.image_list else -1
        if self.current_index >= 0:
            self._show_current()

    def next_image(self):
        if not self.repo or not self.image_list: return
        self.current_index = (self.current_index + 1) % len(self.image_list)
        self._show_current()

    def prev_image(self):
        if not self.repo or not self.image_list: return
        self.current_index = (self.current_index - 1) % len(self.image_list)
        self._show_current()

    def _show_current(self):
        path = self.image_list[self.current_index]
        import numpy as np
        data = np.fromfile(str(path), dtype=np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            QMessageBox.warning(self, "Open image", f"Failed to load image: {path}")
            return
        self.view.show_image_bgr(path, bgr)
        lbl_path = self.repo.label_path_for(path)
        rows = parse_label_file(lbl_path)
        self.view.clear_boxes()
        for (c, x, y, w, h) in rows:
            self.view.add_box_norm(Box(c, x, y, w, h))
        self.labels.set_rows(rows, self.class_names)
        self.statusBar().showMessage(str(path))

    def save_current(self):
        if not self.repo or self.current_index < 0: return
        path = self.image_list[self.current_index]
        lbl_path = self.repo.label_path_for(path)
        table_rows = self.labels.read_rows()
        canvas_boxes = self.view.get_boxes_as_norm()
        rows = []
        for i, b in enumerate(canvas_boxes):
            cls = table_rows[i][0] if i < len(table_rows) else b.cls
            rows.append((cls, b.cx, b.cy, b.w, b.h))
        save_label_file(lbl_path, rows)
        self.statusBar().showMessage(f"Saved: {lbl_path}", 2000)
        # refresh table to reflect saved order/classes
        self.labels.set_rows(rows, self.class_names)

    def _set_selected_to_current_class(self):
        idx = self.cmb_class.currentData()
        if idx is None: return
        self.labels._apply_class()
        self.view._emit_to_current()

    def _on_current_class_changed(self, *_):
        cid = self.cmb_class.currentData()
        if cid is None: cid = 0
        self.view.set_current_class(int(cid), self.class_names)

    def _update_stats(self):
        if not self.repo: return
        split_counts = {s: len(self.repo.splits_map.get(s, [])) for s in ("train","val","test")}
        cls_cnt = [0] * (len(self.repo.names) if self.repo.names else 0)
        if cls_cnt:
            for imgs in self.repo.splits_map.values():
                for p in imgs:
                    rows = parse_label_file(self.repo.label_path_for(p))
                    present = {c for (c, *_rest) in rows}
                    for c in present:
                        if 0 <= c < len(cls_cnt):
                            cls_cnt[c] += 1
        self.stats.set_split_counts(split_counts)
        self.stats.set_class_counts(self.repo.names or [], cls_cnt)

    def open_merge(self):
        dlg = MergeDesignerDialog(self)
        dlg.exec()
