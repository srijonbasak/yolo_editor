from __future__ import annotations
from pathlib import Path
from typing import List
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QMainWindow, QFileDialog, QStatusBar, QTabWidget, QWidget, QSplitter, QVBoxLayout,
    QHBoxLayout, QLabel, QComboBox, QPushButton, QTreeWidget, QTreeWidgetItem,
    QTableWidget, QTableWidgetItem, QHeaderView, QListWidget, QMessageBox
)

# keep your existing image view
from .image_view import ImageView

# NEW helpers
from ..core.dataset_resolver import resolve_dataset, DatasetModel
from ..core.yolo_io import Box, read_yolo_txt, write_yolo_txt, labels_for_image, imread_unicode


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Editor — Offline")
        self.resize(1380, 860)
        self.setStatusBar(QStatusBar(self))

        self.dm: DatasetModel | None = None
        self.ds_name: str = "-"
        self.split: str | None = None
        self.images: List[Path] = []
        self.labels_dir: Path | None = None
        self.names: List[str] = []
        self.idx: int = -1

        self.tabs = QTabWidget(self)
        self.setCentralWidget(self.tabs)

        # --- Label Editor tab ---
        self.editor_root = QWidget()
        self._build_editor_ui(self.editor_root)
        self.tabs.addTab(self.editor_root, "Label Editor")

        # (optional) keep your Merge tab — only if imports are available
        try:
            from .merge_designer.canvas import MergeCanvas
            from .merge_designer.controller import MergeController
            from .merge_designer.palette import MergePalette
            self.merge_ctrl = MergeController()
            self.merge_canvas = MergeCanvas(self.merge_ctrl)
            self.merge_palette = MergePalette(
                on_spawn_dataset=lambda _: None,  # you can wire later
                on_spawn_target_class=lambda *_: None
            )
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

        act_root = QAction("Open Dataset &Root…", self)
        act_yaml = QAction("Open Dataset &YAML…", self)
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

    # ---------------- Open handlers ----------------

    def _open_root(self):
        d = QFileDialog.getExistingDirectory(self, "Open Dataset Root")
        if not d: return
        dm = resolve_dataset(Path(d))
        if not dm.splits:
            QMessageBox.warning(self, "Not a dataset", "Could not detect train/val/test/valid/eval in this folder.")
            return
        self._load_dataset(Path(d).name, dm)

    def _open_yaml(self):
        f, _ = QFileDialog.getOpenFileName(self, "Open data.yaml", "", "YAML (*.yaml *.yml)")
        if not f: return
        dm = resolve_dataset(Path(f))
        if not dm.splits:
            QMessageBox.warning(self, "Invalid YAML", "Could not resolve any split paths from this YAML.")
            return
        self._load_dataset(Path(f).parent.name, dm)

    def _load_dataset(self, name: str, dm: DatasetModel):
        self.dm = dm; self.ds_name = name
        self.lbl_ds.setText(f"Dataset: {name}")
        self.names = dm.names or []
        self.class_combo.blockSignals(True)
        self.class_combo.clear()
        if self.names:
            for n in self.names: self.class_combo.addItem(n)
        else:
            for i in range(100): self.class_combo.addItem(str(i))
        self.class_combo.blockSignals(False)
        self.view.set_current_class(self.class_combo.currentIndex(), self.names)

        # splits
        self.split_combo.blockSignals(True)
        self.split_combo.clear()
        for s in dm.ordered_splits(): self.split_combo.addItem(s)
        self.split_combo.blockSignals(False)

        # default split
        if dm.ordered_splits():
            self._on_split_changed(dm.ordered_splits()[0])

    # ---------------- Split / file tree ----------------

    def _on_split_changed(self, split: str):
        if not self.dm or split not in self.dm.splits: return
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
        self.view.show_image_bgr(img_path, img)
        boxes = read_yolo_txt(labels_for_image(img_path, self.labels_dir))
        self.view.clear_boxes()
        for b in boxes:
            self.view.add_box_norm(b)
        self._fill_table(boxes)
        self.idx = i
        self._highlight_tree_row(i)
        self.statusBar().showMessage(f"{img_path.name}  ({i+1}/{len(self.images)})", 4000)

    def _save_labels(self):
        if self.idx < 0 or not self.images: return
        img_path = self.images[self.idx]
        txt = labels_for_image(img_path, self.labels_dir)
        boxes = self.view.get_boxes_as_norm()
        write_yolo_txt(txt, [Box(int(b.cls), b.cx, b.cy, b.w, b.h) for b in boxes])
        self._fill_table(boxes)
        self.statusBar().showMessage(f"Saved: {txt}", 4000)

    def _fill_table(self, boxes: List[Box]):
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

    def _highlight_tree_row(self, idx: int):
        root = self.file_tree.topLevelItem(0)
        if not root: return
        for i in range(root.childCount()):
            ch = root.child(i); ch.setSelected(i == idx)
        self.file_tree.scrollToItem(root.child(idx))

    def _on_class_changed(self, idx: int):
        self.view.set_current_class(idx, self.names)
        nm = self.names[idx] if 0 <= idx < len(self.names) else str(idx)
        self.statusBar().showMessage(f"Current class → {nm} [{idx}]", 3000)

    # ---------------- Stats ----------------

    def _compute_stats_and_show(self):
        from collections import defaultdict
        per_imgs = defaultdict(int); per_boxes = defaultdict(int)
        for img in self.images:
            seen = set()
            for b in read_yolo_txt(labels_for_image(img, self.labels_dir)):
                per_boxes[b.cls] += 1; seen.add(b.cls)
            for c in seen: per_imgs[c] += 1
        self.stats.clear()
        self.stats.addItem(f"Total images: {len(self.images)}")
        ids = sorted(set(per_imgs.keys()) | set(per_boxes.keys()))
        for cid in ids:
            nm = self.names[cid] if 0 <= cid < len(self.names) else str(cid)
            self.stats.addItem(f"[{cid}] {nm}: {per_imgs.get(cid,0)} imgs / {per_boxes.get(cid,0)} boxes")
