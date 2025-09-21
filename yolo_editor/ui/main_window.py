from pathlib import Path
from typing import List, Optional, Dict
from collections import defaultdict

from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QAction, QPixmap, QImage, QKeySequence
from PySide6.QtWidgets import (
    QMainWindow, QFileDialog, QStatusBar, QToolBar, QLabel, QComboBox, QMessageBox,
    QDockWidget, QTreeWidget, QTreeWidgetItem, QTextEdit
)
import cv2

from ..core.repo import DatasetRepository
from ..core.yolo_io import parse_label_file, save_label_file
from .image_view import ImageScene, ImageView, BBoxItem

def yolo_to_rect(x, y, w, h, iw, ih):
    cx, cy, ww, hh = x * iw, y * ih, w * iw, h * ih
    return QRectF(cx - ww / 2, cy - hh / 2, ww, hh)

def rect_to_yolo(rect: QRectF, iw, ih):
    cx = (rect.left() + rect.width() / 2) / iw
    cy = (rect.top() + rect.height() / 2) / ih
    w  = rect.width() / iw
    h  = rect.height() / ih
    return cx, cy, w, h

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Editor (Offline)")
        self.resize(1400, 900)

        self.repo: Optional[DatasetRepository] = None
        self.images: List[Path] = []
        self.index: int = 0
        self.class_names: List[str] = []
        self.current_class_id: int = 0
        self.path_to_index: Dict[Path, int] = {}

        # stats
        self.class_img_counts: Dict[int, int] = {}
        self.class_box_counts: Dict[int, int] = {}
        self.split_img_counts: Dict[str, int] = {}

        # center viewer
        self.scene = ImageScene(self)
        self.view = ImageView(self.scene)
        self.setCentralWidget(self.view)

        # status + class selector
        self.status = QStatusBar(); self.setStatusBar(self.status)
        self.class_selector = QComboBox(); self.class_selector.setMinimumWidth(220)
        self.class_selector.currentIndexChanged.connect(self._on_class_changed)

        self._build_menu()
        self._build_toolbar()
        self._build_docks()

    # ======== UI ========
    def _build_menu(self):
        menubar = self.menuBar()

        mfile = menubar.addMenu("&File")
        act_open = QAction("Open Dataset Root…", self)
        act_open.triggered.connect(self.open_dataset_root)
        mfile.addAction(act_open)

        act_open_yaml = QAction("Open Dataset YAML…", self)
        act_open_yaml.triggered.connect(self.open_dataset_yaml)
        mfile.addAction(act_open_yaml)

        act_diag = QAction("Show Dataset Diagnostics", self)
        act_diag.triggered.connect(self.show_diagnostics)
        mfile.addAction(act_diag)

        act_save = QAction("Save Labels", self); act_save.setShortcut("S")
        act_save.triggered.connect(self.save_labels); mfile.addAction(act_save)
        mfile.addSeparator()
        act_quit = QAction("Quit", self); act_quit.triggered.connect(self.close); mfile.addAction(act_quit)

        mnav = menubar.addMenu("&Navigate")
        act_prev = QAction("Previous", self)
        act_prev.setShortcuts([QKeySequence(Qt.Key_Left), QKeySequence("P")])
        act_prev.triggered.connect(self.prev_image)
        act_next = QAction("Next", self)
        act_next.setShortcuts([QKeySequence(Qt.Key_Right), QKeySequence("N")])
        act_next.triggered.connect(self.next_image)
        mnav.addAction(act_prev); mnav.addAction(act_next)

        mview = menubar.addMenu("&View")
        act_fit = QAction("Fit to Window", self); act_fit.setShortcut("F"); act_fit.triggered.connect(self.fit_to_window)
        mview.addAction(act_fit)
        act_100 = QAction("Zoom 100%", self); act_100.setShortcut("1"); act_100.triggered.connect(self.zoom_100)
        mview.addAction(act_100)

    def _build_toolbar(self):
        tb = QToolBar("Tools"); self.addToolBar(tb)
        tb.addAction("Add Box", self.start_add_box)
        tb.addSeparator()
        tb.addWidget(QLabel(" Class: "))
        tb.addWidget(self.class_selector)
        tb.addAction("Set Selected", self.change_selected_class)
        tb.addSeparator()
        tb.addAction("Prev", self.prev_image)
        tb.addAction("Next", self.next_image)
        tb.addAction("Save", self.save_labels)

    def _build_docks(self):
        # Left: file tree
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Dataset"])
        self.tree.itemClicked.connect(self._on_tree_clicked)
        dock_left = QDockWidget("Files", self)
        dock_left.setWidget(self.tree)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock_left)

        # Right top: label text for current image
        self.label_text = QTextEdit(); self.label_text.setReadOnly(True)
        dock_right = QDockWidget("Labels (YOLO)", self)
        dock_right.setWidget(self.label_text)
        self.addDockWidget(Qt.RightDockWidgetArea, dock_right)

        # Right bottom: dataset stats
        self.stats_text = QTextEdit(); self.stats_text.setReadOnly(True)
        dock_stats = QDockWidget("Dataset Stats", self)
        dock_stats.setWidget(self.stats_text)
        self.addDockWidget(Qt.RightDockWidgetArea, dock_stats)

    # ======== dataset ========
    def open_dataset_root(self):
        folder = QFileDialog.getExistingDirectory(self, "Select dataset_root")
        if not folder:
            return
        self.repo = DatasetRepository(Path(folder))
        self._after_repo_loaded()

    def open_dataset_yaml(self):
        yaml_path, _ = QFileDialog.getOpenFileName(self, "Select dataset YAML", "", "YAML files (*.yaml *.yml)")
        if not yaml_path:
            return
        yp = Path(yaml_path).resolve()
        self.repo = DatasetRepository(root=yp.parent, yaml_path=yp)  # <-- pass YAML explicitly
        self._after_repo_loaded()

    def show_diagnostics(self):
        if not self.repo:
            QMessageBox.information(self, "Diagnostics", "No dataset loaded yet.")
            return
        QMessageBox.information(self, "Diagnostics", self.repo.debug_info())

    def _after_repo_loaded(self):
        self.images = self.repo.list_images()
        if not self.images:
            QMessageBox.warning(
                self,
                "No images",
                "No images detected.\nWe support:\n"
                "• images/<split>/...\n"
                "• <split>/images/...\n"
                "• <split>/*.jpg + <split>/labels\n"
                "• YAML file with train/val(/valid)/test.\n\n"
                "Tip: File → Open Dataset YAML… to point directly at your YAML."
            )
            if self.repo:
                QMessageBox.information(self, "Diagnostics", self.repo.debug_info())
            return

        # class names
        self.class_names = self.repo.class_names or []
        self.class_selector.clear()
        if self.class_names:
            for i, n in enumerate(self.class_names):
                self.class_selector.addItem(f"{i}: {n}")
        else:
            for i in range(10): self.class_selector.addItem(str(i))

        # build tree + mapping + stats
        self._populate_tree()
        self.path_to_index = {p: i for i, p in enumerate(self.images)}
        self._compute_stats()

        self.index = 0
        self.load_current()

    def _compute_stats(self):
        cls_img_sets = defaultdict(set)
        cls_box_counts = defaultdict(int)
        split_counts = defaultdict(int)

        for split, imgs in self.repo.splits_map.items():
            split_counts[split] += len(imgs)
            for img in imgs:
                rows = parse_label_file(self.repo.label_path_for(img))
                seen = set()
                for cls_idx, *_ in rows:
                    cls_box_counts[cls_idx] += 1
                    seen.add(cls_idx)
                for c in seen:
                    cls_img_sets[c].add(img)

        self.class_img_counts = {c: len(s) for c, s in cls_img_sets.items()}
        self.class_box_counts = dict(cls_box_counts)
        self.split_img_counts = dict(split_counts)

        lines = []
        lines.append("Splits (images):")
        for s in sorted(self.split_img_counts):
            lines.append(f"  {s}: {self.split_img_counts[s]}")
        lines.append("")
        lines.append("Classes (images with class / total boxes):")
        max_id = max(set(self.class_img_counts) | set(self.class_box_counts) | set(range(len(self.class_names)))) if self.class_names else max(self.class_img_counts.keys() or [0])
        for c in range(max_id + 1):
            name = self.class_names[c] if 0 <= c < len(self.class_names) else str(c)
            imgs = self.class_img_counts.get(c, 0)
            boxes = self.class_box_counts.get(c, 0)
            lines.append(f"  {c}: {name} — {imgs} imgs, {boxes} boxes")
        self.stats_text.setPlainText("\n".join(lines))

    def _populate_tree(self):
        self.tree.clear()
        if not self.repo or not self.repo.splits_map:
            return
        for split, imgs in sorted(self.repo.splits_map.items()):
            parent = QTreeWidgetItem([split])
            self.tree.addTopLevelItem(parent)
            for p in imgs:
                child = QTreeWidgetItem([p.name])
                child.setData(0, Qt.UserRole, str(p))
                parent.addChild(child)
            parent.setExpanded(True)

    def _on_tree_clicked(self, item: QTreeWidgetItem, col: int):
        p = item.data(0, Qt.UserRole)
        if not p:
            return
        path = Path(p)
        idx = self.path_to_index.get(path)
        if idx is None:
            self.path_to_index = {pp: i for i, pp in enumerate(self.images)}
            idx = self.path_to_index.get(path)
        if idx is not None:
            self.index = idx
            self.load_current()

    def _cv_read_rgb(self, path: Path):
        img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    def load_current(self):
        if not self.images: return
        img_path = self.images[self.index]
        rgb = self._cv_read_rgb(img_path)
        if rgb is None:
            QMessageBox.critical(self, "Load error", f"Cannot load {img_path}")
            return

        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pm = QPixmap.fromImage(qimg)
        self.scene.set_image(pm)

        # Draw boxes
        lbl_path = self.repo.label_path_for(img_path)
        rows = parse_label_file(lbl_path)
        for cls_idx, x, y, bw, bh in rows:
            rect = yolo_to_rect(x, y, bw, bh, pm.width(), pm.height())
            label = self.class_names[cls_idx] if 0 <= cls_idx < len(self.class_names) else str(cls_idx)
            self.scene.addItem(BBoxItem(rect, cls_idx, label))

        # right panel text
        self._update_label_text(lbl_path, rows)

        split = self._guess_split(img_path)
        self.status.showMessage(f"{split} / {img_path.name}  |  {self.index+1}/{len(self.images)}")

    def _guess_split(self, p: Path) -> str:
        for s in ("train", "val", "valid", "test"):
            if s in p.parts:
                return s
        return ""

    def prev_image(self):
        if not self.images: return
        self.index = (self.index - 1) % len(self.images)
        self.load_current()

    def next_image(self):
        if not self.images: return
        self.index = (self.index + 1) % len(self.images)
        self.load_current()

    # ======== view utils ========
    def fit_to_window(self):
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def zoom_100(self):
        self.view.resetTransform()

    # ======== edit ========
    def start_add_box(self):
        self.scene.enter_add_mode()
        orig_release = self.scene.mouseReleaseEvent
        def wrapper(e):
            if self.scene.add_mode and self.scene.temp:
                rect = self.scene.temp.rect().normalized()
                self.scene.removeItem(self.scene.temp); self.scene.temp=None
                self.scene.exit_add_mode()
                if rect.width() >= 4 and rect.height() >= 4:
                    label = self.class_names[self.current_class_id] if self.class_names else str(self.current_class_id)
                    self.scene.addItem(BBoxItem(rect, self.current_class_id, label))
            orig_release(e)
        self.scene.mouseReleaseEvent = wrapper

    def delete_selected(self):
        for it in list(self.scene.selectedItems()):
            if isinstance(it, BBoxItem):
                self.scene.removeItem(it)

    def _on_class_changed(self, idx: int):
        self.current_class_id = idx

    def change_selected_class(self):
        changed = 0
        for it in self.scene.selectedItems():
            if isinstance(it, BBoxItem):
                it.cls_id = self.current_class_id
                it.label = self.class_names[self.current_class_id] if self.class_names else str(self.current_class_id)
                changed += 1
        if changed:
            self.scene.update()
            self.status.showMessage(f"Changed class for {changed} box(es)", 3000)

    def save_labels(self):
        if not self.images: return
        img_path = self.images[self.index]
        lbl_path = self.repo.label_path_for(img_path)
        if not self.scene.image_item:
            return
        iw = int(self.scene.image_item.pixmap().width())
        ih = int(self.scene.image_item.pixmap().height())
        rows = []
        for it in self.scene.items():
            if isinstance(it, BBoxItem):
                cx, cy, w, h = rect_to_yolo(it.rect(), iw, ih)
                rows.append((it.cls_id, cx, cy, w, h))
        tmp = lbl_path.with_suffix(".txt.tmp")
        save_label_file(tmp, rows)
        tmp.replace(lbl_path)
        self._update_label_text(lbl_path, rows)
        self.status.showMessage(f"Saved {lbl_path}", 3000)

    # ======== right panel text ========
    def _update_label_text(self, lbl_path: Path, rows: List[tuple]):
        lines = []
        for cls_idx, x, y, w, h in rows:
            name = self.class_names[cls_idx] if 0 <= cls_idx < len(self.class_names) else str(cls_idx)
            lines.append(f"{name}\t{cls_idx}\t{x:.6f}\t{y:.6f}\t{w:.6f}\t{h:.6f}")
        header = f"# {lbl_path}\n# name\tid\tx\ty\tw\th\n"
        self.label_text.setPlainText(header + "\n".join(lines))
