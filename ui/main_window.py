from pathlib import Path
from typing import List, Optional
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QAction, QPixmap, QImage
from PySide6.QtWidgets import (
    QMainWindow, QFileDialog, QStatusBar, QToolBar, QLabel, QComboBox, QMessageBox
)
import cv2

from ..core.repo import DatasetRepository
from ..core.yolo_io import parse_label_file, save_label_file
from .image_view import ImageScene, ImageView, BBoxItem

def yolo_to_rect(x, y, w, h, iw, ih):
    """YOLO (cx,cy,w,h normalized) -> QRectF in pixels."""
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
        self.resize(1200, 800)

        self.repo: Optional[DatasetRepository] = None
        self.images: List[Path] = []
        self.index: int = 0
        self.class_names: List[str] = []
        self.current_class_id: int = 0

        self.scene = ImageScene(self)
        self.view = ImageView(self.scene)
        self.setCentralWidget(self.view)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.class_selector = QComboBox()
        self.class_selector.setMinimumWidth(200)
        self.class_selector.currentIndexChanged.connect(self._on_class_changed)

        self._build_menu()
        self._build_toolbar()

    # ========= UI construct =========
    def _build_menu(self):
        menubar = self.menuBar()

        mfile = menubar.addMenu("&File")
        act_open = QAction("Open Dataset Root…", self)
        act_open.triggered.connect(self.open_dataset_root)
        mfile.addAction(act_open)
        act_save = QAction("Save Labels", self)
        act_save.setShortcut("S")
        act_save.triggered.connect(self.save_labels)
        mfile.addAction(act_save)
        mfile.addSeparator()
        act_quit = QAction("Quit", self)
        act_quit.triggered.connect(self.close)
        mfile.addAction(act_quit)

        medit = menubar.addMenu("&Edit")
        act_add = QAction("Add Box", self)
        act_add.setShortcut("A")
        act_add.triggered.connect(self.start_add_box)
        medit.addAction(act_add)
        act_change = QAction("Set Selected to Class", self)
        act_change.setShortcut("C")
        act_change.triggered.connect(self.change_selected_class)
        medit.addAction(act_change)
        act_del = QAction("Delete Selected", self)
        act_del.setShortcut("Delete")
        act_del.triggered.connect(self.delete_selected)
        medit.addAction(act_del)

        mnav = menubar.addMenu("&Navigate")
        act_prev = QAction("Previous", self)
        act_prev.setShortcut("P")
        act_prev.triggered.connect(self.prev_image)
        act_next = QAction("Next", self)
        act_next.setShortcut("N")
        act_next.triggered.connect(self.next_image)
        mnav.addAction(act_prev)
        mnav.addAction(act_next)

        mview = menubar.addMenu("&View")
        act_fit = QAction("Fit to Window", self)
        act_fit.setShortcut("F")
        act_fit.triggered.connect(self.fit_to_window)
        mview.addAction(act_fit)
        act_100 = QAction("Zoom 100%", self)
        act_100.setShortcut("1")
        act_100.triggered.connect(self.zoom_100)
        mview.addAction(act_100)

    def _build_toolbar(self):
        tb = QToolBar("Tools")
        self.addToolBar(tb)
        tb.addAction("Add Box", self.start_add_box)
        tb.addSeparator()
        tb.addWidget(QLabel(" Class: "))
        tb.addWidget(self.class_selector)
        tb.addAction("Set Selected", self.change_selected_class)
        tb.addSeparator()
        tb.addAction("Prev", self.prev_image)
        tb.addAction("Next", self.next_image)
        tb.addAction("Save", self.save_labels)

    # ========= dataset =========
    def open_dataset_root(self):
        folder = QFileDialog.getExistingDirectory(self, "Select dataset_root")
        if not folder:
            return
        root = Path(folder)
        self.repo = DatasetRepository(root)
        self.images = self.repo.list_images()
        if not self.images:
            QMessageBox.warning(self, "No images", "No images under images/*")
            return

        classes_txt = self.repo.classes_file()
        self.class_names = []
        self.class_selector.clear()
        if classes_txt.exists():
            for i, line in enumerate(classes_txt.read_text(encoding="utf-8").splitlines()):
                name = line.strip()
                if name:
                    self.class_names.append(name)
            if self.class_names:
                for i, n in enumerate(self.class_names):
                    self.class_selector.addItem(f"{i}: {n}")
        if not self.class_names:
            for i in range(10):
                self.class_selector.addItem(str(i))

        self.index = 0
        self.load_current()

    def _cv_read_rgb(self, path: Path):
        """Read image via OpenCV and return RGB numpy array, or None."""
        img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    def load_current(self):
        if not self.images:
            return
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
            label = (
                self.class_names[cls_idx]
                if 0 <= cls_idx < len(self.class_names)
                else str(cls_idx)
            )
            self.scene.addItem(BBoxItem(rect, cls_idx, label))

        self.status.showMessage(f"{img_path.name}  |  {self.index+1}/{len(self.images)}")

    def prev_image(self):
        if not self.images:
            return
        self.index = (self.index - 1) % len(self.images)
        self.load_current()

    def next_image(self):
        if not self.images:
            return
        self.index = (self.index + 1) % len(self.images)
        self.load_current()

    # ========= view utils =========
    def fit_to_window(self):
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def zoom_100(self):
        self.view.resetTransform()

    # ========= edit =========
    def start_add_box(self):
        self.scene.enter_add_mode()

        # Hook release event once to finalize creation (simple MVP)
        orig_release = self.scene.mouseReleaseEvent

        def wrapper(e):
            if self.scene.add_mode and self.scene.temp:
                rect = self.scene.temp.rect().normalized()
                self.scene.removeItem(self.scene.temp)
                self.scene.temp = None
                self.scene.exit_add_mode()
                if rect.width() >= 4 and rect.height() >= 4:
                    label = (
                        self.class_names[self.current_class_id]
                        if self.class_names
                        else str(self.current_class_id)
                    )
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
                it.label = (
                    self.class_names[self.current_class_id]
                    if self.class_names
                    else str(self.current_class_id)
                )
                changed += 1
        if changed:
            self.scene.update()
            self.status.showMessage(f"Changed class for {changed} box(es)", 3000)

    def save_labels(self):
        if not self.images:
            return
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
        # atomic write: write to temp then replace
        tmp = lbl_path.with_suffix(".txt.tmp")
        save_label_file(tmp, rows)
        tmp.replace(lbl_path)
        self.status.showMessage(f"Saved {lbl_path}", 3000)
