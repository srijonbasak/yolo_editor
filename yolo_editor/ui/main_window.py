from __future__ import annotations
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import yaml
import cv2

from PySide6.QtCore import Qt, QSize, QPointF, QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QFileDialog, QSplitter, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QTabWidget, QTreeWidget, QTreeWidgetItem,
    QTableWidget, QTableWidgetItem, QHeaderView, QListWidget, QListWidgetItem,
    QTextEdit, QMessageBox, QStatusBar
)

from .image_view import ImageView, Box
# Merge Designer components (added previously)
from .merge_designer import MergeController, MergeCanvas, MergePalette
from .merge_designer.controller import SourceClass


# ---------- helpers: IO ----------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
TXT = ".txt"


def read_yolo_txt(txt_path: Path) -> List[Box]:
    """Read YOLOv5/8 txt (class cx cy w h) normalized -> list of Box"""
    out: List[Box] = []
    if not txt_path.exists():
        return out
    try:
        for line in txt_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            cx, cy, w, h = map(float, parts[1:5])
            out.append(Box(cls=cls, cx=cx, cy=cy, w=w, h=h))
    except Exception:
        # corrupted line? ignore
        pass
    return out


def atomic_write_text(path: Path, content: str):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def write_yolo_txt(txt_path: Path, boxes: List[Box]):
    lines = []
    for b in boxes:
        lines.append(f"{int(b.cls)} {b.cx:.6f} {b.cy:.6f} {b.w:.6f} {b.h:.6f}")
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(txt_path, "\n".join(lines) + ("\n" if lines else ""))


# ---------- dataset resolution ----------
class DatasetPaths:
    def __init__(self):
        # split -> (images_dir, labels_dir)
        self.splits: Dict[str, Tuple[Optional[Path], Optional[Path]]] = {}
        self.yaml_path: Optional[Path] = None
        self.names: List[str] = []

    def class_name(self, cid: int) -> str:
        if 0 <= cid < len(self.names):
            return self.names[cid]
        return str(cid)


def _normalize_split_name(name: str) -> str:
    name = name.lower()
    if name == "valid":
        return "val"
    if name == "eval":
        return "eval"
    return name


def _guess_labels_dir(images_dir: Path) -> Optional[Path]:
    # Try sibling "labels" alongside "images"
    if images_dir.name == "images":
        cand = images_dir.parent / "labels"
        if cand.exists():
            return cand
    # Try images_dir/../labels/<split> if images_dir ends with split/images
    if images_dir.name in ("train", "val", "valid", "test", "eval"):
        # layout C variant (images directly under split, labels under split/labels)
        cand = images_dir / "labels"
        if cand.exists():
            return cand
    # Try <split>/labels sibling to <split>/images
    if images_dir.parent.name in ("train", "val", "valid", "test", "eval"):
        cand = images_dir.parent / "labels"
        if cand.exists():
            return cand
    # Try parent tree search one level up
    up = images_dir.parent
    maybe = up / "labels"
    if maybe.exists():
        return maybe
    return None


def _resolve_from_yaml(yaml_path: Path) -> DatasetPaths:
    dp = DatasetPaths()
    dp.yaml_path = yaml_path
    y = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))

    # names
    names: List[str] = []
    if isinstance(y.get("names"), list):
        names = [str(n) for n in y["names"]]
    elif isinstance(y.get("names"), dict):
        # sometimes dict {0:'a',1:'b',...}
        # order by numeric key
        items = sorted(((int(k), str(v)) for k, v in y["names"].items()), key=lambda t: t[0])
        names = [v for _, v in items]
    dp.names = names

    # splits
    for key in ("train", "val", "valid", "test", "eval"):
        if key in y:
            split = _normalize_split_name(key)
            raw = Path(str(y[key]))
            # paths may be relative to yaml
            base = yaml_path.parent
            # images directory may be provided or a parent folder
            img_dir = (base / raw).resolve()
            # In Ultralytics yaml, train/val/test typically point to ".../images"
            if not img_dir.exists():
                # try without ".." segments: strip leading ../
                txt = str(raw)
                while txt.startswith("../"):
                    txt = txt[3:]
                img_dir = (base / txt).resolve()
            if img_dir.exists() and img_dir.is_dir():
                labels_dir = _guess_labels_dir(img_dir)
                dp.splits[split] = (img_dir, labels_dir)
    return dp


def _resolve_from_root(root: Path) -> DatasetPaths:
    dp = DatasetPaths()
    # If there is a data.yaml at root, prefer it
    cand = root / "data.yaml"
    if cand.exists():
        return _resolve_from_yaml(cand)

    # Otherwise, try to discover
    # Layout A: images/<split>, labels/<split>
    images_root = root / "images"
    labels_root = root / "labels"
    if images_root.exists():
        for sp in ("train", "val", "valid", "test", "eval"):
            img_dir = images_root / sp
            if img_dir.exists():
                labels_dir = None
                if labels_root.exists():
                    lab_dir_a = labels_root / sp
                    if lab_dir_a.exists():
                        labels_dir = lab_dir_a
                if labels_dir is None:
                    labels_dir = _guess_labels_dir(img_dir)
                dp.splits[_normalize_split_name(sp)] = (img_dir, labels_dir)

    # Layout B: <split>/images, <split>/labels
    for sp in ("train", "val", "valid", "test", "eval"):
        sp_dir = root / sp
        if sp_dir.exists():
            img1 = sp_dir / "images"
            if img1.exists():
                lab1 = sp_dir / "labels" if (sp_dir / "labels").exists() else _guess_labels_dir(img1)
                dp.splits[_normalize_split_name(sp)] = (img1, lab1)
            else:
                # Layout C: <split>/*.jpg, <split>/labels/*.txt
                # treat split folder itself as images dir
                labs_c = sp_dir / "labels"
                if any((p.suffix.lower() in IMG_EXTS) for p in sp_dir.glob("*")):
                    dp.splits[_normalize_split_name(sp)] = (sp_dir, labs_c if labs_c.exists() else None)

    # names: try classes.txt or names.txt or nothing (fallback to ids)
    names_txt = root / "classes.txt"
    if names_txt.exists():
        dp.names = [ln.strip() for ln in names_txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
    else:
        dp.names = []
    return dp


def _scan_images(img_dir: Path) -> List[Path]:
    out: List[Path] = []
    for p in img_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            out.append(p)
    out.sort()
    return out


def _label_path_for(img_path: Path, labels_dir: Optional[Path]) -> Path:
    if labels_dir is None:
        # fallback: same folder with .txt
        return img_path.with_suffix(TXT)
    # common convention: labels/<split>/<same_stem>.txt
    return labels_dir / (img_path.stem + TXT)


def _compute_stats_for_split(imgs: List[Path], labels_dir: Optional[Path], names: List[str]) -> Dict:
    per_class_images = {}  # class_id -> count of images that include at least one instance
    per_class_boxes = {}   # class_id -> total boxes
    total_imgs = len(imgs)
    for img in imgs:
        txt = _label_path_for(img, labels_dir)
        seen: set[int] = set()
        for b in read_yolo_txt(txt):
            per_class_boxes[b.cls] = per_class_boxes.get(b.cls, 0) + 1
            seen.add(b.cls)
        for cid in seen:
            per_class_images[cid] = per_class_images.get(cid, 0) + 1
    # map to {cid:{images,boxes,name}}
    per = {}
    max_cid = max(per_class_images.keys() | per_class_boxes.keys(), default=-1)
    for cid in range(max_cid + 1):
        per[cid] = {
            "name": names[cid] if 0 <= cid < len(names) else str(cid),
            "images": per_class_images.get(cid, 0),
            "boxes": per_class_boxes.get(cid, 0)
        }
    return {
        "total_images": total_imgs,
        "per_class": per
    }


# ---------- MainWindow ----------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Editor — Label & Merge (Offline)")
        self.resize(1280, 800)
        self.setStatusBar(QStatusBar(self))

        # state
        self.dataset_roots: Dict[str, Path] = {}          # name -> path (root or yaml parent)
        self.loaded_datasets: Dict[str, Dict] = {}        # name -> {"class_names":[], "splits":{split: {...}}}
        self.current_dataset: Optional[str] = None
        self.current_split: Optional[str] = None
        self.current_images: List[Path] = []
        self.current_index: int = -1
        self.current_labels_dir: Optional[Path] = None
        self.current_names: List[str] = []

        self._build_menu()
        self._build_tabs()

    # ---------- UI build ----------
    def _build_menu(self):
        menubar = self.menuBar()
        m_file = menubar.addMenu("&File")

        act_open_root = QAction("Open Dataset &Root…", self)
        act_open_root.triggered.connect(self._open_root)
        m_file.addAction(act_open_root)

        act_open_yaml = QAction("Open Dataset &YAML…", self)
        act_open_yaml.triggered.connect(self._open_yaml)
        m_file.addAction(act_open_yaml)

        m_file.addSeparator()

        act_diag = QAction("Show Dataset &Diagnostics", self)
        act_diag.triggered.connect(self._show_diagnostics)
        m_file.addAction(act_diag)

        m_file.addSeparator()
        act_quit = QAction("&Quit", self)
        act_quit.triggered.connect(self.close)
        m_file.addAction(act_quit)

        m_edit = menubar.addMenu("&Edit")
        act_save = QAction("&Save Labels (S)", self)
        act_save.triggered.connect(self._save_current_labels)
        act_save.setShortcut("S")
        m_edit.addAction(act_save)

        act_del = QAction("&Delete Selected (Del)", self)
        act_del.triggered.connect(lambda: self.image_view._emit_delete())
        act_del.setShortcut("Delete")
        m_edit.addAction(act_del)

        act_to_cls = QAction("Set Selected → Current Class (C)", self)
        act_to_cls.triggered.connect(lambda: self.image_view._emit_to_current())
        act_to_cls.setShortcut("C")
        m_edit.addAction(act_to_cls)

        act_prev = QAction("&Prev Image (←)", self)
        act_prev.triggered.connect(self._prev_image)
        act_prev.setShortcut(Qt.Key_Left)
        m_edit.addAction(act_prev)

        act_next = QAction("&Next Image (→)", self)
        act_next.triggered.connect(self._next_image)
        act_next.setShortcut(Qt.Key_Right)
        m_edit.addAction(act_next)

    def _build_tabs(self):
        self.tabs = QTabWidget(self)
        self.setCentralWidget(self.tabs)

        # ==== Label Editor tab ====
        editor_root = QWidget()
        editor_splitter = QSplitter(Qt.Orientation.Horizontal, editor_root)

        # left: files tree
        left = QWidget()
        left_v = QVBoxLayout(left)
        self.split_selector = QComboBox()
        self.split_selector.currentTextChanged.connect(self._on_split_changed)
        left_v.addWidget(QLabel("Split:"))
        left_v.addWidget(self.split_selector)

        self.file_tree = QTreeWidget()
        self.file_tree.setHeaderLabels(["Image Files"])
        self.file_tree.itemClicked.connect(self._on_file_clicked)
        left_v.addWidget(self.file_tree, 1)

        # center: ImageView + top controls
        center = QWidget()
        c_v = QVBoxLayout(center)
        top_bar = QHBoxLayout()
        self.lbl_dataset = QLabel("Dataset: -")
        self.class_combo = QComboBox()
        self.class_combo.currentIndexChanged.connect(self._on_class_changed)
        self.btn_save = QPushButton("Save (S)")
        self.btn_save.clicked.connect(self._save_current_labels)
        top_bar.addWidget(self.lbl_dataset, 1)
        top_bar.addWidget(QLabel("Class:"))
        top_bar.addWidget(self.class_combo)
        top_bar.addStretch(1)
        top_bar.addWidget(self.btn_save)
        c_v.addLayout(top_bar)

        self.image_view = ImageView()
        self.image_view.requestPrev.connect(self._prev_image)
        self.image_view.requestNext.connect(self._next_image)
        self.image_view.set_status_sink(self._set_status)
        c_v.addWidget(self.image_view, 1)

        # right: labels & stats
        right = QWidget()
        r_v = QVBoxLayout(right)
        r_v.addWidget(QLabel("Labels (YOLO)"))
        self.labels_table = QTableWidget(0, 6)
        self.labels_table.setHorizontalHeaderLabels(["name", "id", "cx", "cy", "w", "h"])
        self.labels_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.labels_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        r_v.addWidget(self.labels_table, 3)

        r_v.addWidget(QLabel("Dataset Stats"))
        self.stats_list = QListWidget()
        r_v.addWidget(self.stats_list, 2)

        editor_splitter.addWidget(left)
        editor_splitter.addWidget(center)
        editor_splitter.addWidget(right)
        editor_splitter.setStretchFactor(1, 1)

        self.tabs.addTab(editor_splitter, "Label Editor")

        # ==== Merge Designer tab ====
        self.merge_ctrl = MergeController()
        self.merge_canvas = MergeCanvas(self.merge_ctrl)
        self.merge_palette = MergePalette(
            on_spawn_dataset=self._md_spawn_dataset,
            on_spawn_target_class=self._md_spawn_target_class
        )

        merge_split = QSplitter(Qt.Orientation.Horizontal)
        merge_split.addWidget(self.merge_palette)
        merge_split.addWidget(self.merge_canvas)
        merge_split.setStretchFactor(1, 1)
        self.tabs.addTab(merge_split, "Merge Designer")

    # ---------- status ----------
    def _set_status(self, msg: str):
        self.statusBar().showMessage(msg, 4000)

    # ---------- menu actions ----------
    def _open_root(self):
        d = QFileDialog.getExistingDirectory(self, "Open Dataset Root")
        if not d:
            return
        root = Path(d)
        dp = _resolve_from_root(root)
        if not dp.splits:
            QMessageBox.warning(self, "Not a dataset", "Could not detect train/val/test/eval in this folder.")
            return
        name = root.name
        self.dataset_roots[name] = root
        self._register_dataset(name, dp, base=root)
        self._set_current_dataset(name)

    def _open_yaml(self):
        f, _ = QFileDialog.getOpenFileName(self, "Open data.yaml", "", "YAML files (*.yaml *.yml)")
        if not f:
            return
        ypath = Path(f)
        dp = _resolve_from_yaml(ypath)
        if not dp.splits:
            QMessageBox.warning(self, "Invalid YAML", "Could not resolve any split paths from this YAML.")
            return
        name = ypath.parent.name
        self.dataset_roots[name] = ypath.parent
        self._register_dataset(name, dp, base=ypath.parent)
        self._set_current_dataset(name)

    def _show_diagnostics(self):
        if self.current_dataset is None:
            QMessageBox.information(self, "Diagnostics", "No dataset loaded.")
            return
        info = self.loaded_datasets[self.current_dataset]
        msg = []
        msg.append(f"DATASET: {self.current_dataset}")
        msg.append(f"Root: {self.dataset_roots[self.current_dataset]}")
        splits = info["splits"]
        for sp, spinfo in splits.items():
            msg.append(f"  [{sp}] images: {spinfo['images_dir']}")
            msg.append(f"        labels: {spinfo['labels_dir']}")
            msg.append(f"        count: {len(spinfo['images'])} files")
        QMessageBox.information(self, "Dataset Diagnostics", "\n".join(msg))

    # ---------- dataset registry ----------
    def _register_dataset(self, name: str, dp: DatasetPaths, base: Path):
        # Scan images per split and compute stats
        splits = {}
        for sp, (img_dir, lab_dir) in dp.splits.items():
            if img_dir is None:
                continue
            imgs = _scan_images(img_dir)
            splits[sp] = {
                "images_dir": img_dir,
                "labels_dir": lab_dir,
                "images": imgs,
                "stats": _compute_stats_for_split(imgs, lab_dir, dp.names)
            }
        self.loaded_datasets[name] = {
            "class_names": dp.names or [],
            "splits": splits
        }
        # Refresh merge designer palette/controller with *all* datasets
        self._refresh_merge_palette()

    def _set_current_dataset(self, name: str):
        self.current_dataset = name
        self.lbl_dataset.setText(f"Dataset: {name}")
        info = self.loaded_datasets[name]
        self.current_names = info["class_names"] or []
        self._fill_classes(self.current_names)
        self._fill_splits(info["splits"])
        # auto select first split
        if self.split_selector.count() > 0:
            self.split_selector.setCurrentIndex(0)

    def _fill_classes(self, names: List[str]):
        self.class_combo.blockSignals(True)
        self.class_combo.clear()
        if names:
            for n in names:
                self.class_combo.addItem(n)
        else:
            # fallback to some ids when names unknown
            for i in range(100):
                self.class_combo.addItem(str(i))
        self.class_combo.blockSignals(False)
        # sync with image view
        self.image_view.set_current_class(self.class_combo.currentIndex(), names)

    def _fill_splits(self, splits: Dict[str, Dict]):
        self.split_selector.blockSignals(True)
        self.split_selector.clear()
        # deterministic order
        for s in ("train", "val", "eval", "test"):
            if s in splits:
                self.split_selector.addItem(s)
        # add any others discovered
        for s in splits.keys():
            if self.split_selector.findText(s) < 0:
                self.split_selector.addItem(s)
        self.split_selector.blockSignals(False)

    # ---------- split and files ----------
    def _on_split_changed(self, split: str):
        if self.current_dataset is None:
            return
        info = self.loaded_datasets[self.current_dataset]
        spinfo = info["splits"].get(split)
        if not spinfo:
            return
        self.current_split = split
        self.current_images = spinfo["images"]
        self.current_labels_dir = spinfo["labels_dir"]
        self._populate_file_tree(self.current_images)
        self._populate_stats(spinfo["stats"])
        if self.current_images:
            self._open_image_index(0)

    def _populate_file_tree(self, images: List[Path]):
        self.file_tree.clear()
        root = QTreeWidgetItem(["(images)"])
        self.file_tree.addTopLevelItem(root)
        for p in images:
            QTreeWidgetItem(root, [str(p.name)])
        self.file_tree.expandAll()

    def _populate_stats(self, stats: Dict):
        self.stats_list.clear()
        self.stats_list.addItem(f"Total images: {stats.get('total_images', 0)}")
        per = stats.get("per_class", {})
        # Sort by class id
        for cid in sorted(per.keys()):
            item = per[cid]
            self.stats_list.addItem(f"[{cid}] {item['name']}: {item['images']} imgs / {item['boxes']} boxes")

    def _on_file_clicked(self, item: QTreeWidgetItem, col: int):
        if not item.parent():
            return
        idx = item.parent().indexOfChild(item)
        self._open_image_index(idx)

    # ---------- image navigation ----------
    def _prev_image(self):
        if not self.current_images:
            return
        self._open_image_index(max(0, self.current_index - 1))

    def _next_image(self):
        if not self.current_images:
            return
        self._open_image_index(min(len(self.current_images) - 1, self.current_index + 1))

    def _open_image_index(self, idx: int):
        if idx < 0 or idx >= len(self.current_images):
            return
        path = self.current_images[idx]
        img = cv2.imdecode(np_from_file(path), cv2.IMREAD_COLOR)
        if img is None:
            self._set_status(f"Failed to load image: {path.name}")
            return
        self.image_view.show_image_bgr(path, img)

        # load labels
        txt = _label_path_for(path, self.current_labels_dir)
        boxes = read_yolo_txt(txt)
        self.image_view.clear_boxes()
        for b in boxes:
            self.image_view.add_box_norm(b)
        self._fill_labels_table(boxes)
        self.current_index = idx
        self._highlight_tree_row(idx)
        self._set_status(f"{path.name}  ({idx+1}/{len(self.current_images)})")

    def _highlight_tree_row(self, idx: int):
        root = self.file_tree.topLevelItem(0)
        if not root:
            return
        for i in range(root.childCount()):
            child = root.child(i)
            child.setSelected(i == idx)
        self.file_tree.scrollToItem(root.child(idx))

    # ---------- labels table ----------
    def _fill_labels_table(self, boxes: List[Box]):
        self.labels_table.setRowCount(0)
        for b in boxes:
            row = self.labels_table.rowCount()
            self.labels_table.insertRow(row)
            name = self._class_name(b.cls)
            self.labels_table.setItem(row, 0, QTableWidgetItem(name))
            self.labels_table.setItem(row, 1, QTableWidgetItem(str(b.cls)))
            self.labels_table.setItem(row, 2, QTableWidgetItem(f"{b.cx:.4f}"))
            self.labels_table.setItem(row, 3, QTableWidgetItem(f"{b.cy:.4f}"))
            self.labels_table.setItem(row, 4, QTableWidgetItem(f"{b.w:.4f}"))
            self.labels_table.setItem(row, 5, QTableWidgetItem(f"{b.h:.4f}"))

    def _class_name(self, cid: int) -> str:
        if 0 <= cid < len(self.current_names):
            return self.current_names[cid]
        return str(cid)

    # ---------- save ----------
    def _save_current_labels(self):
        if self.current_index < 0 or not self.current_images:
            return
        img_path = self.current_images[self.current_index]
        txt = _label_path_for(img_path, self.current_labels_dir)
        boxes = self.image_view.get_boxes_as_norm()
        write_yolo_txt(txt, boxes)
        self._set_status(f"Saved: {txt.name}")
        # refresh labels table
        self._fill_labels_table(boxes)
        # optionally refresh split stats (only image-level counts; skip for perf)
        # If you want live stats, uncomment:
        # spinfo = self.loaded_datasets[self.current_dataset]["splits"][self.current_split]
        # spinfo["stats"] = _compute_stats_for_split(self.current_images, self.current_labels_dir, self.current_names)
        # self._populate_stats(spinfo["stats"])

    # ---------- class change ----------
    def _on_class_changed(self, idx: int):
        self.image_view.set_current_class(idx, self.current_names)
        self._set_status(f"Current class → {self._class_name(idx)} [{idx}]")

    # ---------- Merge Designer integration ----------
    def _refresh_merge_palette(self):
        ds_names = []
        for ds_name, info in self.loaded_datasets.items():
            ds_names.append(ds_name)
            classes = []
            # compute per-class totals across all splits (train/val/test/eval) by images and boxes
            agg_images: Dict[int, int] = {}
            agg_boxes: Dict[int, int] = {}
            for sp, spinfo in info["splits"].items():
                per = spinfo["stats"]["per_class"]
                for cid, entry in per.items():
                    agg_images[cid] = agg_images.get(cid, 0) + int(entry.get("images", 0))
                    agg_boxes[cid]  = agg_boxes.get(cid, 0) + int(entry.get("boxes", 0))
            names = info["class_names"] or []
            for cid in sorted(agg_images.keys() | agg_boxes.keys()):
                classes.append(SourceClass(
                    dataset_id=ds_name,
                    class_id=cid,
                    class_name=names[cid] if 0 <= cid < len(names) else str(cid),
                    images=agg_images.get(cid, 0),
                    boxes=agg_boxes.get(cid, 0),
                ))
            self.merge_ctrl.upsert_dataset(ds_name, classes)
        self.merge_palette.populate(ds_names)

    def _md_spawn_dataset(self, ds_name: str):
        pos = QPointF(40 + 40 * len(self.merge_canvas.nodes), 40)
        classes = self.merge_ctrl.model.sources.get(ds_name, [])
        if not classes:
            QMessageBox.warning(self, "No classes", f"Dataset '{ds_name}' has no classes/stats loaded.")
            return
        self.merge_canvas.spawn_dataset_node(ds_name, classes, pos)

    def _md_spawn_target_class(self, name: str, quota: Optional[int]):
        tid = self.merge_ctrl.add_target_class(name=name, quota_images=quota)
        pos = QPointF(600 + 20 * len(self.merge_canvas.target_nodes), 60)
        self.merge_canvas.spawn_target_node(tid, name, quota, pos)


# ---------- cv2 imread with Windows/Unicode-safe path ----------
def np_from_file(path: Path):
    """Read file as np buffer and decode via cv2.imdecode (avoids Unicode issues)."""
    import numpy as np
    data = np.fromfile(str(path), dtype=np.uint8)
    return data
