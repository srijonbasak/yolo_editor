from __future__ import annotations
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import yaml
import cv2

from PySide6.QtCore import Qt, QSize, QPointF
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QFileDialog, QSplitter, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QTabWidget, QTreeWidget, QTreeWidgetItem,
    QTableWidget, QTableWidgetItem, QHeaderView, QListWidget, QMessageBox, QStatusBar
)

from .image_view import ImageView, Box
from .merge_designer.controller import SourceClass
from .merge_designer.controller import MergeController
from .merge_designer.canvas import MergeCanvas
from .merge_designer.palette import MergePalette

# ---------- helpers ----------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
TXT = ".txt"

def read_yolo_txt(txt_path: Path) -> List[Box]:
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
        return "val"   # export rule: eval → val bucket
    return name

def _guess_labels_dir(images_dir: Path) -> Optional[Path]:
    if images_dir.name == "images":
        cand = images_dir.parent / "labels"
        if cand.exists():
            return cand
    if images_dir.name in ("train", "val", "valid", "test", "eval"):
        cand = images_dir / "labels"
        if cand.exists():
            return cand
    if images_dir.parent.name in ("train", "val", "valid", "test", "eval"):
        cand = images_dir.parent / "labels"
        if cand.exists():
            return cand
    up = images_dir.parent
    maybe = up / "labels"
    if maybe.exists():
        return maybe
    return None

def _resolve_from_yaml(yaml_path: Path) -> DatasetPaths:
    dp = DatasetPaths()
    dp.yaml_path = yaml_path
    y = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}

    names: List[str] = []
    if isinstance(y.get("names"), list):
        names = [str(n) for n in y["names"]]
    elif isinstance(y.get("names"), dict):
        items = sorted(((int(k), str(v)) for k, v in y["names"].items()), key=lambda t: t[0])
        names = [v for _, v in items]
    dp.names = names

    for key in ("train", "val", "valid", "test", "eval"):
        if key in y:
            split = _normalize_split_name(key)
            raw = Path(str(y[key]))
            base = yaml_path.parent
            img_dir = (base / raw).resolve()
            if not img_dir.exists():
                txt = str(raw)
                while txt.startswith("../"):
                    txt = txt[3:]
                img_dir = (base / txt).resolve()
            if img_dir.exists() and img_dir.is_dir():
                labels_dir = _guess_labels_dir(img_dir)
                dp.splits[split] = (img_dir, labels_dir)
    return dp

def _resolve_from_root(root: Path) -> DatasetPaths:
    """
    Try YAML first; if nothing is found, fall back to scanning the folder tree.
    """
    dp = DatasetPaths()
    cand = root / "data.yaml"
    if cand.exists():
        dp_yaml = _resolve_from_yaml(cand)
        if dp_yaml.splits:
            return dp_yaml

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

    for sp in ("train", "val", "valid", "test", "eval"):
        sp_dir = root / sp
        if sp_dir.exists():
            img1 = sp_dir / "images"
            if img1.exists():
                lab1 = sp_dir / "labels" if (sp_dir / "labels").exists() else _guess_labels_dir(img1)
                dp.splits[_normalize_split_name(sp)] = (img1, lab1)
            else:
                labs_c = sp_dir / "labels"
                if any((p.suffix.lower() in IMG_EXTS) for p in sp_dir.rglob("*")):
                    dp.splits[_normalize_split_name(sp)] = (sp_dir, labs_c if labs_c.exists() else None)

    names_txt = root / "classes.txt"
    if names_txt.exists():
        dp.names = [ln.strip() for ln in names_txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
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
        return img_path.with_suffix(TXT)
    return labels_dir / (img_path.stem + TXT)

def _compute_stats_for_split(imgs: List[Path], labels_dir: Optional[Path], names: List[str]) -> Dict:
    per_class_images = {}
    per_class_boxes = {}
    total_imgs = len(imgs)
    for img in imgs:
        txt = _label_path_for(img, labels_dir)
        seen: set[int] = set()
        for b in read_yolo_txt(txt):
            per_class_boxes[b.cls] = per_class_boxes.get(b.cls, 0) + 1
            seen.add(b.cls)
        for cid in seen:
            per_class_images[cid] = per_class_images.get(cid, 0) + 1
    per = {}
    max_cid = max(per_class_images.keys() | per_class_boxes.keys(), default=-1)
    for cid in range(max_cid + 1):
        per[cid] = {
            "name": names[cid] if 0 <= cid < len(names) else str(cid),
            "images": per_class_images.get(cid, 0),
            "boxes": per_class_boxes.get(cid, 0)
        }
    return {"total_images": total_imgs, "per_class": per}

# ---------- MainWindow ----------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Editor — Label & Merge (Offline)")
        self.resize(1380, 860)
        self.setStatusBar(QStatusBar(self))

        self.dataset_roots: Dict[str, Path] = {}
        self.loaded_datasets: Dict[str, Dict] = {}
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
        act_save = QAction("&Save Labels (S)", self); act_save.setShortcut("S")
        act_del  = QAction("&Delete Selected (Del)", self); act_del.setShortcut("Delete")
        act_to_cls = QAction("Set Selected → Current Class (C)", self); act_to_cls.setShortcut("C")
        act_prev = QAction("&Prev Image (←)", self); act_prev.setShortcut(Qt.Key_Left)
        act_next = QAction("&Next Image (→)", self); act_next.setShortcut(Qt.Key_Right)

        act_save.triggered.connect(self._save_current_labels)
        act_del.triggered.connect(lambda: self.image_view._emit_delete())
        act_to_cls.triggered.connect(lambda: self.image_view._emit_to_current())
        act_prev.triggered.connect(self._prev_image)
        act_next.triggered.connect(self._next_image)

        for a in (act_save, act_del, act_to_cls, act_prev, act_next):
            m_edit.addAction(a)

    def _build_tabs(self):
        self.tabs = QTabWidget(self)
        self.setCentralWidget(self.tabs)

        # ==== Label Editor tab ====
        editor_root = QWidget()
        editor_splitter = QSplitter(Qt.Orientation.Horizontal, editor_root)

        left = QWidget(); left_v = QVBoxLayout(left)
        self.split_selector = QComboBox()
        self.split_selector.currentTextChanged.connect(self._on_split_changed)
        left_v.addWidget(QLabel("Split:")); left_v.addWidget(self.split_selector)
        self.file_tree = QTreeWidget(); self.file_tree.setHeaderLabels(["Image Files"])
        self.file_tree.itemClicked.connect(self._on_file_clicked)
        left_v.addWidget(self.file_tree, 1)

        center = QWidget(); c_v = QVBoxLayout(center)
        top_bar = QHBoxLayout()
        self.lbl_dataset = QLabel("Dataset: -")
        self.class_combo = QComboBox(); self.class_combo.currentIndexChanged.connect(self._on_class_changed)
        self.btn_save = QPushButton("Save (S)"); self.btn_save.clicked.connect(self._save_current_labels)
        top_bar.addWidget(self.lbl_dataset, 1)
        top_bar.addWidget(QLabel("Class:")); top_bar.addWidget(self.class_combo)
        top_bar.addStretch(1); top_bar.addWidget(self.btn_save)
        c_v.addLayout(top_bar)
        self.image_view = ImageView()
        self.image_view.requestPrev.connect(self._prev_image)
        self.image_view.requestNext.connect(self._next_image)
        self.image_view.set_status_sink(self._set_status)
        c_v.addWidget(self.image_view, 1)

        right = QWidget(); r_v = QVBoxLayout(right)
        r_v.addWidget(QLabel("Labels (YOLO)"))
        self.labels_table = QTableWidget(0, 6)
        self.labels_table.setHorizontalHeaderLabels(["name", "id", "cx", "cy", "w", "h"])
        self.labels_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.labels_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        r_v.addWidget(self.labels_table, 3)
        r_v.addWidget(QLabel("Dataset Stats"))
        self.stats_list = QListWidget(); r_v.addWidget(self.stats_list, 2)

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
        self.merge_palette.requestLoadDataset.connect(self._open_dataset_from_merge_panel)
        self.merge_palette.requestExportMerged.connect(self._merge_export)

        merge_split = QSplitter(Qt.Orientation.Horizontal)
        merge_split.addWidget(self.merge_palette)
        merge_split.addWidget(self.merge_canvas)
        merge_split.setStretchFactor(1, 1)
        self.tabs.addTab(merge_split, "Merge Designer")

    # ---------- status ----------
    def _set_status(self, msg: str):
        self.statusBar().showMessage(msg, 4000)

    # ---------- menu actions (and merge-panel load) ----------
    def _open_dataset_from_merge_panel(self):
        # Choose either YAML or folder on demand
        choice = QMessageBox.question(self, "Load Dataset", "Open a data.yaml? (No = open root folder)")
        if choice == QMessageBox.StandardButton.Yes:
            self._open_yaml()
        else:
            self._open_root()

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
        self._refresh_merge_palette()

    def _set_current_dataset(self, name: str):
        self.current_dataset = name
        self.lbl_dataset.setText(f"Dataset: {name}")
        info = self.loaded_datasets[name]
        self.current_names = info["class_names"] or []
        self._fill_classes(self.current_names)
        self._fill_splits(info["splits"])
        if self.split_selector.count() > 0:
            self.split_selector.setCurrentIndex(0)

    def _fill_classes(self, names: List[str]):
        self.class_combo.blockSignals(True)
        self.class_combo.clear()
        if names:
            for n in names:
                self.class_combo.addItem(n)
        else:
            for i in range(100):
                self.class_combo.addItem(str(i))
        self.class_combo.blockSignals(False)
        self.image_view.set_current_class(self.class_combo.currentIndex(), names)

    def _fill_splits(self, splits: Dict[str, Dict]):
        self.split_selector.blockSignals(True)
        self.split_selector.clear()
        for s in ("train", "val", "test"):
            if s in splits:
                self.split_selector.addItem(s)
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

    def _save_current_labels(self):
        if self.current_index < 0 or not self.current_images:
            return
        img_path = self.current_images[self.current_index]
        txt = _label_path_for(img_path, self.current_labels_dir)
        boxes = self.image_view.get_boxes_as_norm()
        write_yolo_txt(txt, boxes)
        self._set_status(f"Saved: {txt.name}")
        self._fill_labels_table(boxes)

    def _on_class_changed(self, idx: int):
        self.image_view.set_current_class(idx, self.current_names)
        self._set_status(f"Current class → {self._class_name(idx)} [{idx}]")

    # ---------- Merge Designer integration ----------
    def _refresh_merge_palette(self):
        ds_names = []
        for ds_name, info in self.loaded_datasets.items():
            ds_names.append(ds_name)
            classes = []
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
        pos = QPointF(40 + 40 * len(self.merge_canvas.nodes), 60)
        classes = self.merge_ctrl.model.sources.get(ds_name, [])
        if not classes:
            QMessageBox.warning(self, "No classes", f"Dataset '{ds_name}' has no classes/stats loaded.")
            return
        self.merge_canvas.spawn_dataset_node(ds_name, classes, pos)

    def _md_spawn_target_class(self, name: str, quota: Optional[int]):
        tid = self.merge_ctrl.add_target_class(name=name, quota_images=quota)
        pos = QPointF(900 + 20 * len(self.merge_canvas.target_nodes), 60)
        self.merge_canvas.spawn_target_node(tid, name, quota, pos)

    # ---------- Export merged dataset (FIXED: no label loss; robust quotas; clean YAML) ----------
    def _merge_export(self):
        if not self.merge_ctrl.model.targets:
            QMessageBox.warning(self, "Nothing to export", "Create at least one target class and connect sources.")
            return
        out_dir = QFileDialog.getExistingDirectory(self, "Select output folder for merged dataset")
        if not out_dir:
            return
        out = Path(out_dir)

        # Build mapping (dataset, src_class) -> target_id
        src2tgt: Dict[Tuple[str, int], int] = {}
        for e in self.merge_ctrl.model.edges:
            src2tgt[(e.source_key[0], e.source_key[1])] = e.target_id
        if not src2tgt:
            QMessageBox.warning(self, "No connections", "Connect at least one source class to a target class.")
            return

        # Planned quotas per target (total IMAGES for that target)
        quotas: Dict[int, int] = {}
        for tid in self.merge_ctrl.model.targets.keys():
            # sum the per-source allocations -> target total
            alloc = self.merge_ctrl.planned_allocation(tid)
            quotas[tid] = sum(int(n) for n in alloc.values()) if alloc else 0
            # 0 means unlimited
        # Track achieved counts per target
        picked_per_target: Dict[int, int] = {tid: 0 for tid in self.merge_ctrl.model.targets.keys()}

        # Ensure split dirs in output
        def ensure_split_dirs(base: Path, split: str):
            (base / split / "images").mkdir(parents=True, exist_ok=True)
            (base / split / "labels").mkdir(parents=True, exist_ok=True)
        for sp in ("train", "val", "test"):
            ensure_split_dirs(out, sp)

        # Build dataset lookups
        per_dataset = {}
        for ds_name, info in self.loaded_datasets.items():
            d = {}
            for sp, spinfo in info["splits"].items():
                d[sp] = (spinfo["images"], spinfo["labels_dir"])
            per_dataset[ds_name] = d

        # Helper to avoid name clashes
        used_names: Set[str] = set()
        def copy_unique(src_img: Path, dst_img_dir: Path, prefix: str) -> Path:
            stem = f"{prefix}_{src_img.stem}"
            name = stem + src_img.suffix
            i = 1
            while name.lower() in used_names or (dst_img_dir / name).exists():
                name = f"{stem}_{i}{src_img.suffix}"
                i += 1
            used_names.add(name.lower())
            dst = dst_img_dir / name
            shutil.copy2(src_img, dst)
            return dst

        # Pass 1: iterate all images, compute mapped boxes per image (across ALL targets)
        # Also record which targets each image contributes to.
        class ImageEntry:
            __slots__ = ("src_img", "src_ds", "src_split", "mapped_boxes", "targets_hit")
            def __init__(self, src_img: Path, src_ds: str, src_split: str):
                self.src_img = src_img
                self.src_ds = src_ds
                self.src_split = src_split
                self.mapped_boxes: List[Box] = []
                self.targets_hit: Set[int] = set()

        candidates: List[ImageEntry] = []
        for ds_name, splits in per_dataset.items():
            for sp_key in ("train", "val", "test", "eval"):
                if sp_key not in splits:
                    continue
                imgs, lbl_dir = splits[sp_key]
                for img in imgs:
                    txt = _label_path_for(img, lbl_dir)
                    if not txt.exists():
                        continue
                    mapped: List[Box] = []
                    targets_here: Set[int] = set()
                    for b in read_yolo_txt(txt):
                        key = (ds_name, int(b.cls))
                        if key in src2tgt:
                            tid = src2tgt[key]
                            mapped.append(Box(cls=tid, cx=b.cx, cy=b.cy, w=b.w, h=b.h))
                            targets_here.add(tid)
                    if mapped:
                        ent = ImageEntry(img, ds_name, sp_key)
                        ent.mapped_boxes = mapped
                        ent.targets_hit = targets_here
                        candidates.append(ent)

        if not candidates:
            QMessageBox.warning(self, "No matches", "No images contain mapped classes.")
            return

        # Pass 2: greedy pick images to satisfy per-target quotas.
        # An image that hits multiple targets helps all of them at once.
        selected: List[ImageEntry] = []
        for ent in candidates:
            if not ent.targets_hit:
                continue
            # If all targets this image would help are already at/over quota, skip.
            needs = [
                tid for tid in ent.targets_hit
                if quotas.get(tid, 0) == 0 or picked_per_target.get(tid, 0) < quotas.get(tid, 0)
            ]
            if not needs:
                continue
            selected.append(ent)
            # credit all targets it contributes to
            for tid in ent.targets_hit:
                if quotas.get(tid, 0) != 0:
                    picked_per_target[tid] = picked_per_target.get(tid, 0) + 1

            # Optional early break if ALL targets with nonzero quotas are satisfied
            all_ok = True
            for tid, q in quotas.items():
                if q != 0 and picked_per_target.get(tid, 0) < q:
                    all_ok = False
                    break
            if all_ok:
                break

        # Pass 3: copy each selected image ONCE and write ALL mapped boxes in it
        for ent in selected:
            out_split = "val" if ent.src_split in ("val", "eval") else ent.src_split
            dst_img = copy_unique(ent.src_img, out / out_split / "images", prefix=ent.src_ds)
            dst_txt = (out / out_split / "labels" / dst_img.stem).with_suffix(".txt")
            if ent.mapped_boxes:
                write_yolo_txt(dst_txt, ent.mapped_boxes)
            else:
                if dst_txt.exists():
                    dst_txt.unlink(missing_ok=True)

        # Write data.yaml (names = target order by id)
        names_vec = [None] * (max(self.merge_ctrl.model.targets.keys()) + 1)
        for tid, tcls in self.merge_ctrl.model.targets.items():
            if tid < len(names_vec):
                names_vec[tid] = tcls.class_name
        names_vec = [n if n is not None else f"class_{i}" for i, n in enumerate(names_vec)]
        y = {
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "nc": len(names_vec),
            "names": names_vec
        }
        atomic_write_text(out / "data.yaml", yaml.safe_dump(y, sort_keys=False))
        QMessageBox.information(self, "Export complete", f"Merged dataset written to:\n{out}")

# ---------- cv2 imread with Unicode-safe path ----------
def np_from_file(path: Path):
    import numpy as np
    data = np.fromfile(str(path), dtype=np.uint8)
    return data
