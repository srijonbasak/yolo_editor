from __future__ import annotations
from pathlib import Path
from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QTreeWidget, QTreeWidgetItem, QVBoxLayout

class FileTree(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Files"])
        self.tree.itemDoubleClicked.connect(self._on_double)
        lay = QVBoxLayout(self)
        lay.addWidget(self.tree)
        self._open_cb: Callable[[Path], None] | None = None

    def on_open(self, fn: Callable[[Path], None]):
        self._open_cb = fn

    def populate_from_splits(self, splits_map: dict[str, list[Path]]):
        self.tree.clear()
        for split in ("train","val","test"):
            if split not in splits_map: continue
            imgs = splits_map[split]
            if not imgs: continue
            root = QTreeWidgetItem([split])
            self.tree.addTopLevelItem(root)
            for p in imgs:
                item = QTreeWidgetItem([p.name])
                item.setData(0, Qt.ItemDataRole.UserRole, str(p))
                root.addChild(item)
            root.setExpanded(True)

    def _on_double(self, item: QTreeWidgetItem, col: int):
        if not self._open_cb: return
        p = Path(item.data(0, Qt.ItemDataRole.UserRole) or "")
        if p.exists():
            self._open_cb(p)
