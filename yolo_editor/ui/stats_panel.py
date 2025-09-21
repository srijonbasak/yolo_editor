from __future__ import annotations
from typing import Dict, List
from PySide6.QtWidgets import QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox, QFormLayout

class StatsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        self.grp_split = QGroupBox("Split counts")
        self.tbl_split = QTableWidget(0, 2)
        self.tbl_split.setHorizontalHeaderLabels(["Split", "#images"])
        self.tbl_split.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        fs = QFormLayout(self.grp_split); fs.addRow(self.tbl_split)
        lay.addWidget(self.grp_split)

        self.grp_cls = QGroupBox("Class coverage (#images containing class)")
        self.tbl_cls = QTableWidget(0, 2)
        self.tbl_cls.setHorizontalHeaderLabels(["Class", "#images"])
        self.tbl_cls.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        fc = QFormLayout(self.grp_cls); fc.addRow(self.tbl_cls)
        lay.addWidget(self.grp_cls)
        lay.addStretch(1)

    def set_split_counts(self, counts: Dict[str, int]):
        self.tbl_split.setRowCount(0)
        for s in ("train","val","test"):
            if s not in counts: continue
            r = self.tbl_split.rowCount(); self.tbl_split.insertRow(r)
            self.tbl_split.setItem(r, 0, QTableWidgetItem(s))
            self.tbl_split.setItem(r, 1, QTableWidgetItem(str(counts[s])))

    def set_class_counts(self, names: List[str], counts: List[int]):
        self.tbl_cls.setRowCount(0)
        for i, n in enumerate(names):
            r = self.tbl_cls.rowCount(); self.tbl_cls.insertRow(r)
            self.tbl_cls.setItem(r, 0, QTableWidgetItem(f"{i}: {n}"))
            self.tbl_cls.setItem(r, 1, QTableWidgetItem(str(counts[i] if i < len(counts) else 0)))
