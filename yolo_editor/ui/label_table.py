from __future__ import annotations
from typing import List, Tuple
from PySide6.QtWidgets import QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QPushButton, QHBoxLayout, QComboBox, QLabel

Row = Tuple[int, float, float, float, float]

class LabelTable(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.tbl = QTableWidget(0, 6)
        self.tbl.setHorizontalHeaderLabels(["#", "Class", "cx", "cy", "w", "h"])
        self.cmb_class = QComboBox()
        self.btn_apply = QPushButton("Set selected → class")
        self.btn_delete = QPushButton("Delete selected row(s)")

        top = QHBoxLayout()
        top.addWidget(QLabel("Current class:"))
        top.addWidget(self.cmb_class, 1)
        top.addWidget(self.btn_apply)
        top.addWidget(self.btn_delete)

        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self.tbl)

        self.btn_apply.clicked.connect(self._apply_class)
        self.btn_delete.clicked.connect(self._del_rows)

    def set_class_names(self, names: List[str]):
        self.cmb_class.clear()
        for i, n in enumerate(names or []):
            self.cmb_class.addItem(f"{i}: {n}", i)

    def set_rows(self, rows: List[Row], class_names: List[str]):
        self.tbl.setRowCount(0)
        for i, (c, x, y, w, h) in enumerate(rows):
            r = self.tbl.rowCount(); self.tbl.insertRow(r)
            self.tbl.setItem(r, 0, QTableWidgetItem(str(i+1)))
            self.tbl.setItem(r, 1, QTableWidgetItem(f"{c} ({class_names[c] if 0<=c<len(class_names) else c})"))
            self.tbl.setItem(r, 2, QTableWidgetItem(f"{x:.6f}"))
            self.tbl.setItem(r, 3, QTableWidgetItem(f"{y:.6f}"))
            self.tbl.setItem(r, 4, QTableWidgetItem(f"{w:.6f}"))
            self.tbl.setItem(r, 5, QTableWidgetItem(f"{h:.6f}"))

    def read_rows(self) -> List[Row]:
        out: List[Row] = []
        for r in range(self.tbl.rowCount()):
            cell = self.tbl.item(r, 1).text().split()[0]
            c = int(cell)
            x = float(self.tbl.item(r, 2).text())
            y = float(self.tbl.item(r, 3).text())
            w = float(self.tbl.item(r, 4).text())
            h = float(self.tbl.item(r, 5).text())
            out.append((c, x, y, w, h))
        return out

    def _apply_class(self):
        idx = self.cmb_class.currentData()
        if idx is None: return
        for r in {i.row() for i in self.tbl.selectedIndexes()}:
            ccell = self.tbl.item(r, 1)
            parts = ccell.text().split()
            parts[0] = str(int(idx))
            ccell.setText(" ".join(parts))

    def _del_rows(self):
        rows = sorted({i.row() for i in self.tbl.selectedIndexes()}, reverse=True)
        for r in rows:
            self.tbl.removeRow(r)
