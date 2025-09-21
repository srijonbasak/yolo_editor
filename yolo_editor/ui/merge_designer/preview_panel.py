from __future__ import annotations
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem, QHeaderView, QTextEdit, QGroupBox
)
from PySide6.QtCore import Qt

class PreviewPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)

        self.group_targets = QGroupBox("Targets Overview")
        gl = QVBoxLayout(self.group_targets)
        self.tbl_targets = QTableWidget(0, 4)
        self.tbl_targets.setHorizontalHeaderLabels(["Target Index", "Name", "Supply", "Selected / Quota"])
        self.tbl_targets.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        gl.addWidget(self.tbl_targets)
        root.addWidget(self.group_targets)

        self.group_edges = QGroupBox("Edges (Per Source Class)")
        gl2 = QVBoxLayout(self.group_edges)
        self.tbl_edges = QTableWidget(0, 4)
        self.tbl_edges.setHorizontalHeaderLabels(["Target", "Dataset ID", "Source Class ID", "Selected / Supply"])
        self.tbl_edges.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        gl2.addWidget(self.tbl_edges)
        root.addWidget(self.group_edges)

        self.lbl_warn = QLabel("Warnings")
        self.txt_warn = QTextEdit()
        self.txt_warn.setReadOnly(True)
        root.addWidget(self.lbl_warn)
        root.addWidget(self.txt_warn)
        root.addStretch(1)

        self._target_names = {}

    def set_target_names(self, names_dict: dict[int, str]):
        self._target_names = dict(names_dict)

    def set_preview(self, preview_supply: dict, preview_edges: dict, warnings: list[str]):
        self.tbl_targets.setRowCount(0)
        for k, stats in sorted(preview_supply.items(), key=lambda kv: int(kv[0]) if isinstance(kv[0], str) else kv[0]):
            t = int(k) if isinstance(k, str) else k
            r = self.tbl_targets.rowCount(); self.tbl_targets.insertRow(r)
            name = self._target_names.get(t, f"target-{t}")
            supply = stats.get("supply", 0)
            selected = stats.get("selected", 0)
            quota = stats.get("quota", supply)
            self.tbl_targets.setItem(r, 0, QTableWidgetItem(str(t)))
            self.tbl_targets.setItem(r, 1, QTableWidgetItem(name))
            self.tbl_targets.setItem(r, 2, QTableWidgetItem(str(supply)))
            it = QTableWidgetItem(f"{selected} / {quota}")
            if selected < quota: it.setForeground(Qt.red)
            self.tbl_targets.setItem(r, 3, it)

        self.tbl_edges.setRowCount(0)
        for k, rows in preview_edges.items():
            t = int(k) if isinstance(k, str) else k
            tname = self._target_names.get(t, f"target-{t}")
            for edge, supply, taken in rows:
                dsid, src_cls = edge
                r = self.tbl_edges.rowCount(); self.tbl_edges.insertRow(r)
                self.tbl_edges.setItem(r, 0, QTableWidgetItem(f"{t} ({tname})"))
                self.tbl_edges.setItem(r, 1, QTableWidgetItem(str(dsid)))
                self.tbl_edges.setItem(r, 2, QTableWidgetItem(str(src_cls)))
                self.tbl_edges.setItem(r, 3, QTableWidgetItem(f"{taken} / {supply}"))

        self.txt_warn.clear()
        self.txt_warn.setPlainText("\n".join(warnings) if warnings else "No warnings.")
