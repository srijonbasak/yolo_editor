from __future__ import annotations
from typing import List, Dict
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton, QHBoxLayout, QSpinBox, QLineEdit

class MergePalette(QWidget):
    """
    Left panel listing:
    - available datasets (from the app)
    - target ops: add target class with optional quota
    Emits callbacks provided by MainWindow to spawn nodes.
    """
    def __init__(self, on_spawn_dataset, on_spawn_target_class):
        super().__init__()
        self._on_spawn_dataset = on_spawn_dataset
        self._on_spawn_target_class = on_spawn_target_class

        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Datasets"))
        self.list = QListWidget()
        lay.addWidget(self.list)

        btn_row = QHBoxLayout()
        self.btn_add_ds = QPushButton("Add to Canvas")
        self.btn_add_ds.clicked.connect(self._spawn_selected_dataset)
        btn_row.addWidget(self.btn_add_ds)
        lay.addLayout(btn_row)

        lay.addWidget(QLabel("New Target Class"))
        row2 = QHBoxLayout()
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("class name (e.g., vehicle)")
        self.quota = QSpinBox()
        self.quota.setRange(0, 10_000_000)
        self.quota.setValue(0)
        self.quota.setToolTip("0 = unlimited")
        self.btn_add_target = QPushButton("Create")
        self.btn_add_target.clicked.connect(self._spawn_target)
        row2.addWidget(self.name_edit)
        row2.addWidget(QLabel("quota"))
        row2.addWidget(self.quota)
        row2.addWidget(self.btn_add_target)
        lay.addLayout(row2)
        lay.addStretch(1)

    def populate(self, dataset_names: List[str]):
        self.list.clear()
        for n in dataset_names:
            QListWidgetItem(n, self.list)

    # callbacks
    def _spawn_selected_dataset(self):
        it = self.list.currentItem()
        if not it:
            return
        self._on_spawn_dataset(it.text())

    def _spawn_target(self):
        name = self.name_edit.text().strip()
        quota = int(self.quota.value())
        if not name:
            return
        self._on_spawn_target_class(name, None if quota == 0 else quota)
        self.name_edit.clear()
        self.quota.setValue(0)
