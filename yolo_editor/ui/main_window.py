from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QMainWindow,
    QStatusBar,
    QTabWidget,
    QWidget,
    QSplitter,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QPushButton,
    QTreeWidget,
    QTableWidget,
    QHeaderView,
    QListWidget,
)

from .image_view import ImageView
from .main_window_presenter import EditorWidgets, MainWindowPresenter, MergeWidgets


class MainWindow(QMainWindow):
    """Application shell that wires UI components to the presenter."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("YOLO Editor - Offline")
        self.resize(1380, 860)
        self.setStatusBar(QStatusBar(self))

        self.tabs = QTabWidget(self)
        self.setCentralWidget(self.tabs)

        self._build_editor_tab()
        merge_widgets = self._build_merge_tab_if_available()

        editor_widgets = EditorWidgets(
            tabs=self.tabs,
            dataset_label=self.lbl_ds,
            split_combo=self.split_combo,
            file_tree=self.file_tree,
            class_combo=self.class_combo,
            save_button=self.btn_save,
            image_view=self.view,
            labels_table=self.tbl,
            stats_list=self.stats,
            status_bar=self.statusBar(),
        )

        self.presenter = MainWindowPresenter(self, editor_widgets, merge_widgets)
        self.presenter.bind()

        self._build_menu()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_editor_tab(self) -> None:
        self.editor_root = QWidget(self)
        splitter = QSplitter(Qt.Orientation.Horizontal, self.editor_root)

        # left column
        left = QWidget(splitter)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(6, 6, 6, 6)
        self.split_combo = QComboBox(left)
        self.file_tree = QTreeWidget(left)
        self.file_tree.setHeaderLabels(["Image Files"])
        left_layout.addWidget(QLabel("Split:", left))
        left_layout.addWidget(self.split_combo)
        left_layout.addWidget(self.file_tree, 1)

        # center column
        center = QWidget(splitter)
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(6, 6, 6, 6)
        bar = QHBoxLayout()
        self.lbl_ds = QLabel("Dataset: -", center)
        self.class_combo = QComboBox(center)
        self.btn_save = QPushButton("Save", center)
        bar.addWidget(self.lbl_ds, 1)
        bar.addWidget(QLabel("Class:", center))
        bar.addWidget(self.class_combo)
        bar.addStretch(1)
        bar.addWidget(self.btn_save)
        self.view = ImageView(center)
        self.view.set_status_sink(lambda msg: self.statusBar().showMessage(msg, 3000))
        center_layout.addLayout(bar)
        center_layout.addWidget(self.view, 1)

        # right column
        right = QWidget(splitter)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(6, 6, 6, 6)
        right_layout.addWidget(QLabel("Labels (YOLO)", right))
        self.tbl = QTableWidget(0, 6, right)
        self.tbl.setHorizontalHeaderLabels(["name", "id", "cx", "cy", "w", "h"])
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tbl.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        right_layout.addWidget(self.tbl, 3)
        right_layout.addWidget(QLabel("Dataset Stats", right))
        self.stats = QListWidget(right)
        right_layout.addWidget(self.stats, 2)

        splitter.addWidget(left)
        splitter.addWidget(center)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)

        layout = QVBoxLayout(self.editor_root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        self.tabs.addTab(self.editor_root, "Label Editor")

    def _build_merge_tab_if_available(self) -> Optional[MergeWidgets]:
        try:
            from .merge_designer.canvas import MergeCanvas
            from .merge_designer.controller import MergeController
            from .merge_designer.palette import MergePalette
        except Exception:
            return None

        self.merge_ctrl = MergeController()
        self.merge_canvas = MergeCanvas(self.merge_ctrl)
        self.merge_palette = MergePalette(
            on_spawn_dataset=self._on_merge_spawn_dataset,
            on_spawn_target_class=self._on_merge_spawn_target,
        )
        self.merge_palette.populate([])
        self.merge_palette.requestLoadDataset.connect(self._on_merge_load_dataset)
        self.merge_palette.requestExportMerged.connect(self._on_merge_export_dataset)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.addWidget(self.merge_palette)
        splitter.addWidget(self.merge_canvas)
        splitter.setStretchFactor(1, 1)
        self.tabs.addTab(splitter, "Merge Designer")

        return MergeWidgets(
            controller=self.merge_ctrl,
            canvas=self.merge_canvas,
            palette=self.merge_palette,
        )

    # ------------------------------------------------------------------
    # Menu wiring
    # ------------------------------------------------------------------
    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")

        act_root = QAction("Open Dataset &Root...", self)
        act_yaml = QAction("Open Dataset &YAML...", self)
        act_quit = QAction("&Quit", self)
        act_root.triggered.connect(self.presenter.open_dataset_root)
        act_yaml.triggered.connect(self.presenter.open_data_yaml)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_root)
        file_menu.addAction(act_yaml)
        file_menu.addSeparator()
        file_menu.addAction(act_quit)

        edit_menu = self.menuBar().addMenu("&Edit")
        act_save = QAction("&Save Labels", self)
        act_prev = QAction("&Prev", self)
        act_next = QAction("&Next", self)
        act_save.setShortcut("S")
        act_prev.setShortcut(Qt.Key_Left)
        act_next.setShortcut(Qt.Key_Right)
        act_save.triggered.connect(self.presenter.save_current_labels)
        act_prev.triggered.connect(self.presenter.go_previous_image)
        act_next.triggered.connect(self.presenter.go_next_image)
        edit_menu.addAction(act_save)
        edit_menu.addAction(act_prev)
        edit_menu.addAction(act_next)

        tools_menu = self.menuBar().addMenu("&Tools")
        act_diag = QAction("Show &Diagnostics...", self)
        act_diag.triggered.connect(self.presenter.show_diagnostics)
        tools_menu.addAction(act_diag)

    # ------------------------------------------------------------------
    # Merge callbacks delegate to presenter lazily
    # ------------------------------------------------------------------
    def _on_merge_spawn_dataset(self, dataset_name: str) -> None:
        if getattr(self, "presenter", None):
            self.presenter.spawn_dataset_node(dataset_name)

    def _on_merge_spawn_target(self, name: str, quota: Optional[int]) -> None:
        if getattr(self, "presenter", None):
            self.presenter.spawn_target_node(name, quota)

    def _on_merge_load_dataset(self) -> None:
        if getattr(self, "presenter", None):
            self.presenter.load_dataset_for_merge()

    def _on_merge_export_dataset(self) -> None:
        if getattr(self, "presenter", None):
            self.presenter.export_merged_dataset()
