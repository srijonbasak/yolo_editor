from __future__ import annotations
from PySide6.QtWidgets import QGraphicsScene
from PySide6.QtCore import Signal, QObject

class SceneSignals(QObject):
    edgeAdded = Signal(object)     # EdgeItem
    edgeRemoved = Signal(object)   # EdgeItem

class MergeScene(QGraphicsScene):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.sigs = SceneSignals()
        # Set up scene for better interaction
        self.setItemIndexMethod(QGraphicsScene.ItemIndexMethod.NoIndex)
