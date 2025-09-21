from __future__ import annotations
from typing import Optional
from PySide6.QtWidgets import QGraphicsPathItem, QGraphicsItem
from PySide6.QtGui import QPainterPath, QPen, QColor
from PySide6.QtCore import QPointF, Qt

class EdgeItem(QGraphicsPathItem):
    """
    Cubic curve between two ports; updates itself when endpoints move.
    """
    def __init__(self, src_port: QGraphicsItem, dst_port: QGraphicsItem, color=QColor("#5b9bd5")):
        super().__init__()
        self.setZValue(-1)
        self.setPen(QPen(color, 2))
        self.src_port = src_port
        self.dst_port = dst_port
        self._update_path()

    def _anchor(self, port: QGraphicsItem) -> QPointF:
        return port.scenePos()  # draw from port center

    def _update_path(self):
        p1 = self._anchor(self.src_port)
        p2 = self._anchor(self.dst_port)
        dx = abs(p2.x() - p1.x())
        c1 = QPointF(p1.x() + dx * 0.5, p1.y())
        c2 = QPointF(p2.x() - dx * 0.5, p2.y())
        path = QPainterPath(p1)
        path.cubicTo(c1, c2, p2)
        self.setPath(path)

    def advance(self, phase: int):
        # called by scene; keep path fresh if endpoints moved
        self._update_path()
        return super().advance(phase)
