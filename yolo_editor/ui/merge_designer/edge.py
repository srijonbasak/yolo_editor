from __future__ import annotations
from PySide6.QtWidgets import QGraphicsPathItem, QGraphicsItem
from PySide6.QtGui import QPainterPath, QPen, QColor
from PySide6.QtCore import QPointF

class EdgeItem(QGraphicsPathItem):
    """
    Cubic curve between two ports/items; if dst_item is None, use a floating point.
    """
    def __init__(self, src_item: QGraphicsItem, dst_item: QGraphicsItem | None, color=QColor("#5b9bd5")):
        super().__init__()
        self.setZValue(-1)
        self.setPen(QPen(color, 2))
        self.src_item = src_item
        self.dst_item = dst_item
        self.floating_pos: QPointF | None = None
        self._update_path()

    def set_floating(self, pos: QPointF):
        self.floating_pos = QPointF(pos)
        self._update_path()

    def attach_dst(self, dst_item: QGraphicsItem):
        self.dst_item = dst_item
        self.floating_pos = None
        self._update_path()

    def _anchor(self, item: QGraphicsItem | None) -> QPointF:
        if item is None:
            return self.floating_pos if self.floating_pos is not None else QPointF()
        return item.scenePos()

    def _update_path(self):
        p1 = self._anchor(self.src_item)
        p2 = self._anchor(self.dst_item)
        if p1 is None or p2 is None:
            return
        dx = abs(p2.x() - p1.x())
        c1 = QPointF(p1.x() + dx * 0.5, p1.y())
        c2 = QPointF(p2.x() - dx * 0.5, p2.y())
        path = QPainterPath(p1)
        path.cubicTo(c1, c2, p2)
        self.setPath(path)

    def advance(self, phase: int):
        self._update_path()
        return super().advance(phase)
