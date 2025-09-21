from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QPixmap, QPen, QBrush
from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem
)

class ImageScene(QGraphicsScene):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_item: QGraphicsPixmapItem | None = None
        self.add_mode = False
        self._start = None
        self.temp: QGraphicsRectItem | None = None

    def set_image(self, pix: QPixmap):
        self.clear()
        self.image_item = self.addPixmap(pix)
        self.setSceneRect(pix.rect())

    def enter_add_mode(self):
        self.add_mode = True
        self._start = None
        if self.temp:
            self.removeItem(self.temp)
            self.temp = None

    def exit_add_mode(self):
        self.add_mode = False
        self._start = None
        if self.temp:
            self.removeItem(self.temp)
            self.temp = None

    def mousePressEvent(self, e):
        if self.add_mode and self.image_item:
            self._start = e.scenePos()
            self.temp = self.addRect(QRectF(self._start, self._start), QPen(Qt.yellow, 1))
        else:
            super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self.add_mode and self._start and self.temp:
            rect = QRectF(self._start, e.scenePos()).normalized()
            self.temp.setRect(rect)
        else:
            super().mouseMoveEvent(e)

class BBoxItem(QGraphicsRectItem):
    def __init__(self, rect: QRectF, cls_id: int, label: str | None = None):
        super().__init__(rect)
        self.cls_id = cls_id
        self.label = label or str(cls_id)
        self.setFlags(QGraphicsRectItem.ItemIsSelectable | QGraphicsRectItem.ItemIsMovable)
        self.setPen(QPen(Qt.white, 2))
        self.setBrush(QBrush(Qt.transparent))

    def paint(self, painter, option, widget=None):
        pen = QPen(Qt.green if self.isSelected() else Qt.white, 2)
        painter.setPen(pen)
        painter.setBrush(Qt.transparent)
        super().paint(painter, option, widget)
        r = self.rect()
        tag = QRectF(r.left(), r.top() - 18, max(40, len(self.label) * 8), 18)
        painter.fillRect(tag, Qt.black)
        painter.setPen(QPen(Qt.yellow))
        painter.drawText(tag.adjusted(4, 0, -4, 0), Qt.AlignVCenter | Qt.AlignLeft, self.label)

class ImageView(QGraphicsView):
    def __init__(self, scene: ImageScene):
        super().__init__(scene)
        self.setRenderHints(self.renderHints() | self.RenderHint.SmoothPixmapTransform | self.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

    def wheelEvent(self, e):
        factor = 1.15 if e.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)
