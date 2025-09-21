from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QPixmap, QPen, QBrush, QPainter, QCursor
from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QGraphicsRectItem, QGraphicsItem
)

HANDLE = 10.0     # px
MIN_W = 4.0       # px
MIN_H = 4.0


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


class CornerHandle(QGraphicsRectItem):
    """Small square handle used to resize BBoxItem (safe, no itemChange recursion)."""
    def __init__(self, parent_box: "BBoxItem", role: str):
        super().__init__(-HANDLE/2.0, -HANDLE/2.0, HANDLE, HANDLE, parent_box)
        self.box = parent_box
        self.role = role  # 'tl','tr','bl','br'
        self.setBrush(QBrush(Qt.white))
        self.setPen(QPen(Qt.black, 1))
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setAcceptedMouseButtons(Qt.LeftButton)
        self.setCursor({
            "tl": QCursor(Qt.SizeFDiagCursor),
            "tr": QCursor(Qt.SizeBDiagCursor),
            "bl": QCursor(Qt.SizeBDiagCursor),
            "br": QCursor(Qt.SizeFDiagCursor),
        }[role])
        self.setZValue(10)

    def mouseMoveEvent(self, event):
        # Event position is in handle local coords -> map to parent (box) coords
        parent_pos: QPointF = self.mapToParent(event.pos())
        r = self.box.rect()

        # Clamp new rect based on which corner we drag
        if self.role == "tl":
            new_left = min(parent_pos.x(), r.right() - MIN_W)
            new_top  = min(parent_pos.y(), r.bottom() - MIN_H)
            r = QRectF(new_left, new_top, r.right() - new_left, r.bottom() - new_top)
        elif self.role == "tr":
            new_right = max(parent_pos.x(), r.left() + MIN_W)
            new_top   = min(parent_pos.y(), r.bottom() - MIN_H)
            r = QRectF(r.left(), new_top, new_right - r.left(), r.bottom() - new_top)
        elif self.role == "bl":
            new_left   = min(parent_pos.x(), r.right() - MIN_W)
            new_bottom = max(parent_pos.y(), r.top() + MIN_H)
            r = QRectF(new_left, r.top(), r.right() - new_left, new_bottom - r.top())
        elif self.role == "br":
            new_right  = max(parent_pos.x(), r.left() + MIN_W)
            new_bottom = max(parent_pos.y(), r.top() + MIN_H)
            r = QRectF(r.left(), r.top(), new_right - r.left(), new_bottom - r.top())

        self.box.setRect(r.normalized())
        self.box.sync_handles()
        # do not let the handle move independently (the box just moved it)
        event.accept()

    # prevent Qt from trying to set a new pos we don't want
    def itemChange(self, change, value):
        return super().itemChange(change, value)


class BBoxItem(QGraphicsRectItem):
    def __init__(self, rect: QRectF, cls_id: int, label: str | None = None):
        super().__init__(rect)
        self.cls_id = cls_id
        self.label = label or str(cls_id)
        self.setFlags(
            QGraphicsRectItem.ItemIsSelectable |
            QGraphicsRectItem.ItemIsMovable |
            QGraphicsRectItem.ItemSendsGeometryChanges
        )
        self.setPen(QPen(Qt.white, 2))
        self.setBrush(QBrush(Qt.transparent))
        self.setZValue(5)
        self.setAcceptHoverEvents(True)

        # four resize handles (children). Hidden until selected.
        self.h_tl = CornerHandle(self, "tl")
        self.h_tr = CornerHandle(self, "tr")
        self.h_bl = CornerHandle(self, "bl")
        self.h_br = CornerHandle(self, "br")
        self.sync_handles()
        self.set_handles_visible(False)

    def set_handles_visible(self, vis: bool):
        for h in (self.h_tl, self.h_tr, self.h_bl, self.h_br):
            h.setVisible(vis)

    def sync_handles(self):
        """Position handles on the corners (in box/item coordinates)."""
        r = self.rect()
        self.h_tl.setPos(QPointF(r.left(),  r.top()))
        self.h_tr.setPos(QPointF(r.right(), r.top()))
        self.h_bl.setPos(QPointF(r.left(),  r.bottom()))
        self.h_br.setPos(QPointF(r.right(), r.bottom()))

    def itemChange(self, change, value):
        # keep handles aligned when user moves the whole box or selection toggles
        if change == QGraphicsItem.ItemSelectedHasChanged:
            self.set_handles_visible(bool(value))
        if change in (QGraphicsItem.ItemPositionHasChanged, QGraphicsItem.ItemTransformHasChanged):
            self.sync_handles()
        return super().itemChange(change, value)

    def setRect(self, *args):
        super().setRect(*args)
        self.sync_handles()

    def paint(self, painter, option, widget=None):
        pen = QPen(Qt.green if self.isSelected() else Qt.white, 2)
        painter.setPen(pen)
        painter.setBrush(Qt.transparent)
        super().paint(painter, option, widget)

        # draw a label tag at top-left of the rect (in item coords)
        r = self.rect()
        tag = QRectF(r.left(), r.top() - 18, max(40, len(self.label) * 8), 18)
        painter.fillRect(tag, Qt.black)
        painter.setPen(QPen(Qt.yellow))
        painter.drawText(tag.adjusted(4, 0, -4, 0), Qt.AlignVCenter | Qt.AlignLeft, self.label)


class ImageView(QGraphicsView):
    def __init__(self, scene: ImageScene):
        super().__init__(scene)
        self.setRenderHints(self.renderHints() | QPainter.SmoothPixmapTransform | QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

    def wheelEvent(self, e):
        factor = 1.15 if e.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)
