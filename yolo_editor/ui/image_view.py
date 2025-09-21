from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Callable
from pathlib import Path

from PySide6.QtCore import Qt, QRectF, QPointF, Signal
from PySide6.QtGui import QPixmap, QImage, QPen, QBrush, QAction
from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem,
    QGraphicsItem, QWidget, QMenu
)

def qimage_from_cv_bgr(bgr) -> QImage:
    import numpy as np
    rgb = bgr[..., ::-1].copy()
    h, w, c = rgb.shape
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()

@dataclass
class Box:
    cls: int
    cx: float
    cy: float
    w: float
    h: float
    def to_rect(self, w_img: int, h_img: int) -> QRectF:
        x = (self.cx - self.w / 2) * w_img
        y = (self.cy - self.h / 2) * h_img
        return QRectF(x, y, self.w * w_img, self.h * h_img)
    @staticmethod
    def from_rect(cls: int, r: QRectF, w_img: int, h_img: int) -> "Box":
        cx = (r.x() + r.width()/2) / w_img
        cy = (r.y() + r.height()/2) / h_img
        ww = r.width() / w_img
        hh = r.height() / h_img
        return Box(cls=cls, cx=float(cx), cy=float(cy), w=float(ww), h=float(hh))

class Handle(QGraphicsRectItem):
    def __init__(self, parent_rect: "BBoxItem", pos: str, size: float = 8.0):
        super().__init__(-size/2, -size/2, size, size, parent_rect)
        self.setBrush(QBrush(Qt.white))
        self.setPen(QPen(Qt.black, 1))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setZValue(10)
        self.pos_id = pos
        self.parent_rect = parent_rect
    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            self.parent_rect.on_handle_moved(self)
        return super().itemChange(change, value)

class BBoxItem(QGraphicsRectItem):
    def __init__(self, rect: QRectF, cls: int, color=Qt.red):
        super().__init__(rect)
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setPen(QPen(color, 2))
        self.setBrush(QBrush(Qt.transparent))
        self.cls = cls
        self._handles = {pid: Handle(self, pid) for pid in ("tl","tr","bl","br")}
        self.update_handles()
    def on_handle_moved(self, handle: Handle):
        r = self.rect()
        p = handle.pos()
        if handle.pos_id == "tl":
            new = QRectF(p.x(), p.y(), r.right()-p.x(), r.bottom()-p.y())
        elif handle.pos_id == "tr":
            new = QRectF(r.left(), p.y(), p.x()-r.left(), r.bottom()-p.y())
        elif handle.pos_id == "bl":
            new = QRectF(p.x(), r.top(), r.right()-p.x(), p.y()-r.top())
        else:
            new = QRectF(r.left(), r.top(), p.x()-r.left(), p.y()-r.top())
        if new.width() < 1: new.setWidth(1)
        if new.height() < 1: new.setHeight(1)
        self.setRect(new.normalized()); self.update_handles()
    def itemChange(self, change, value):
        if change in (
            QGraphicsItem.GraphicsItemChange.ItemPositionChange,
            QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged,
            QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged,
            QGraphicsItem.GraphicsItemChange.ItemTransformHasChanged
        ):
            self.update_handles()
        return super().itemChange(change, value)
    def update_handles(self):
        r = self.rect()
        self._handles["tl"].setPos(r.topLeft())
        self._handles["tr"].setPos(r.topRight())
        self._handles["bl"].setPos(r.bottomLeft())
        self._handles["br"].setPos(r.bottomRight())

class ImageView(QGraphicsView):
    requestPrev = Signal()
    requestNext = Signal()
    boxSelectionChanged = Signal()
    contextDelete = Signal()
    contextToCurrentClass = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHints(self.renderHints() | self.RenderHint.Antialiasing | self.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.img_item: Optional[QGraphicsPixmapItem] = None
        self.image_path: Optional[Path] = None
        self.img_w = 1
        self.img_h = 1
        self._adding = False
        self._start_pos: Optional[QPointF] = None
        self._current_class: int = 0
        self._class_names: List[str] = []
        self._on_status: Optional[Callable[[str], None]] = None

        self.act_delete = QAction("Delete selected", self)
        self.act_delete.triggered.connect(self._emit_delete)
        self.act_to_cls = QAction("Set to current class", self)
        self.act_to_cls.triggered.connect(self._emit_to_current)

    def set_status_sink(self, fn: Callable[[str], None]): self._on_status = fn
    def set_current_class(self, cid: int, names: List[str]): self._current_class = int(cid); self._class_names = names or []

    def show_image_bgr(self, path: Path, bgr_img):
        self.scene.clear()
        qimg = qimage_from_cv_bgr(bgr_img)
        self.img_w, self.img_h = qimg.width(), qimg.height()
        pix = QPixmap.fromImage(qimg)
        self.img_item = self.scene.addPixmap(pix)
        self.image_path = Path(path)
        self.fitInView(self.img_item, Qt.AspectRatioMode.KeepAspectRatio)

    def add_box_norm(self, box: Box, color=Qt.red):
        rect = box.to_rect(self.img_w, self.img_h)
        self.scene.addItem(BBoxItem(rect, cls=box.cls, color=color))

    def clear_boxes(self):
        for it in list(self.scene.items()):
            if isinstance(it, BBoxItem):
                self.scene.removeItem(it)

    def selected_boxes(self) -> List[BBoxItem]:
        return [it for it in self.scene.selectedItems() if isinstance(it, BBoxItem)]

    def all_boxes(self) -> List[BBoxItem]:
        return [it for it in self.scene.items() if isinstance(it, BBoxItem)]

    def get_boxes_as_norm(self) -> List[Box]:
        out = []
        for it in self.all_boxes():
            r_scene = it.mapRectToScene(it.rect())
            out.append(Box.from_rect(it.cls, r_scene, self.img_w, self.img_h))
        return out

    def keyPressEvent(self, e):
        k = e.key()
        if k in (Qt.Key_Left, Qt.Key.Key_A) and (e.modifiers() == Qt.NoModifier):
            self.requestPrev.emit(); e.accept(); return
        if k in (Qt.Key_Right, Qt.Key.Key_D) and (e.modifiers() == Qt.NoModifier):
            self.requestNext.emit(); e.accept(); return
        if k == Qt.Key.Key_Delete:
            self._emit_delete(); e.accept(); return
        if k == Qt.Key.Key_C:
            self._emit_to_current(); e.accept(); return
        if k == Qt.Key.Key_F:
            if self.img_item:
                self.fitInView(self.img_item, Qt.AspectRatioMode.KeepAspectRatio)
                e.accept(); return
        if k == Qt.Key.Key_1:
            self.resetTransform(); e.accept(); return
        super().keyPressEvent(e)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and (e.modifiers() & Qt.KeyboardModifier.ControlModifier):
            self._adding = True
            self._start_pos = self.mapToScene(e.pos())
            e.accept(); return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._adding and self._start_pos is not None:
            p = self.mapToScene(e.pos())
            rect = QRectF(self._start_pos, p).normalized()
            self._draw_temp_rect(rect)
            e.accept(); return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self._adding and self._start_pos is not None:
            p = self.mapToScene(e.pos())
            rect = QRectF(self._start_pos, p).normalized()
            self._adding = False
            self._start_pos = None
            self._clear_temp_rect()
            if rect.width() > 4 and rect.height() > 4:
                self.scene.addItem(BBoxItem(rect, cls=self._current_class))
                if self._on_status:
                    name = self._class_names[self._current_class] if 0 <= self._current_class < len(self._class_names) else str(self._current_class)
                    self._on_status(f"Added box → class {self._current_class} ({name})")
            e.accept(); return
        super().mouseReleaseEvent(e)

    def contextMenuEvent(self, e):
        m = QMenu(self)
        m.addAction(self.act_to_cls)
        m.addSeparator()
        m.addAction(self.act_delete)
        m.exec(e.globalPos())

    def _draw_temp_rect(self, rect: QRectF):
        if getattr(self, "_temp_rect", None) is None:
            self._temp_rect = self.scene.addRect(rect, QPen(Qt.green, 2, Qt.PenStyle.DashLine))
        else:
            self._temp_rect.setRect(rect)

    def _clear_temp_rect(self):
        if getattr(self, "_temp_rect", None) is not None:
            self.scene.removeItem(self._temp_rect)
            self._temp_rect = None

    def _emit_delete(self):
        for it in self.selected_boxes():
            self.scene.removeItem(it)
        self.contextDelete.emit()

    def _emit_to_current(self):
        for it in self.selected_boxes():
            it.cls = self._current_class
        self.contextToCurrentClass.emit()
