from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Callable
from pathlib import Path

from PySide6.QtCore import Qt, QRectF, QPointF, Signal
from PySide6.QtGui import QPixmap, QImage, QPen, QBrush, QAction, QPainter
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

class BBoxItem(QGraphicsRectItem):
    def __init__(self, rect: QRectF, cls: int, color=Qt.red, on_change: Optional[Callable[[], None]] = None):
        super().__init__(rect)
        self._on_change = on_change
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setPen(QPen(color, 2))
        self.setBrush(QBrush(Qt.transparent))
        self.cls = cls

        # guard to avoid infinite recursion when we reposition handles ourselves
        self._suppress_handle_feedback: bool = False

        # Create 4 corner handles
        self._handles: dict[str, Handle] = {}
        for pid in ("tl", "tr", "bl", "br"):
            h = Handle(self, pid)
            self._handles[pid] = h
        self.update_handles()

    def on_handle_moved(self, handle: "Handle"):
        """Called when a user drags a corner handle; update the rect."""
        r = self.rect()
        p = handle.pos()
        if handle.pos_id == "tl":
            new = QRectF(p.x(), p.y(), r.right() - p.x(), r.bottom() - p.y())
        elif handle.pos_id == "tr":
            new = QRectF(r.left(), p.y(), p.x() - r.left(), r.bottom() - p.y())
        elif handle.pos_id == "bl":
            new = QRectF(p.x(), r.top(), r.right() - p.x(), p.y() - r.top())
        else:  # "br"
            new = QRectF(r.left(), r.top(), p.x() - r.left(), p.y() - r.top())

        # enforce minimum size
        if new.width() < 1:  new.setWidth(1)
        if new.height() < 1: new.setHeight(1)

        # Update rect and reposition handles WITHOUT sending feedback back from the handles
        self.setRect(new.normalized())
        self._suppress_handle_feedback = True
        try:
            self.update_handles()
        finally:
            self._suppress_handle_feedback = False
        self._emit_change()

    def itemChange(self, change, value):
        # Keep handles in place when the whole rect is moved/selected/transformed
        if change in (
            QGraphicsItem.GraphicsItemChange.ItemPositionChange,
            QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged,
            QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged,
            QGraphicsItem.GraphicsItemChange.ItemTransformHasChanged
        ):
            self._suppress_handle_feedback = True
            try:
                self.update_handles()
            finally:
                self._suppress_handle_feedback = False
        result = super().itemChange(change, value)
        if change in (
            QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged,
            QGraphicsItem.GraphicsItemChange.ItemTransformHasChanged
        ):
            self._emit_change()
        return result

    def update_handles(self):
        """Reposition corner handles to rect corners."""
        r = self.rect()
        # Move handles programmatically; their itemChange will fire,
        # but Handle.itemChange checks parent._suppress_handle_feedback.
        self._handles["tl"].setPos(r.topLeft())
        self._handles["tr"].setPos(r.topRight())
        self._handles["bl"].setPos(r.bottomLeft())
        self._handles["br"].setPos(r.bottomRight())

    def _emit_change(self):
        if self._on_change:
            self._on_change()

class Handle(QGraphicsRectItem):
    def __init__(self, parent_rect: BBoxItem, pos: str, size: float = 8.0):
        super().__init__(-size/2, -size/2, size, size, parent_rect)
        self.setBrush(QBrush(Qt.white))
        self.setPen(QPen(Qt.black, 1))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setZValue(10)
        self.pos_id = pos
        self.parent_rect = parent_rect

    def itemChange(self, change, value):
        # When we are moved by the user, tell the parent to reshape the bbox.
        # But if parent is currently updating us programmatically, do nothing.
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            if not self.parent_rect._suppress_handle_feedback:
                # Call parent to recompute the bbox
                self.parent_rect.on_handle_moved(self)
        return super().itemChange(change, value)

class ImageView(QGraphicsView):
    requestPrev = Signal()
    requestNext = Signal()
    boxSelectionChanged = Signal()
    boxesChanged = Signal()
    contextDelete = Signal()
    contextToCurrentClass = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHints(self.renderHints() | QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.scene.selectionChanged.connect(self._on_scene_selection_changed)

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
        self.scene.addItem(BBoxItem(rect, cls=box.cls, color=color, on_change=self._notify_boxes_changed))

    def clear_boxes(self):
        changed = False
        for it in list(self.scene.items()):
            if isinstance(it, BBoxItem):
                self.scene.removeItem(it)
                changed = True
        if changed:
            self._notify_boxes_changed()

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
        if k in (Qt.Key_Left, Qt.Key.Key_A, Qt.Key.Key_P) and (e.modifiers() == Qt.NoModifier):
            self.requestPrev.emit(); e.accept(); return
        if k in (Qt.Key_Right, Qt.Key.Key_D, Qt.Key.Key_N) and (e.modifiers() == Qt.NoModifier):
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
        if e.button() == Qt.MouseButton.LeftButton:
            target = self.itemAt(e.pos())
            allow_blank = (target is None) or (target is self.img_item)
            if (e.modifiers() & Qt.KeyboardModifier.ControlModifier) or allow_blank:
                self._adding = True
                self._start_pos = self.mapToScene(e.pos())
                e.accept()
                return
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
                self.scene.addItem(BBoxItem(rect, cls=self._current_class, on_change=self._notify_boxes_changed))
                self._notify_boxes_changed()
                if self._on_status:
                    name = self._class_names[self._current_class] if 0 <= self._current_class < len(self._class_names) else str(self._current_class)
                    self._on_status(f"Added box -> class {self._current_class} ({name})")
            e.accept(); return
        super().mouseReleaseEvent(e)

    def wheelEvent(self, e):
        if e.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.2 if e.angleDelta().y() > 0 else 1 / 1.2
            self.scale(factor, factor)
            e.accept()
            return
        super().wheelEvent(e)

    def contextMenuEvent(self, e):
        m = QMenu(self)
        m.addAction(self.act_to_cls)
        m.addSeparator()
        m.addAction(self.act_delete)
        m.exec(e.globalPos())

    def _notify_boxes_changed(self):
        self.boxesChanged.emit()

    def _on_scene_selection_changed(self):
        self.boxSelectionChanged.emit()
        if self._on_status:
            count = len(self.selected_boxes())
            if count:
                word = 'box' if count == 1 else 'boxes'
                self._on_status(f"Selected {count} {word}")

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
        selected = self.selected_boxes()
        if not selected:
            return
        for it in selected:
            self.scene.removeItem(it)
        self.contextDelete.emit()
        self._notify_boxes_changed()
        if self._on_status:
            count = len(selected)
            word = 'box' if count == 1 else 'boxes'
            self._on_status(f"Deleted {count} {word}")

    def _emit_to_current(self):
        selected = self.selected_boxes()
        if not selected:
            return
        for it in selected:
            it.cls = self._current_class
        self.contextToCurrentClass.emit()
        self._notify_boxes_changed()
        if self._on_status:
            name = self._class_names[self._current_class] if 0 <= self._current_class < len(self._class_names) else str(self._current_class)
            count = len(selected)
            word = 'box' if count == 1 else 'boxes'
            self._on_status(f"Set {count} {word} -> class {self._current_class} ({name})")
