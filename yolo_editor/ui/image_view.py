from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Callable, Dict
from pathlib import Path

from PySide6.QtCore import Qt, QRectF, QPointF, Signal
from PySide6.QtGui import QPixmap, QImage, QPen, QBrush, QAction, QPainter, QColor
from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem,
    QGraphicsSimpleTextItem,
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
        # Convert normalized YOLO coordinates to screen coordinates
        x = (self.cx - self.w / 2) * w_img
        y = (self.cy - self.h / 2) * h_img
        width = self.w * w_img
        height = self.h * h_img
        
        # Ensure minimum size to avoid horizontal lines
        width = max(width, 2)
        height = max(height, 2)
        
        return QRectF(x, y, width, height)
    @staticmethod
    def from_rect(cls: int, r: QRectF, w_img: int, h_img: int) -> "Box":
        cx = (r.x() + r.width()/2) / w_img
        cy = (r.y() + r.height()/2) / h_img
        ww = r.width() / w_img
        hh = r.height() / h_img
        return Box(cls=cls, cx=float(cx), cy=float(cy), w=float(ww), h=float(hh))

class BBoxItem(QGraphicsRectItem):
    def __init__(self, rect: QRectF, cls: int, class_name: str = '', color=Qt.red, on_change: Optional[Callable[[], None]] = None):
        scene_rect = rect.normalized()
        width = max(scene_rect.width(), 2)
        height = max(scene_rect.height(), 2)
        scene_rect = QRectF(scene_rect.left(), scene_rect.top(), width, height)
        super().__init__(scene_rect)

        self._on_change = on_change
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setBrush(QBrush(Qt.transparent))
        self.cls = cls
        self.class_name = class_name or str(cls)

        self._pen_color = QColor(color) if isinstance(color, QColor) else QColor(color)
        pen = QPen(self._pen_color, 2)
        pen.setCosmetic(True)
        self.setPen(pen)

        self._label_bg = QGraphicsRectItem(self)
        self._label_bg.setBrush(QBrush(QColor(0, 0, 0, 160)))
        self._label_bg.setPen(QPen(Qt.PenStyle.NoPen))
        self._label_bg.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self._label_bg.setZValue(self.zValue() + 1)

        self._label = QGraphicsSimpleTextItem(self.class_name, self)
        self._label.setBrush(QBrush(Qt.white))
        self._label.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self._label.setZValue(self.zValue() + 2)

        self._suppress_handle_feedback: bool = False
        self._handles: dict[str, Handle] = {}
        for pid in ("tl", "tr", "bl", "br"):
            h = Handle(self, pid)
            self._handles[pid] = h
        self.update_handles()
        self._update_label_position()
    def set_color(self, color):
        qc = QColor(color) if not isinstance(color, QColor) else QColor(color)
        self._pen_color = qc
        pen = QPen(qc, 2)
        pen.setCosmetic(True)
        self.setPen(pen)
        bg = QColor(qc)
        bg.setAlpha(140)
        self._label_bg.setBrush(QBrush(bg))
    def set_class(self, cls: int, class_name: Optional[str] = None):
        self.cls = cls
        if class_name:
            self.class_name = class_name
        else:
            self.class_name = str(cls)
        self._label.setText(self.class_name)
        self._update_label_position()
    def _update_label_position(self):
        rect = self.rect()
        self._label.setPos(rect.left() + 4, rect.top() + 4)
        br = self._label.boundingRect()
        self._label_bg.setRect(0, 0, br.width() + 8, br.height() + 6)
        self._label_bg.setPos(rect.left() + 2, rect.top() + 2)
    def _set_scene_rect(self, scene_rect: QRectF):
        scene_rect = scene_rect.normalized()
        w = max(scene_rect.width(), 2)
        h = max(scene_rect.height(), 2)
        scene_rect = QRectF(scene_rect.left(), scene_rect.top(), w, h)
        self.prepareGeometryChange()
        self.setRect(scene_rect)
        self.update_handles()
        self._update_label_position()
    def on_handle_moved(self, handle: "Handle"):
        scene_rect = self.mapRectToScene(self.rect())
        handle_scene = handle.mapToScene(QPointF(0, 0))
        if handle.pos_id == "tl":
            fixed = scene_rect.bottomRight()
        elif handle.pos_id == "tr":
            fixed = scene_rect.bottomLeft()
        elif handle.pos_id == "bl":
            fixed = scene_rect.topRight()
        else:
            fixed = scene_rect.topLeft()

        new_scene = QRectF(handle_scene, fixed).normalized()
        if new_scene.width() < 2:
            new_scene.setWidth(2)
        if new_scene.height() < 2:
            new_scene.setHeight(2)

        self._suppress_handle_feedback = True
        try:
            self._set_scene_rect(new_scene)
        finally:
            self._suppress_handle_feedback = False
        self._emit_change()

    def itemChange(self, change, value):
        if change in (
            QGraphicsItem.GraphicsItemChange.ItemPositionChange,
            QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged,
            QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged,
            QGraphicsItem.GraphicsItemChange.ItemTransformHasChanged
        ):
            self._suppress_handle_feedback = True
            try:
                self.update_handles()
                self._update_label_position()
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
        r = self.rect()
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
        self._class_color_cache: Dict[int, QColor] = {}
        self._temp_rect = None

        self.act_delete = QAction("Delete selected", self)
        self.act_delete.triggered.connect(self._emit_delete)
        self.act_to_cls = QAction("Set to current class", self)
        self.act_to_cls.triggered.connect(self._emit_to_current)

    def set_status_sink(self, fn: Callable[[str], None]): self._on_status = fn
    def set_current_class(self, cid: int, names: List[str]): self._current_class = int(cid); self._class_names = names or []
    def _class_name_for(self, cls: int) -> str:
        if 0 <= cls < len(self._class_names):
            return self._class_names[cls]
        return str(cls)

    def _color_for_class(self, cls: int) -> QColor:
        color = self._class_color_cache.get(cls)
        if color is None:
            hue = (hash((cls, 'yolo-editor')) % 360)
            color = QColor.fromHsv(hue, 200, 255, 220)
            self._class_color_cache[cls] = color
        return color

    def show_image_bgr(self, path: Path, bgr_img):
        self.resetTransform()
        self.scene.clear()
        self._temp_rect = None
        self.img_item = None
        qimg = qimage_from_cv_bgr(bgr_img)
        self.img_w, self.img_h = qimg.width(), qimg.height()
        pix = QPixmap.fromImage(qimg)
        self.img_item = self.scene.addPixmap(pix)
        # Ensure image is positioned at (0,0) in the scene
        self.img_item.setPos(0, 0)
        self.image_path = Path(path)
        self.fitInView(self.img_item, Qt.AspectRatioMode.KeepAspectRatio)
        
        # Update scene rect to match the image
        self.scene.setSceneRect(0, 0, self.img_w, self.img_h)

    def add_box_norm(self, box, color=None):
        if isinstance(box, Box):
            vb = box
        else:
            cls = int(getattr(box, 'cls', 0))
            cx = float(getattr(box, 'cx', 0.5))
            cy = float(getattr(box, 'cy', 0.5))
            w = float(getattr(box, 'w', 0.1))
            h = float(getattr(box, 'h', 0.1))
            vb = Box(cls=cls, cx=cx, cy=cy, w=w, h=h)
        rect = vb.to_rect(self.img_w, self.img_h)
        name = self._class_name_for(vb.cls)
        pen_color = color or self._color_for_class(vb.cls)
        item = BBoxItem(rect, cls=vb.cls, class_name=name, color=pen_color, on_change=self._notify_boxes_changed)
        self.scene.addItem(item)

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
            # Get the rectangle in scene coordinates
            r_scene = it.mapRectToScene(it.rect())
            # Since image is at (0,0), scene coordinates = image coordinates
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
                color = self._color_for_class(self._current_class)
                name = self._class_name_for(self._current_class)
                item = BBoxItem(rect, cls=self._current_class, class_name=name, color=color, on_change=self._notify_boxes_changed)
                self.scene.addItem(item)
                self._notify_boxes_changed()
                if self._on_status:
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
            name = self._class_name_for(self._current_class)
            color = self._color_for_class(self._current_class)
            it.set_class(self._current_class, name)
            it.set_color(color)
        self.contextToCurrentClass.emit()
        self._notify_boxes_changed()
        if self._on_status:
            name = self._class_names[self._current_class] if 0 <= self._current_class < len(self._class_names) else str(self._current_class)
            count = len(selected)
            word = 'box' if count == 1 else 'boxes'
            self._on_status(f"Set {count} {word} -> class {self._current_class} ({name})")
