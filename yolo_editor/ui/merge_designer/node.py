# yolo_editor/ui/merge_designer/node.py
from __future__ import annotations
from typing import Callable, Optional, List
from PySide6.QtWidgets import (
    QGraphicsItem, QGraphicsRectItem, QGraphicsSimpleTextItem, QGraphicsEllipseItem,
    QGraphicsProxyWidget, QPushButton, QMenu
)
from PySide6.QtGui import QBrush, QPen, QColor, QCursor, QAction  # QAction is in QtGui on Qt6
from PySide6.QtCore import QRectF, QPointF, Qt

PORT_R = 6.0


class Port(QGraphicsEllipseItem):
    """
    Circular port used to wire connections.
    role: "source" or "target"
    """
    def __init__(self, parent: QGraphicsItem, role: str, key):
        super().__init__(-PORT_R, -PORT_R, 2 * PORT_R, 2 * PORT_R, parent)
        self.setBrush(QBrush(QColor("#ffffff")))
        self.setPen(QPen(QColor("#444"), 1.5))
        self.setZValue(2)
        self.role = role
        self.key = key  # (dataset_id, class_id) if source; int target_id if target
        self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsScenePositionChanges)


class ClassBlock(QGraphicsRectItem):
    """
    Visual block line: label + counts + a port (left for source, right for target).
    """
    def __init__(self, w: float, h: float, text: str, subtext: str = "",
                 role: str = "source", key=None, color: str = "#efefef"):
        super().__init__(0, 0, w, h)
        self.setBrush(QBrush(QColor(color)))
        self.setPen(QPen(QColor("#c9c9c9"), 1))
        self.title = QGraphicsSimpleTextItem(text, self)
        self.subtitle = QGraphicsSimpleTextItem(subtext, self)
        self.role = role
        self.key = key
        self._layout()

        # port
        self.port = Port(self, role=role, key=key)
        if role == "source":
            self.port.setPos(6, h / 2)
        else:
            self.port.setPos(w - 6, h / 2)

    def set_subtext(self, s: str):
        self.subtitle.setText(s)
        self._layout()

    def _layout(self):
        self.title.setPos(10, 6)
        self.subtitle.setPos(10, 6 + 16)


class NodeItem(QGraphicsRectItem):
    """
    A node with a header and repeated ClassBlocks.
    kind: "dataset" (left) or "target" (right)
    """
    def __init__(self, title: str, kind: str, x: float, y: float, width: float = 280.0):
        super().__init__(0, 0, width, 60)
        self.setPos(x, y)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setPen(QPen(QColor("#888"), 1.5))
        self.setBrush(QBrush(QColor("#fdfdfd")))
        self.kind = kind
        self.width = width

        self.header = QGraphicsRectItem(0, 0, width, 28, self)
        self.header.setBrush(QBrush(QColor("#5b9bd5")))
        self.header.setPen(QPen(QColor("#4a84b9"), 0))
        self.titleItem = QGraphicsSimpleTextItem(title, self.header)
        self.titleItem.setBrush(QBrush(Qt.white))
        self.titleItem.setPos(10, 6)

        self.blocks: List[ClassBlock] = []
        self._plus_proxy: Optional[QGraphicsProxyWidget] = None
        self.relayout()

    def add_class_block(self, text: str, subtext: str, role: str, key, color: str = "#efefef"):
        blk = ClassBlock(self.width - 8, 38, text=text, subtext=subtext, role=role, key=key, color=color)
        self.blocks.append(blk)
        self.relayout()
        return blk

    def enable_plus(self, on_click: Callable[[], None]):
        if self.kind != "target":
            return
        if self._plus_proxy is not None:
            self.scene().removeItem(self._plus_proxy)
            self._plus_proxy = None
        btn = QPushButton("+ Add target class")
        btn.clicked.connect(on_click)
        btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._plus_proxy = QGraphicsProxyWidget(self)
        self._plus_proxy.setWidget(btn)
        self.relayout()

    def relayout(self):
        # place blocks vertically under header
        y = 32
        for blk in self.blocks:
            blk.setParentItem(self)
            blk.setPos(4, y)
            y += blk.rect().height() + 6
        # resize node height
        h = max(60, y + 6)
        self.setRect(0, 0, self.width, h)
        self.header.setRect(0, 0, self.width, 28)
        if self._plus_proxy is not None:
            self._plus_proxy.setPos(6, h - 34)

    # Context menu hook (node removal is handled by canvas)
    def contextMenuEvent(self, event):
        m = QMenu()
        act_del = QAction("Remove node", m)
        m.addAction(act_del)
        chosen = m.exec(event.screenPos().toPoint())
        if chosen == act_del:
            # canvas will delete selection
            self.setSelected(True)
            v = self.scene().views()[0]
            if hasattr(v.parentWidget(), "delete_selection"):
                v.parentWidget().delete_selection()
