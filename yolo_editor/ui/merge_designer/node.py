from __future__ import annotations
from typing import Dict, List, Optional, Callable, Tuple, Any
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsRectItem,
    QGraphicsTextItem,
    QGraphicsEllipseItem,
    QMenu,
    QStyleOptionGraphicsItem,
    QWidget,
    QGraphicsDropShadowEffect,
)
from PySide6.QtGui import QPen, QBrush, QColor, QPainter, QFont
from PySide6.QtCore import Qt, QRectF, QPointF, Signal, QObject

SOURCE_BLOCK_COLOR = "#E1EFFE"
TARGET_BLOCK_COLOR = "#E7F9ED"
SOURCE_PORT_COLOR = "#2563EB"
TARGET_PORT_COLOR = "#059669"
TEXT_COLOR = "#1F2933"
SUBTEXT_COLOR = "#475569"
TITLE_COLOR = "#111827"
NODE_BACKGROUND = "#FFFFFF"
NODE_BORDER = "#CBD5E1"


class Port(QGraphicsEllipseItem):
    """A connection port on a node."""
    
    def __init__(self, role: str, key: Any, parent=None):
        super().__init__(-8, -8, 16, 16, parent)
        self.role = role  # "source" or "target"
        self.key = key    # identifier for this port
        color = SOURCE_PORT_COLOR if role == "source" else TARGET_PORT_COLOR
        self.setBrush(QBrush(QColor(color)))
        self.setPen(QPen(QColor("#1f2933"), 1.6))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setZValue(10)
    
    def mousePressEvent(self, event):
        # Let the canvas handle port interactions
        if hasattr(self.scene(), 'views') and self.scene().views():
            view = self.scene().views()[0]
            if hasattr(view, 'parent') and hasattr(view.parent(), 'mousePressOnPort'):
                view.parent().mousePressOnPort(self)
        super().mousePressEvent(event)


class ClassBlock(QGraphicsRectItem):
    """A class block within a node."""

    def __init__(self, text: str, subtext: str = "", role: str = "source",
                 key: Any = None, color: str = "#E2E8F0", parent=None):
        super().__init__(parent)
        self.role = role
        self.key = key
        self._text = text
        self._subtext = subtext
        self._color = color
        self._on_double_click: Optional[Callable] = None
        self._context_menu_factory: Optional[Callable[[], QMenu]] = None

        self.port = Port(role, key, self)

        self.text_item = QGraphicsTextItem(text, self)
        self.text_item.setDefaultTextColor(QColor(TEXT_COLOR))
        self.subtext_item = QGraphicsTextItem(subtext, self) if subtext else None
        if self.subtext_item:
            self.subtext_item.setDefaultTextColor(QColor(SUBTEXT_COLOR))

        self.setBrush(QBrush(QColor(color)))
        self.setPen(QPen(QColor("#cbd5e1"), 1))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)

        self._layout()

    def _layout(self):
        """Layout the text and port within the block."""
        font = QFont()
        font.setPointSize(11)
        font.setBold(True)
        self.text_item.setFont(font)
        self.text_item.setDefaultTextColor(QColor(TEXT_COLOR))

        if self.subtext_item:
            subfont = QFont()
            subfont.setPointSize(9)
            self.subtext_item.setFont(subfont)
            self.subtext_item.setDefaultTextColor(QColor(SUBTEXT_COLOR))

        text_rect = self.text_item.boundingRect()
        subtext_rect = self.subtext_item.boundingRect() if self.subtext_item else QRectF()

        width = max(text_rect.width(), subtext_rect.width()) + 48
        height = text_rect.height() + (subtext_rect.height() if self.subtext_item else 0) + 24

        self.setRect(0, 0, width, height)

        self.text_item.setPos(18, 6)
        if self.subtext_item:
            self.subtext_item.setPos(18, text_rect.height() + 8)

        if self.role == "source":
            self.port.setPos(width - 10, height / 2)
        else:
            self.port.setPos(10, height / 2)

    def set_title(self, text: str):
        """Update the main text."""
        self._text = text
        self.text_item.setPlainText(text)
        self.text_item.setDefaultTextColor(QColor(TEXT_COLOR))
        self._layout()

    def set_subtext(self, text: str):
        """Update the subtext."""
        self._subtext = text
        if not self.subtext_item:
            self.subtext_item = QGraphicsTextItem(text, self)
        else:
            self.subtext_item.setPlainText(text)
        self.subtext_item.setDefaultTextColor(QColor(SUBTEXT_COLOR))
        self._layout()

    def set_on_double_click(self, callback: Callable):
        """Set double-click callback."""
        self._on_double_click = callback

    def set_context_menu_factory(self, factory: Callable[[], QMenu]):
        """Set context menu factory."""
        self._context_menu_factory = factory

    def mouseDoubleClickEvent(self, event):
        if self._on_double_click:
            self._on_double_click()
        super().mouseDoubleClickEvent(event)

    def contextMenuEvent(self, event):
        if self._context_menu_factory:
            menu = self._context_menu_factory()
            if menu and menu.actions():
                menu.exec(event.screenPos())
        super().contextMenuEvent(event)




class _PlusButton(QGraphicsRectItem):
    """Small interactive plus button rendered inside target nodes."""

    def __init__(self, parent: QGraphicsItem, callback: Callable[[], None]):
        super().__init__(parent)
        self._callback = callback
        self.setRect(0, 0, 24, 24)
        self.setBrush(QBrush(QColor("#dcfce7")))
        self.setPen(QPen(QColor("#16a34a"), 1))
        self.setZValue(20)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        self._label = QGraphicsTextItem("+", self)
        font = QFont()
        font.setPointSize(14)
        font.setBold(True)
        self._label.setFont(font)
        self._label.setDefaultTextColor(QColor("#047857"))
        br = self._label.boundingRect()
        self._label.setPos((24 - br.width()) / 2, (24 - br.height()) / 2 - 2)

    def set_callback(self, callback: Callable[[], None]):
        self._callback = callback

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._callback:
            self._callback()
        super().mousePressEvent(event)

class NodeItem(QGraphicsRectItem):
    """A node containing multiple class blocks."""
    
    def __init__(self, title: str, kind: str = "dataset", x: float = 0, y: float = 0):
        super().__init__()
        self.title = title
        self.kind = kind  # "dataset" or "target"
        self.blocks: List[ClassBlock] = []
        self._plus_callback: Optional[Callable] = None
        self._plus_button: Optional[_PlusButton] = None
        
        # Create title text
        self.title_item = QGraphicsTextItem(title, self)
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        self.title_item.setFont(font)
        self.title_item.setDefaultTextColor(QColor(TITLE_COLOR))
        
        # Style the node
        self.setBrush(QBrush(QColor(NODE_BACKGROUND)))
        self.setPen(QPen(QColor(NODE_BORDER), 1.6))

        shadow = QGraphicsDropShadowEffect()
        shadow.setOffset(0, 6)
        shadow.setBlurRadius(22)
        shadow.setColor(QColor(0, 0, 0, 60))
        self.setGraphicsEffect(shadow)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        
        self.setPos(x, y)
        self.relayout()
    
    def add_class_block(self, text: str, subtext: str = "", role: str = "source",
                       key: Any = None, color: Optional[str] = None,
                       on_double_click: Optional[Callable] = None,
                       context_menu_factory: Optional[Callable[[], QMenu]] = None) -> ClassBlock:
        """Add a class block to this node."""
        palette_color = color
        if palette_color is None:
            if role == "source":
                palette_color = SOURCE_BLOCK_COLOR
            elif role == "target":
                palette_color = TARGET_BLOCK_COLOR
            else:
                palette_color = "#E2E8F0"
        block = ClassBlock(text, subtext, role, key, palette_color, self)
        if on_double_click:
            block.set_on_double_click(on_double_click)
        if context_menu_factory:
            block.set_context_menu_factory(context_menu_factory)
        
        self.blocks.append(block)
        self.relayout()
        return block
    
    def enable_plus(self, callback: Callable):
        """Enable plus button for adding new blocks."""
        self._plus_callback = callback
        if self._plus_button is None:
            self._plus_button = _PlusButton(self, self._invoke_plus)
        else:
            self._plus_button.set_callback(self._invoke_plus)
        self.relayout()

    def _invoke_plus(self):
        if self._plus_callback:
            self._plus_callback()

    def relayout(self):
        """Relayout all blocks within the node."""
        if not self.blocks and not self.title_item:
            return

        title_rect = self.title_item.boundingRect()
        self.title_item.setPos(16, 14)

        y_offset = title_rect.height() + 32
        max_width = max(title_rect.width() + 32, 160)

        for block in self.blocks:
            block.setPos(16, y_offset)
            block_rect = block.boundingRect()
            y_offset += block_rect.height() + 12
            max_width = max(max_width, block_rect.width() + 32)

        total_height = y_offset + 16
        self.setRect(0, 0, max_width, total_height)

        if self._plus_button:
            visible = self.kind == "target" and self._plus_callback is not None
            self._plus_button.setVisible(visible)
            if visible:
                rect = self._plus_button.rect()
                self._plus_button.setPos(self.rect().width() - rect.width() - 16, 16)


    def remove_block(self, block: ClassBlock):
        """Remove a block from this node."""
        if block in self.blocks:
            self.blocks.remove(block)
            if block.scene():
                block.scene().removeItem(block)
            self.relayout()
