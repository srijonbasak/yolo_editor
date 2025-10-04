from __future__ import annotations
from typing import Dict, List, Optional, Callable, Tuple, Any
from PySide6.QtWidgets import (
    QGraphicsItem, QGraphicsRectItem, QGraphicsTextItem, QGraphicsEllipseItem,
    QMenu, QStyleOptionGraphicsItem, QWidget
)
from PySide6.QtGui import QPen, QBrush, QColor, QPainter, QFont
from PySide6.QtCore import Qt, QRectF, QPointF, Signal, QObject


class Port(QGraphicsEllipseItem):
    """A connection port on a node."""
    
    def __init__(self, role: str, key: Any, parent=None):
        super().__init__(-8, -8, 16, 16, parent)
        self.role = role  # "source" or "target"
        self.key = key    # identifier for this port
        self.setBrush(QBrush(QColor("#4CAF50" if role == "source" else "#2196F3")))
        self.setPen(QPen(QColor("#333"), 2))
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
                 key: Any = None, color: str = "#f0f0f0", parent=None):
        super().__init__(parent)
        self.role = role
        self.key = key
        self._text = text
        self._subtext = subtext
        self._color = color
        self._on_double_click: Optional[Callable] = None
        self._context_menu_factory: Optional[Callable[[], QMenu]] = None
        
        # Create port
        self.port = Port(role, key, self)
        
        # Create text items
        self.text_item = QGraphicsTextItem(text, self)
        self.subtext_item = QGraphicsTextItem(subtext, self) if subtext else None
        
        # Style the block
        self.setBrush(QBrush(QColor(color)))
        self.setPen(QPen(QColor("#ccc"), 1))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        
        self._layout()
    
    def _layout(self):
        """Layout the text and port within the block."""
        font = QFont()
        font.setPointSize(10)
        self.text_item.setFont(font)
        
        if self.subtext_item:
            subfont = QFont()
            subfont.setPointSize(8)
            self.subtext_item.setFont(subfont)
            self.subtext_item.setDefaultTextColor(QColor("#666"))
        
        # Calculate size
        text_rect = self.text_item.boundingRect()
        subtext_rect = self.subtext_item.boundingRect() if self.subtext_item else QRectF()
        
        width = max(text_rect.width(), subtext_rect.width()) + 40
        height = text_rect.height() + (subtext_rect.height() if self.subtext_item else 0) + 20
        
        self.setRect(0, 0, width, height)
        
        # Position text
        self.text_item.setPos(20, 5)
        if self.subtext_item:
            self.subtext_item.setPos(20, text_rect.height() + 5)
        
        # Position port
        if self.role == "source":
            self.port.setPos(width - 8, height / 2)
        else:  # target
            self.port.setPos(8, height / 2)
    
    def set_title(self, text: str):
        """Update the main text."""
        self._text = text
        self.text_item.setPlainText(text)
        self._layout()
    
    def set_subtext(self, text: str):
        """Update the subtext."""
        self._subtext = text
        if not self.subtext_item:
            self.subtext_item = QGraphicsTextItem(text, self)
        else:
            self.subtext_item.setPlainText(text)
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


class NodeItem(QGraphicsRectItem):
    """A node containing multiple class blocks."""
    
    def __init__(self, title: str, kind: str = "dataset", x: float = 0, y: float = 0):
        super().__init__()
        self.title = title
        self.kind = kind  # "dataset" or "target"
        self.blocks: List[ClassBlock] = []
        self._plus_callback: Optional[Callable] = None
        
        # Create title text
        self.title_item = QGraphicsTextItem(title, self)
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        self.title_item.setFont(font)
        
        # Style the node
        self.setBrush(QBrush(QColor("#ffffff")))
        self.setPen(QPen(QColor("#333"), 2))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        
        self.setPos(x, y)
        self.relayout()
    
    def add_class_block(self, text: str, subtext: str = "", role: str = "source",
                       key: Any = None, color: str = "#f0f0f0",
                       on_double_click: Optional[Callable] = None,
                       context_menu_factory: Optional[Callable[[], QMenu]] = None) -> ClassBlock:
        """Add a class block to this node."""
        block = ClassBlock(text, subtext, role, key, color, self)
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
    
    def relayout(self):
        """Relayout all blocks within the node."""
        if not self.blocks and not self.title_item:
            return
        
        # Calculate title size
        title_rect = self.title_item.boundingRect()
        
        # Position title
        self.title_item.setPos(10, 10)
        
        # Position blocks
        y_offset = title_rect.height() + 20
        max_width = title_rect.width() + 20
        
        for block in self.blocks:
            block.setPos(10, y_offset)
            block_rect = block.boundingRect()
            y_offset += block_rect.height() + 5
            max_width = max(max_width, block_rect.width() + 20)
        
        # Update node size
        total_height = y_offset + 10
        self.setRect(0, 0, max_width, total_height)
    
    def remove_block(self, block: ClassBlock):
        """Remove a block from this node."""
        if block in self.blocks:
            self.blocks.remove(block)
            if block.scene():
                block.scene().removeItem(block)
            self.relayout()