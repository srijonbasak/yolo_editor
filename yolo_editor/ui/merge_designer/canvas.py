from __future__ import annotations
from PySide6.QtWidgets import QWidget, QVBoxLayout, QGraphicsView, QGraphicsScene, QGraphicsSimpleTextItem, QGraphicsLineItem
from PySide6.QtGui import QPen, QColor, QPainter  # <-- added QPainter
from PySide6.QtCore import QPointF

class MappingCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        self.view = QGraphicsView()
        self.scene = QGraphicsScene(self)
        self.view.setScene(self.scene)
        # FIX: use QPainter flags
        self.view.setRenderHints(self.view.renderHints() | QPainter.Antialiasing)
        lay.addWidget(self.view)
        self.sources = []
        self.targets = []
        self.edges   = []


    def set_data(self, sources, targets, edges):
        self.sources = sources
        self.targets = targets
        self.edges = edges
        self._rebuild()

    def _rebuild(self):
        self.scene.clear()
        left_x, right_x = 50, 650
        y = 20
        src_pos = {}
        tgt_pos = {}

        yy = y
        for dsid, classes in self.sources:
            ds_item = QGraphicsSimpleTextItem(f"{dsid}")
            ds_item.setPos(left_x, yy); self.scene.addItem(ds_item)
            yy += 20
            for (cid, name) in classes:
                it = QGraphicsSimpleTextItem(f"{cid}: {name}")
                it.setPos(left_x+10, yy); self.scene.addItem(it)
                br = it.boundingRect()
                src_pos[(dsid,cid)] = it.mapToScene(br.right(), br.center().y())
                yy += 18
            yy += 12

        yy = y
        for (tid, name) in self.targets:
            it = QGraphicsSimpleTextItem(f"{tid}: {name}")
            it.setPos(right_x, yy); self.scene.addItem(it)
            br = it.boundingRect()
            tgt_pos[tid] = it.mapToScene(br.left(), br.center().y())
            yy += 22

        pen = QPen(QColor("#4488ff")); pen.setWidth(2)
        for ((dsid,cid), tid) in self.edges:
            if (dsid,cid) in src_pos and tid in tgt_pos:
                p1 = src_pos[(dsid,cid)]
                p2 = tgt_pos[tid]
                line = QGraphicsLineItem(p1.x(), p1.y(), p2.x(), p2.y())
                line.setPen(pen)
                self.scene.addItem(line)
