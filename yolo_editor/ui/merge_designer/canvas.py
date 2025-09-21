from __future__ import annotations
from typing import Dict, Tuple, Optional
from PySide6.QtWidgets import QWidget, QVBoxLayout, QGraphicsView, QMenu, QAction
from PySide6.QtGui import QPainter, QMouseEvent, QWheelEvent
from PySide6.QtCore import Qt, QPointF

from .scene import MergeScene
from .node import NodeItem, ClassBlock, Port
from .edge import EdgeItem
from .controller import MergeController, SourceClass

class MergeCanvas(QWidget):
    """
    Zoomable, pannable canvas with dataset/target nodes and wireable ports.
    Drag: middle mouse or hold Space. Zoom: wheel.
    Connections update live stats via controller.
    """
    def __init__(self, controller: MergeController, parent=None):
        super().__init__(parent)
        self.ctrl = controller

        lay = QVBoxLayout(self)
        self.view = _GraphicsView()
        self.scene = MergeScene(self)
        self.view.setScene(self.scene)
        lay.addWidget(self.view)

        # bookkeeping
        self.nodes: Dict[str, NodeItem] = {}       # dataset nodes by dataset_id (== display name)
        self.target_nodes: Dict[int, NodeItem] = {}# target nodes by target_id
        self._temp_src: Optional[Port] = None      # dragging from port…

        # scene signals
        self.scene.sigs.edgeAdded.connect(lambda _: self._recalc_all_targets())
        self.scene.sigs.edgeRemoved.connect(lambda _: self._recalc_all_targets())

        # right-click to remove edge under cursor (optional)
        self.view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.view.customContextMenuRequested.connect(self._context_menu)

        # periodic advance to update edges’ paths as nodes move
        self.view.startTimer(30)

    # ---------- SPAWN ----------
    def spawn_dataset_node(self, dataset_id: str, classes: list[SourceClass], pos: QPointF = QPointF(40, 40)):
        node = NodeItem(title=f"Dataset: {dataset_id}", kind="dataset", x=pos.x(), y=pos.y())
        self.scene.addItem(node)
        self.nodes[dataset_id] = node
        # add class lines
        for sc in classes:
            sub = f"{sc.images} imgs, {sc.boxes} boxes"
            blk = node.add_class_block(text=f"{sc.class_name} ({sc.class_id})", subtext=sub, role="source",
                                       key=(dataset_id, sc.class_id))
        node.relayout()

    def spawn_target_node(self, target_id: int, name: str, quota: Optional[int], pos: QPointF = QPointF(600, 60)):
        node = NodeItem(title=f"Target Dataset", kind="target", x=pos.x(), y=pos.y())
        self.scene.addItem(node)
        self.target_nodes[target_id] = node

        # initial class block for this target
        sub = self._target_subtext(target_id)
        color = "#f7fff0"
        blk = node.add_class_block(text=f"{name} [{target_id}]", subtext=sub, role="target", key=target_id, color=color)

        # plus button to add more target classes directly under this node
        def add_more():
            # create a sibling target class under the SAME node
            new_tid = self.ctrl.add_target_class(name=f"{name}*", quota_images=quota)
            blk2 = node.add_class_block(
                text=f"{self.ctrl.model.targets[new_tid].class_name} [{new_tid}]",
                subtext=self._target_subtext(new_tid),
                role="target",
                key=new_tid,
                color=color
            )
            node.relayout()
        node.enable_plus(add_more)
        node.relayout()

    # ---------- WIRING ----------
    def mousePressOnPort(self, port: Port):
        if port.role == "source":
            self._temp_src = port

    def mouseReleaseOnPort(self, port: Port):
        if self._temp_src and port.role == "target":
            src_key = self._temp_src.key  # (dataset_id, class_id)
            tgt_id = port.key             # int
            # create or reuse edge
            self.ctrl.connect(src_key[0], src_key[1], tgt_id)
            edge = EdgeItem(self._temp_src, port)
            self.scene.addItem(edge)
            self.scene.sigs.edgeAdded.emit(edge)
        self._temp_src = None

    # ---------- REFRESH ----------
    def _recalc_all_targets(self):
        for tid, node in self.target_nodes.items():
            for blk in node.blocks:
                if blk.role == "target":
                    blk.set_subtext(self._target_subtext(blk.key))

    def _target_subtext(self, target_id: int) -> str:
        st = self.ctrl.target_stats(target_id)
        plan = self.ctrl.planned_allocation(target_id)
        if plan:
            parts = [f"{ds}:{cid}→{n}" for (ds, cid), n in plan.items()]
            return f"{st['images']} imgs / {st['boxes']} boxes | plan: {'; '.join(parts)}"
        return f"{st['images']} imgs / {st['boxes']} boxes"

    # ---------- CONTEXT ----------
    def _context_menu(self, pos):
        sp = self.view.mapToScene(pos)
        item = self.scene.itemAt(sp, self.view.transform())
        m = QMenu(self)
        if isinstance(item, EdgeItem):
            act = QAction("Delete connection", self)
            def _del():
                # find which mapping it is (best-effort)
                src = item.src_port
                dst = item.dst_port
                if isinstance(src, Port) and isinstance(dst, Port):
                    if src.role == "source" and dst.role == "target":
                        ds, cid = src.key
                        tid = dst.key
                        self.ctrl.disconnect(ds, cid, tid)
                self.scene.removeItem(item)
                self.scene.sigs.edgeRemoved.emit(item)
            act.triggered.connect(_del)
            m.addAction(act)
        m.exec(self.view.mapToGlobal(pos))

# --------- View with zoom/pan and port hit dispatch ----------
class _GraphicsView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setRenderHints(self.renderHints() | QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self._panning = False
        self._last = None
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.BoundingRectViewportUpdate)

    def wheelEvent(self, e: QWheelEvent):
        # smooth zoom
        factor = 1.15 if e.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.MiddleButton or (e.button() == Qt.MouseButton.LeftButton and (e.modifiers() & Qt.KeyboardModifier.SpaceModifier)):
            self._panning = True
            self._last = e.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            e.accept(); return

        # port press?
        item = self.itemAt(e.pos().toPoint())
        from .node import Port
        if isinstance(item, Port):
            w: MergeCanvas = self.parentWidget()
            w.mousePressOnPort(item)
            e.accept(); return

        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._panning and self._last is not None:
            delta = e.position() - self._last
            self._last = e.position()
            self.translate(delta.x(), delta.y())
            e.accept(); return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self._panning:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            e.accept(); return

        # port release?
        item = self.itemAt(e.pos().toPoint())
        from .node import Port
        if isinstance(item, Port):
            w: MergeCanvas = self.parentWidget()
            w.mouseReleaseOnPort(item)
            e.accept(); return

        super().mouseReleaseEvent(e)

    def timerEvent(self, e):
        # drive EdgeItem.advance to keep curves updated while dragging nodes
        if self.scene():
            for it in self.scene().items():
                if hasattr(it, "advance"):
                    it.advance(0)
        super().timerEvent(e)
