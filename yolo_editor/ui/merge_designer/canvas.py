from __future__ import annotations
from typing import Dict, Optional, Tuple, List
from PySide6.QtWidgets import QWidget, QVBoxLayout, QGraphicsView, QMenu
from PySide6.QtGui import QPainter, QWheelEvent, QAction
from PySide6.QtCore import Qt, QPointF

from .scene import MergeScene
from .node import NodeItem, Port
from .edge import EdgeItem
from .controller import MergeController, SourceClass


class MergeCanvas(QWidget):
    """
    Zoomable, pannable canvas with dataset/target nodes and wireable ports.
    Keys:
      - Space + drag or Middle mouse: pan
      - Delete: remove selected nodes/edges
      - Wheel: zoom
    """
    def __init__(self, controller: MergeController, parent=None):
        super().__init__(parent)
        self.ctrl = controller

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.view = _GraphicsView(self)
        self.scene = MergeScene(self)
        self.view.setScene(self.scene)
        lay.addWidget(self.view)

        # bookkeeping
        self.nodes: Dict[str, NodeItem] = {}        # dataset nodes
        self.target_nodes: Dict[int, NodeItem] = {} # target nodes
        self._temp_src: Optional[Port] = None
        self._rubber_edge: Optional[EdgeItem] = None

        # right-click
        self.view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.view.customContextMenuRequested.connect(self._context_menu)

        # periodic advance to keep edges smooth while dragging
        self.view.startTimer(30)

    # ---------- SPAWN ----------
    def spawn_dataset_node(self, dataset_id: str, classes: list[SourceClass], pos: QPointF = QPointF(40, 60)):
        # force left column
        pos = QPointF(60, pos.y())
        node = NodeItem(title=f"Dataset: {dataset_id}", kind="dataset", x=pos.x(), y=pos.y())
        self.scene.addItem(node)
        self.nodes[dataset_id] = node
        for sc in classes:
            sub = f"{sc.images} imgs, {sc.boxes} boxes"
            node.add_class_block(text=f"{sc.class_name} ({sc.class_id})", subtext=sub, role="source",
                                 key=(dataset_id, sc.class_id))
        node.relayout()

    def spawn_target_node(self, target_id: int, name: str, quota: Optional[int], pos: QPointF = QPointF(900, 60)):
        # force right column
        pos = QPointF(900, pos.y())
        node = NodeItem(title=f"Target Dataset", kind="target", x=pos.x(), y=pos.y())
        self.scene.addItem(node)
        self.target_nodes[target_id] = node
        sub = self._target_subtext(target_id)
        blk = node.add_class_block(text=f"{name} [{target_id}]", subtext=sub, role="target", key=target_id, color="#f7fff0")

        def add_more():
            new_tid = self.ctrl.add_target_class(name=f"{name}*", quota_images=quota)
            blk2 = node.add_class_block(
                text=f"{self.ctrl.model.targets[new_tid].class_name} [{new_tid}]",
                subtext=self._target_subtext(new_tid),
                role="target",
                key=new_tid,
                color="#f7fff0"
            )
            node.relayout()
        node.enable_plus(add_more)
        node.relayout()

    # ---------- WIRING ----------
    def mousePressOnPort(self, port: Port):
        if port.role == "source":
            self._temp_src = port
            # start rubber-band edge
            self._rubber_edge = EdgeItem(src_item=port, dst_item=None)
            self.scene.addItem(self._rubber_edge)

    def mouseMovePos(self, scene_pos: QPointF):
        if self._rubber_edge is not None:
            self._rubber_edge.set_floating(scene_pos)

    def mouseReleaseOnPort(self, port_or_none):
        # If released on a valid target port, create edge + register
        if self._temp_src is not None:
            made = False
            if isinstance(port_or_none, Port) and port_or_none.role == "target":
                src_key = self._temp_src.key
                tgt_id = port_or_none.key
                self.ctrl.connect(src_key[0], src_key[1], tgt_id)
                if self._rubber_edge:
                    self._rubber_edge.attach_dst(port_or_none)
                made = True
            # if not made, drop rubber edge
            if not made and self._rubber_edge is not None:
                self.scene.removeItem(self._rubber_edge)
            self._rubber_edge = None
            self._temp_src = None
            self._recalc_all_targets()

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

    # ---------- DELETE ----------
    def delete_selection(self):
        # delete selected edges or nodes; disconnect controller accordingly
        for it in list(self.scene.selectedItems()):
            if isinstance(it, EdgeItem):
                # try to disconnect
                src = getattr(it, "src_item", None)
                dst = getattr(it, "dst_item", None)
                if isinstance(src, Port) and isinstance(dst, Port) and dst.role == "target" and src.role == "source":
                    ds, cid = src.key
                    self.ctrl.disconnect(ds, cid, dst.key)
                self.scene.removeItem(it)
            elif isinstance(it, NodeItem):
                # remove edges touching its ports
                ports: List[Port] = []
                for blk in it.blocks:
                    ports.append(blk.port)
                for obj in list(self.scene.items()):
                    if isinstance(obj, EdgeItem):
                        if getattr(obj, "src_item", None) in ports or getattr(obj, "dst_item", None) in ports:
                            # disconnect if it's a valid mapping
                            src = getattr(obj, "src_item", None)
                            dst = getattr(obj, "dst_item", None)
                            if isinstance(src, Port) and isinstance(dst, Port) and dst.role == "target" and src.role == "source":
                                ds, cid = src.key
                                self.ctrl.disconnect(ds, cid, dst.key)
                            self.scene.removeItem(obj)
                # remove node bookkeeping
                if it.kind == "dataset":
                    for k, v in list(self.nodes.items()):
                        if v is it:
                            self.nodes.pop(k)
                elif it.kind == "target":
                    for k, v in list(self.target_nodes.items()):
                        if v is it:
                            self.target_nodes.pop(k)
                self.scene.removeItem(it)
        self._recalc_all_targets()

    # ---------- CONTEXT ----------
    def _context_menu(self, pos):
        sp = self.view.mapToScene(pos)
        item = self.scene.itemAt(sp, self.view.transform())
        m = QMenu(self)
        if isinstance(item, EdgeItem):
            act = QAction("Delete connection", self)
            act.triggered.connect(lambda: (item.setSelected(True), self.delete_selection()))
            m.addAction(act)
        if isinstance(item, NodeItem):
            actn = QAction("Remove node", self)
            actn.triggered.connect(lambda: (item.setSelected(True), self.delete_selection()))
            m.addAction(actn)
        if m.actions():
            m.exec(self.view.mapToGlobal(pos))


# --------- View with zoom/pan and port hit dispatch ----------
class _GraphicsView(QGraphicsView):
    def __init__(self, canvas: MergeCanvas):
        super().__init__()
        self.canvas = canvas
        self.setRenderHints(self.renderHints() | QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self._panning = False
        self._last_pos = None
        self._space_down = False
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.BoundingRectViewportUpdate)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, e: QWheelEvent):
        factor = 1.15 if e.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Space:
            self._space_down = True
            e.accept(); return
        if e.key() == Qt.Key.Key_Delete:
            self.canvas.delete_selection()
            e.accept(); return
        super().keyPressEvent(e)

    def keyReleaseEvent(self, e):
        if e.key() == Qt.Key.Key_Space:
            self._space_down = False
            e.accept(); return
        super().keyReleaseEvent(e)

    def mousePressEvent(self, e):
        # pan
        if e.button() == Qt.MouseButton.MiddleButton or (e.button() == Qt.MouseButton.LeftButton and self._space_down):
            self._panning = True
            self._last_pos = e.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            e.accept(); return

        # port press?
        item = self.itemAt(e.pos())
        from .node import Port
        if isinstance(item, Port):
            self.canvas.mousePressOnPort(item)
            e.accept(); return

        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._panning and self._last_pos is not None:
            delta = e.position() - self._last_pos
            self._last_pos = e.position()
            self.translate(delta.x(), delta.y())
            e.accept(); return
        # update rubber edge
        self.canvas.mouseMovePos(self.mapToScene(e.pos()))
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self._panning:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            e.accept(); return

        item = self.itemAt(e.pos())
        from .node import Port
        target = item if isinstance(item, Port) else None
        self.canvas.mouseReleaseOnPort(target)
        super().mouseReleaseEvent(e)

    def timerEvent(self, e):
        # keep curves updated as nodes move
        if self.scene():
            for it in self.scene().items():
                if hasattr(it, "advance"):
                    it.advance(0)
        super().timerEvent(e)
