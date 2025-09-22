from __future__ import annotations
from typing import Dict, Optional, Tuple, List
from PySide6.QtWidgets import QWidget, QVBoxLayout, QGraphicsView, QMenu, QInputDialog
from PySide6.QtGui import QPainter, QWheelEvent, QAction
from PySide6.QtCore import Qt, QPointF

from .scene import MergeScene
from .node import NodeItem, Port, ClassBlock
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

        # Set up scene bounds for proper mouse interaction
        self.scene.setSceneRect(-1000, -1000, 2000, 2000)

        # bookkeeping
        self.nodes: Dict[str, NodeItem] = {}        # dataset nodes
        self.target_nodes: Dict[int, NodeItem] = {} # target nodes
        self.edge_items: Dict[Tuple[str, int], EdgeItem] = {} # (dataset_id, class_id) -> edge item
        self.dataset_placeholders: Dict[str, ClassBlock] = {}
        self._temp_src: Optional[Port] = None
        self._rubber_edge: Optional[EdgeItem] = None

        # right-click
        self.view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.view.customContextMenuRequested.connect(self._context_menu)

        # periodic advance to keep edges smooth while dragging
        self.view.startTimer(30)

    # ---------- SPAWN ----------
    def spawn_dataset_node(self, dataset_id: str, classes: list[SourceClass], loading: bool = False, pos: QPointF = QPointF(40, 60)):
        # force left column
        if dataset_id in self.nodes:
            return
        pos = QPointF(60, pos.y())
        node = NodeItem(title=f"Dataset: {dataset_id}", kind="dataset", x=pos.x(), y=pos.y())
        self.scene.addItem(node)
        self.nodes[dataset_id] = node

        if classes:
            for sc in classes:
                sub = f"{sc.images} imgs, {sc.boxes} boxes"
                node.add_class_block(
                    text=f"{sc.class_name} ({sc.class_id})",
                    subtext=sub,
                    role="source",
                    key=(dataset_id, sc.class_id),
                )
        elif loading:
            placeholder = node.add_class_block(
                text="Loading classes...",
                subtext="Scanning dataset...",
                role="source",
                key=(dataset_id, None),
                color="#f0f0f0",
            )
            self.dataset_placeholders[dataset_id] = placeholder

        node.relayout()

    def spawn_target_node(self, target_id: int, name: str, quota: Optional[int], pos: QPointF = QPointF(900, 60)):
        # force right column
        pos = QPointF(900, pos.y())
        node = NodeItem(title="Target Dataset", kind="target", x=pos.x(), y=pos.y())
        self.scene.addItem(node)
        self.target_nodes[target_id] = node
        self._add_target_block(node, target_id)
        node.enable_plus(lambda n=node: self._on_plus_new_target(n))
        node.relayout()

    def update_dataset_stats(self, dataset_id: str, classes: list[SourceClass]):
        node = self.nodes.get(dataset_id)
        if not node:
            return

        if not classes:
            placeholder = self.dataset_placeholders.get(dataset_id)
            if not placeholder:
                placeholder = node.add_class_block(
                    text="Loading classes...",
                    subtext="Scanning dataset...",
                    role="source",
                    key=(dataset_id, None),
                    color="#f0f0f0",
                )
                self.dataset_placeholders[dataset_id] = placeholder
            else:
                placeholder.set_title("Loading classes...")
                placeholder.set_subtext("Scanning dataset...")
            node.relayout()
            return

        placeholder = self.dataset_placeholders.pop(dataset_id, None)
        if placeholder and placeholder in node.blocks:
            node.blocks.remove(placeholder)
            if placeholder.scene():
                placeholder.scene().removeItem(placeholder)

        existing: Dict[int, ClassBlock] = {}
        for blk in node.blocks:
            if blk.role == "source" and isinstance(blk.key, tuple) and blk.key[0] == dataset_id and blk.key[1] is not None:
                existing[int(blk.key[1])] = blk

        seen: set[int] = set()
        for sc in classes:
            text = f"{sc.class_name} ({sc.class_id})"
            sub = f"{sc.images} imgs, {sc.boxes} boxes"
            blk = existing.get(sc.class_id)
            if blk:
                blk.set_title(text)
                blk.set_subtext(sub)
            else:
                node.add_class_block(text=text, subtext=sub, role="source", key=(dataset_id, sc.class_id))
            seen.add(sc.class_id)

        for cid, blk in list(existing.items()):
            if cid not in seen and blk in node.blocks:
                node.blocks.remove(blk)
                if blk.scene():
                    blk.scene().removeItem(blk)

        node.relayout()
        for edge in list(self.edge_items.values()):
            if edge.edge_key and edge.edge_key[0] == dataset_id:
                self._update_edge_tooltip(edge)
        self._recalc_all_targets()

    def set_dataset_error(self, dataset_id: str, message: str):
        node = self.nodes.get(dataset_id)
        if not node:
            return
        msg = (message.splitlines()[0] if message else "Error")
        if len(msg) > 80:
            msg = msg[:77] + "..."
        placeholder = self.dataset_placeholders.get(dataset_id)
        if not placeholder:
            placeholder = node.add_class_block(
                text="Failed to load",
                subtext=msg,
                role="source",
                key=(dataset_id, None),
                color="#ffeaea",
            )
            self.dataset_placeholders[dataset_id] = placeholder
        else:
            placeholder.set_title("Failed to load")
            placeholder.set_subtext(msg)
        node.relayout()
    def _target_block_title(self, target_id: int) -> str:
        tgt = self.ctrl.get_target(target_id)
        name = tgt.class_name if tgt else f"target_{target_id}"
        return f"{name} [{target_id}]"

    def _add_target_block(self, node: NodeItem, target_id: int):
        title = self._target_block_title(target_id)
        sub = self._target_subtext(target_id)
        def make_menu(tid=target_id):
            return self._build_target_block_menu(tid)
        blk = node.add_class_block(
            text=title,
            subtext=sub,
            role="target",
            key=target_id,
            color="#f7fff0",
            on_double_click=lambda tid=target_id: self._edit_target(tid),
            context_menu_factory=make_menu
        )
        return blk

    def _on_plus_new_target(self, node: NodeItem):
        details = self._prompt_target_details(title="New target class", default_name="")
        if not details:
            return
        name, quota = details
        new_tid = self.ctrl.add_target_class(name=name, quota_images=quota)
        self.target_nodes[new_tid] = node
        self._add_target_block(node, new_tid)
        node.relayout()
        self._recalc_all_targets()

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
                changed = self.ctrl.connect(src_key[0], src_key[1], tgt_id)
                if changed:
                    prev = self.edge_items.pop(src_key, None)
                    if prev and prev is not self._rubber_edge:
                        self._delete_edge_item(prev, disconnect=False)
                    if self._rubber_edge:
                        self._rubber_edge.attach_dst(port_or_none)
                        self._register_edge(self._rubber_edge, src_key, tgt_id)
                    made = True
                else:
                    if self._rubber_edge is not None:
                        self.scene.removeItem(self._rubber_edge)
            if not made and self._rubber_edge is not None:
                # drop rubber edge
                self.scene.removeItem(self._rubber_edge)
            self._rubber_edge = None
            self._temp_src = None
            self._recalc_all_targets()

    def _register_edge(self, edge: EdgeItem, src_key: Tuple[str, int], target_id: int):
        edge.assign_metadata(src_key, target_id)
        self.edge_items[src_key] = edge
        self._update_edge_tooltip(edge)

    def _update_edge_tooltip(self, edge: EdgeItem):
        if not edge.edge_key:
            edge.setToolTip("")
            return
        ds, cid = edge.edge_key
        src = self.ctrl.get_source_class(ds, cid)
        tgt = self.ctrl.get_target(edge.target_id) if edge.target_id is not None else None
        src_name = src.class_name if src else str(cid)
        tgt_name = tgt.class_name if tgt else str(edge.target_id)
        limit = self.ctrl.get_edge_limit(ds, cid)
        parts = [f"{ds}:{cid} ({src_name}) -> {tgt_name}"]
        if limit:
            parts.append(f"limit {limit}")
        edge.setToolTip(" | ".join(parts))

    # ---------- REFRESH ----------
    def _recalc_all_targets(self):
        for tid, node in self.target_nodes.items():
            for blk in node.blocks:
                if blk.role == "target" and blk.key == tid:
                    blk.set_title(self._target_block_title(tid))
                    blk.set_subtext(self._target_subtext(tid))
            node.relayout()
        for edge in self.edge_items.values():
            self._update_edge_tooltip(edge)

    def _target_subtext(self, target_id: int) -> str:
        st = self.ctrl.target_stats(target_id)
        plan = self.ctrl.planned_allocation(target_id)
        quota = self.ctrl.get_target_quota(target_id)
        info = f"{st['images']} imgs / {st['boxes']} boxes | quota {quota if quota else 'inf'}"
        if plan:
            parts = [f"{ds}:{cid}->{n}" for (ds, cid), n in plan.items()]
            info += f" | plan: {'; '.join(parts)}"
        return info

    # ---------- DELETE ----------
    def _delete_edge_item(self, edge_item: EdgeItem, disconnect: bool = True):
        src = getattr(edge_item, "src_item", None)
        dst = getattr(edge_item, "dst_item", None)
        if disconnect and isinstance(src, Port) and isinstance(dst, Port) and dst.role == "target" and src.role == "source":
            ds, cid = src.key
            self.ctrl.disconnect(ds, cid, dst.key)
        if edge_item.edge_key:
            self.edge_items.pop(edge_item.edge_key, None)
        if edge_item.scene():
            edge_item.scene().removeItem(edge_item)

    def delete_selection(self):
        # delete selected edges or nodes; disconnect controller accordingly
        for it in list(self.scene.selectedItems()):
            if isinstance(it, EdgeItem):
                self._delete_edge_item(it)
            elif isinstance(it, NodeItem):
                if it.kind == "dataset":
                    dataset_id = None
                    for k, v in list(self.nodes.items()):
                        if v is it:
                            dataset_id = k
                            self.nodes.pop(k)
                            break
                    ports: List[Port] = [blk.port for blk in it.blocks]
                    for edge in list(self.edge_items.values()):
                        if getattr(edge, "src_item", None) in ports or getattr(edge, "dst_item", None) in ports:
                            self._delete_edge_item(edge)
                    if dataset_id:
                        self.dataset_placeholders.pop(dataset_id, None)
                        self.ctrl.remove_dataset(dataset_id)
                        # remove cached mapping entries for dataset
                        for key in list(self.edge_items.keys()):
                            if key[0] == dataset_id:
                                edge = self.edge_items.pop(key)
                                if edge.scene():
                                    edge.scene().removeItem(edge)
                    if it.scene():
                        it.scene().removeItem(it)
                elif it.kind == "target":
                    target_ids = [tid for tid, node in list(self.target_nodes.items()) if node is it]
                    for tid in target_ids:
                        self._remove_target(tid)
        self._recalc_all_targets()

    def _remove_target(self, target_id: int):
        node = self.target_nodes.pop(target_id, None)
        if not node:
            return
        # disconnect + remove edges pointing to this target
        for edge_key, edge in list(self.edge_items.items()):
            if edge.target_id == target_id:
                self._delete_edge_item(edge)
        self.ctrl.remove_target_class(target_id)
        removed_block = None
        for blk in list(node.blocks):
            if blk.role == "target" and blk.key == target_id:
                removed_block = blk
                node.blocks.remove(blk)
                if blk.scene():
                    blk.scene().removeItem(blk)
        node.relayout()
        if not any(blk.role == "target" for blk in node.blocks):
            if node.scene():
                node.scene().removeItem(node)
            for tid in list(self.target_nodes.keys()):
                if self.target_nodes.get(tid) is node:
                    self.target_nodes.pop(tid, None)
        self._recalc_all_targets()

    # ---------- CONTEXT ----------
    def _context_menu(self, pos):
        sp = self.view.mapToScene(pos)
        item = self.scene.itemAt(sp, self.view.transform())
        m = QMenu(self)
        if isinstance(item, EdgeItem):
            if item.edge_key:
                act_limit = QAction("Set limit...", self)
                act_limit.triggered.connect(lambda checked=False, edge=item: self._prompt_edge_limit(edge))
                m.addAction(act_limit)
                if self.ctrl.get_edge_limit(*item.edge_key):
                    act_clear = QAction("Clear limit", self)
                    act_clear.triggered.connect(lambda checked=False, edge=item: self._clear_edge_limit(edge))
                    m.addAction(act_clear)
            act = QAction("Delete connection", self)
            act.triggered.connect(lambda checked=False, edge=item: (edge.setSelected(True), self._delete_edge_item(edge)))
            m.addAction(act)
        if isinstance(item, NodeItem):
            actn = QAction("Remove node", self)
            actn.triggered.connect(lambda checked=False, node=item: (node.setSelected(True), self.delete_selection()))
            m.addAction(actn)
        if m.actions():
            m.exec(self.view.mapToGlobal(pos))

    def _build_target_block_menu(self, target_id: int) -> QMenu:
        menu = QMenu(self)
        act_edit = QAction("Edit target...", menu)
        act_edit.triggered.connect(lambda checked=False, tid=target_id: self._edit_target(tid))
        menu.addAction(act_edit)
        act_remove = QAction("Remove target", menu)
        act_remove.triggered.connect(lambda checked=False, tid=target_id: self._remove_target(tid))
        menu.addAction(act_remove)
        return menu

    def _prompt_edge_limit(self, edge: EdgeItem):
        if not edge.edge_key:
            return
        ds, cid = edge.edge_key
        current = self.ctrl.get_edge_limit(ds, cid) or 0
        limit, ok = QInputDialog.getInt(self, "Set edge limit", f"Max images for {ds}:{cid} (0 = unlimited):", current, 0, 10_000_000)
        if not ok:
            return
        if limit <= 0:
            self.ctrl.set_edge_limit(ds, cid, None)
        else:
            self.ctrl.set_edge_limit(ds, cid, limit)
        self._update_edge_tooltip(edge)

    def _clear_edge_limit(self, edge: EdgeItem):
        if not edge.edge_key:
            return
        ds, cid = edge.edge_key
        self.ctrl.set_edge_limit(ds, cid, None)
        self._update_edge_tooltip(edge)

    def _edit_target(self, target_id: int):
        tgt = self.ctrl.get_target(target_id)
        if not tgt:
            return
        details = self._prompt_target_details(title="Edit target", default_name=tgt.class_name,
                                              default_quota=tgt.quota_images or 0)
        if not details:
            return
        name, quota = details
        self.ctrl.rename_target_class(target_id, name)
        self.ctrl.set_target_quota(target_id, quota)
        self._recalc_all_targets()

    def _prompt_target_details(self, title: str, default_name: str, default_quota: Optional[int] = None) -> Optional[Tuple[str, Optional[int]]]:
        name, ok = QInputDialog.getText(self, title, "Target class name:", text=default_name)
        if not ok:
            return None
        name = name.strip()
        if not name:
            return None
        quota_val = default_quota or 0
        quota, ok = QInputDialog.getInt(self, title, "Images quota (0 = unlimited):", quota_val, 0, 10_000_000)
        if not ok:
            return None
        return name, (None if quota <= 0 else quota)


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
