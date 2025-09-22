from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, Optional

from PySide6.QtCore import Qt, QObject, Signal, QThread
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QListWidget, QListWidgetItem,
    QFileDialog, QLabel, QGroupBox, QFormLayout, QSpinBox, QLineEdit, QTableWidget,
    QTableWidgetItem, QComboBox, QSplitter, QMessageBox, QCheckBox, QProgressDialog, QTabWidget
)

from ...core.multi_repo import MultiRepo
from ...core.merge_model import MergePlan, TargetClass, CopyMode, CollisionPolicy, SplitStrategy, BalanceMode
from ...core.merge_selector import build_edge_index, select_with_quotas, SelectionResult
from ...core.merger import merge_execute
from ...core.report import write_report
from ...core.quality.dups import phash_hex, too_similar
from ...core.quality.filters import load_bgr, blur_score, exposure_score, meets_min_resolution
from .preview_panel import PreviewPanel
from .canvas import MappingCanvas

EdgeKey = Tuple[str, int]  # (dataset_id, class_id)

class MergeWorker(QObject):
    progress = Signal(int, int)   # value, total
    finished = Signal(Path)       # output_dir
    failed = Signal(str)
    def __init__(self, plan: MergePlan, sources, selection):
        super().__init__()
        self.plan = plan
        self.sources = list(sources)
        self.selection = selection
    def _progress_cb(self, prog):
        self.progress.emit(prog.value, prog.total)
    def run(self):
        try:
            merge_execute(
                plan=self.plan,
                sources=self.sources,
                progress_cb=self._progress_cb,
                cancel=None,
                selection=self.selection
            )
            self.finished.emit(self.plan.output_dir)
        except Exception as e:
            self.failed.emit(str(e))

class MergeDesignerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Merge Datasets - Designer")
        self.resize(1400, 820)

        self.repo = MultiRepo()
        self.datasets_list = QListWidget()
        self.btn_add_root = QPushButton("Add Dataset Root...")
        self.btn_add_yaml = QPushButton("Add Dataset YAML...")

        self.tbl_targets = QTableWidget(0, 3)
        self.tbl_targets.setHorizontalHeaderLabels(["Index", "Name", "Quota (images)"])
        self.btn_add_target = QPushButton("Add Target Class")
        self.btn_del_target = QPushButton("Delete Selected Target")

        self.tbl_mapping = QTableWidget(0, 5)
        self.tbl_mapping.setHorizontalHeaderLabels(["Dataset", "Source ID", "Source Name", "Limit", "Target"])
        self.tbl_mapping.verticalHeader().setVisible(False)
        self.tbl_mapping.setSortingEnabled(False)

        self.ed_output = QLineEdit()
        self.btn_output = QPushButton("Select Output Dir...")
        self.chk_dedup = QCheckBox("Drop near-duplicates (pHash)")
        self.spn_dedup = QSpinBox(); self.spn_dedup.setRange(0, 32); self.spn_dedup.setValue(6)
        self.chk_quality = QCheckBox("Quality filter (min res + blur)")
        self.spn_minw = QSpinBox(); self.spn_minh = QSpinBox(); self.spn_blur = QSpinBox(); self.spn_expo = QSpinBox()
        self.spn_minw.setRange(1, 8192); self.spn_minh.setRange(1, 8192)
        self.spn_blur.setRange(0, 100000); self.spn_expo.setRange(0, 255)
        self.spn_minw.setValue(320); self.spn_minh.setValue(240)
        self.spn_blur.setValue(50); self.spn_expo.setValue(0)

        self.btn_preview = QPushButton("Preview")
        self.btn_merge = QPushButton("Merge...")

        tabs = QTabWidget()
        self.canvas = MappingCanvas()
        self.preview = PreviewPanel()
        tabs.addTab(self.canvas, "Canvas")
        tabs.addTab(self.preview, "Preview")

        left = QWidget(); left_l = QVBoxLayout(left)
        row = QHBoxLayout(); row.addWidget(self.btn_add_root); row.addWidget(self.btn_add_yaml)
        left_l.addLayout(row)
        left_l.addWidget(QLabel("Datasets"))
        left_l.addWidget(self.datasets_list, 1)

        grp_tgt = QGroupBox("Target Classes")
        fl = QFormLayout(grp_tgt)
        fl.addRow(self.tbl_targets)
        rowt = QHBoxLayout(); rowt.addWidget(self.btn_add_target); rowt.addWidget(self.btn_del_target)
        fl.addRow(rowt)
        left_l.addWidget(grp_tgt)

        grp_map = QGroupBox("Source -> Target Mapping & Limits")
        fl2 = QFormLayout(grp_map)
        fl2.addRow(self.tbl_mapping)
        left_l.addWidget(grp_map, 2)

        grp_opt = QGroupBox("Options")
        flo = QFormLayout(grp_opt)
        outrow = QHBoxLayout(); outrow.addWidget(self.ed_output, 1); outrow.addWidget(self.btn_output)
        flo.addRow("Output folder", outrow)
        drow = QHBoxLayout(); drow.addWidget(self.chk_dedup); drow.addWidget(QLabel("threshold")); drow.addWidget(self.spn_dedup)
        flo.addRow(drow)
        qrow = QHBoxLayout(); qrow.addWidget(self.chk_quality)
        qrow.addWidget(QLabel("minW")); qrow.addWidget(self.spn_minw)
        qrow.addWidget(QLabel("minH")); qrow.addWidget(self.spn_minh)
        qrow.addWidget(QLabel("minBlur")); qrow.addWidget(self.spn_blur)
        flo.addRow(qrow)
        left_l.addWidget(grp_opt)

        btnrow = QHBoxLayout(); btnrow.addStretch(1); btnrow.addWidget(self.btn_preview); btnrow.addWidget(self.btn_merge)
        left_l.addLayout(btnrow)

        split = QSplitter()
        split.addWidget(left)
        split.addWidget(tabs)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 3)

        root = QVBoxLayout(self); root.addWidget(split)

        self._ds_count = 0
        self._last_preview: Optional[SelectionResult] = None

        self.btn_add_root.clicked.connect(self._on_add_root)
        self.btn_add_yaml.clicked.connect(self._on_add_yaml)
        self.btn_add_target.clicked.connect(self._on_add_target)
        self.btn_del_target.clicked.connect(self._on_del_target)
        self.btn_output.clicked.connect(self._on_pick_output)
        self.btn_preview.clicked.connect(self._on_preview)
        self.btn_merge.clicked.connect(self._on_merge)

    # Datasets

    def _on_add_root(self):
        d = QFileDialog.getExistingDirectory(self, "Select dataset root (contains train/eval/test or images/labels)")
        if not d: return
        self._add_dataset(Path(d), None)

    def _on_add_yaml(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select data.yaml", filter="YAML files (*.yaml *.yml)")
        if not f: return
        yaml_path = Path(f)
        self._add_dataset(yaml_path.parent, yaml_path)

    def _add_dataset(self, root: Path, yaml_path: Optional[Path]):
        self._ds_count += 1
        ds_id = f"ds{self._ds_count}"
        self.repo.add(ds_id, root=root, yaml_path=yaml_path, display_name=root.name)
        item = QListWidgetItem(f"{ds_id}: {root}")
        self.datasets_list.addItem(item)
        self._rebuild_mapping_table()
        self._refresh_canvas()

    # Tables

    def _target_names_dict(self) -> Dict[int, str]:
        names = {}
        for r in range(self.tbl_targets.rowCount()):
            idx = int(self.tbl_targets.item(r, 0).text())
            nm = self.tbl_targets.item(r, 1).text()
            names[idx] = nm
        return names

    def _rebuild_mapping_table(self):
        rows = []
        for ds in self.repo:
            repo = ds.repo
            names = getattr(repo, "names", None) or [f"id_{i}" for i in range(getattr(repo, "nc", 0) or 0)]
            for cid, cname in enumerate(names):
                rows.append((ds.id, cid, str(cname)))
        rows.sort(key=lambda t: (t[0], t[1]))

        self.tbl_mapping.setRowCount(0)
        for (dsid, cid, cname) in rows:
            r = self.tbl_mapping.rowCount()
            self.tbl_mapping.insertRow(r)
            self.tbl_mapping.setItem(r, 0, QTableWidgetItem(dsid))
            self.tbl_mapping.setItem(r, 1, QTableWidgetItem(str(cid)))
            self.tbl_mapping.setItem(r, 2, QTableWidgetItem(cname))
            spn = QSpinBox(); spn.setRange(0, 10_000_000); spn.setValue(10_000_000)
            self.tbl_mapping.setCellWidget(r, 3, spn)
            cmb = QComboBox(); self._refresh_target_combo(cmb)
            self.tbl_mapping.setCellWidget(r, 4, cmb)

    def _refresh_target_combo(self, combo: QComboBox):
        combo.clear()
        targets = []
        for r in range(self.tbl_targets.rowCount()):
            idx = self.tbl_targets.item(r, 0)
            name = self.tbl_targets.item(r, 1)
            if idx and name:
                targets.append((int(idx.text()), name.text()))
        targets.sort(key=lambda t: t[0])
        for idx, name in targets:
            combo.addItem(f"{idx}: {name}", idx)

    def _on_add_target(self):
        r = self.tbl_targets.rowCount()
        self.tbl_targets.insertRow(r)
        self.tbl_targets.setItem(r, 0, QTableWidgetItem(str(r)))
        self.tbl_targets.setItem(r, 1, QTableWidgetItem(f"class_{r}"))
        spn = QSpinBox(); spn.setRange(0, 10_000_000); spn.setValue(0)
        self.tbl_targets.setCellWidget(r, 2, spn)
        for rr in range(self.tbl_mapping.rowCount()):
            cmb: QComboBox = self.tbl_mapping.cellWidget(rr, 4)
            self._refresh_target_combo(cmb)
        self._refresh_canvas()

    def _on_del_target(self):
        rows = sorted({i.row() for i in self.tbl_targets.selectedIndexes()}, reverse=True)
        if not rows: return
        for r in rows:
            self.tbl_targets.removeRow(r)
        for r in range(self.tbl_targets.rowCount()):
            self.tbl_targets.item(r, 0).setText(str(r))
        for rr in range(self.tbl_mapping.rowCount()):
            cmb: QComboBox = self.tbl_mapping.cellWidget(rr, 4)
            self._refresh_target_combo(cmb)
        self._refresh_canvas()

    # Build plan

    def _build_plan(self) -> Optional[MergePlan]:
        out = self.ed_output.text().strip()
        if not out:
            QMessageBox.warning(self, "Output folder missing", "Please select an output folder.")
            return None

        tclasses = []
        quotas = {}
        for r in range(self.tbl_targets.rowCount()):
            idx = int(self.tbl_targets.item(r, 0).text())
            name = self.tbl_targets.item(r, 1).text().strip() or f"class_{idx}"
            spn: QSpinBox = self.tbl_targets.cellWidget(r, 2)
            q = int(spn.value())
            tclasses.append(TargetClass(index=idx, name=name))
            if q > 0:
                quotas[idx] = q
        tclasses.sort(key=lambda tc: tc.index)

        mapping: Dict[EdgeKey, Optional[int]] = {}
        edge_limit: Dict[EdgeKey, int] = {}
        for r in range(self.tbl_mapping.rowCount()):
            dsid = self.tbl_mapping.item(r, 0).text()
            cid = int(self.tbl_mapping.item(r, 1).text())
            cmb: QComboBox = self.tbl_mapping.cellWidget(r, 4)
            tgt_idx = cmb.currentData()
            if tgt_idx is not None:
                mapping[(dsid, cid)] = int(tgt_idx)
            spn: QSpinBox = self.tbl_mapping.cellWidget(r, 3)
            edge_limit[(dsid, cid)] = int(spn.value())

        plan = MergePlan(
            name="merged",
            output_dir=Path(out),
            target_classes=tclasses,
            mapping=mapping,
            target_quota=quotas,
            edge_limit=edge_limit,
            balance_mode=BalanceMode.EQUAL,
            random_seed=1337,
            split_strategy=SplitStrategy.KEEP,
            copy_mode=CopyMode.HARDLINK,
            collision_policy=CollisionPolicy.RENAME,
            drop_empty_images=True,
            target_train_name="train",
            target_val_name="eval",
            target_test_name="test",
        )
        return plan

    # Filters & Preview

    def _apply_filters(self, per_target: dict[int, list], dedup_on: bool, dedup_thr: int,
                       qual_on: bool, min_w: int, min_h: int, min_blur: int, min_expo: int):
        phash_cache: Dict[Path, str] = {}
        qual_cache: Dict[Path, bool] = {}

        for tgt, groups in per_target.items():
            seen_hashes: list[str] = []
            for g in groups:
                kept = []
                for (dsid, img_path) in g.images:
                    if qual_on:
                        ok = qual_cache.get(img_path)
                        if ok is None:
                            bgr = load_bgr(img_path)
                            ok = meets_min_resolution(bgr, min_w, min_h) and (blur_score(bgr) >= min_blur) and (exposure_score(bgr) >= min_expo)
                            qual_cache[img_path] = ok
                        if not ok: continue
                    if dedup_on:
                        h = phash_cache.get(img_path)
                        if h is None:
                            try: h = phash_hex(img_path)
                            except Exception: h = ""
                            phash_cache[img_path] = h
                        if h:
                            dup = any(too_similar(hs, h, max_dist=dedup_thr) for hs in seen_hashes)
                            if dup: continue
                            seen_hashes.append(h)
                    kept.append((dsid, img_path))
                g.images = kept

    def _on_preview(self):
        if len(self.repo) == 0:
            QMessageBox.information(self, "No datasets", "Add at least one dataset to preview.")
            return
        plan = self._build_plan()
        if not plan: return

        per_target = build_edge_index(plan, self.repo)
        self._apply_filters(
            per_target,
            dedup_on=self.chk_dedup.isChecked(), dedup_thr=self.spn_dedup.value(),
            qual_on=self.chk_quality.isChecked(),
            min_w=self.spn_minw.value(), min_h=self.spn_minh.value(),
            min_blur=self.spn_blur.value(), min_expo=self.spn_expo.value()
        )

        sel = select_with_quotas(plan, per_target)
        self._last_preview = sel

        tnames = {tc.index: tc.name for tc in plan.target_classes}
        self.preview.set_target_names(tnames)
        self.preview.set_preview(sel.preview_supply, sel.preview_edges, sel.warnings)

        self._refresh_canvas()

    def _refresh_canvas(self):
        srcs = []
        current_dsid = None
        buf = None
        for i in range(self.tbl_mapping.rowCount()):
            dsid = self.tbl_mapping.item(i, 0).text()
            cid = int(self.tbl_mapping.item(i, 1).text())
            cname = self.tbl_mapping.item(i, 2).text()
            if dsid != current_dsid:
                if buf: srcs.append((current_dsid, buf))
                current_dsid = dsid
                buf = []
            buf.append((cid, cname))
        if buf: srcs.append((current_dsid, buf))

        tgts = []
        for r in range(self.tbl_targets.rowCount()):
            idx = int(self.tbl_targets.item(r, 0).text())
            name = self.tbl_targets.item(r, 1).text()
            tgts.append((idx, name))

        edges = []
        for i in range(self.tbl_mapping.rowCount()):
            dsid = self.tbl_mapping.item(i, 0).text()
            cid = int(self.tbl_mapping.item(i, 1).text())
            cmb: QComboBox = self.tbl_mapping.cellWidget(i, 4)
            tgt_idx = cmb.currentData()
            if tgt_idx is not None:
                edges.append(((dsid, cid), int(tgt_idx)))

        self.canvas.set_data(srcs, tgts, edges)

    # Merge

    def _on_pick_output(self):
        d = QFileDialog.getExistingDirectory(self, "Select output folder")
        if d:
            self.ed_output.setText(d)

    def _on_merge(self):
        if not self._last_preview:
            self._on_preview()
            if not self._last_preview:
                return

        plan = self._build_plan(); 
        if not plan: return

        output_dir = plan.output_dir
        if output_dir.exists() and any(output_dir.iterdir()):
            if QMessageBox.question(self, "Output not empty",
                                    "The output folder is not empty. Continue and write files?",
                                    QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
                return

        worker = MergeWorker(plan, self.repo, self._last_preview.selected_images)
        thread = QThread(self)
        worker.moveToThread(thread)

        progress = QProgressDialog("Merging...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setAutoClose(False)
        progress.setAutoReset(False)

        def on_prog(val, tot):
            progress.setMaximum(tot if tot > 0 else 100)
            progress.setValue(val)

        def on_done(_path: Path):
            progress.setValue(progress.maximum())
            write_report(plan.output_dir, plan, self._last_preview)
            QMessageBox.information(self, "Done", f"Merged dataset written to:\n{plan.output_dir}\n\nReport:\n{plan.output_dir / 'reports' / 'merge_report.json'}")
            thread.quit(); thread.wait()

        def on_fail(msg: str):
            progress.cancel()
            QMessageBox.critical(self, "Merge failed", msg)
            thread.quit(); thread.wait()

        worker.progress.connect(on_prog)
        worker.finished.connect(on_done)
        worker.failed.connect(on_fail)
        thread.started.connect(worker.run)
        progress.canceled.connect(thread.quit)

        thread.start()
        progress.exec()
