from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

@dataclass
class SourceClass:
    dataset_id: str      # human-readable dataset name or UUID
    class_id: int        # numeric ID in the source dataset
    class_name: str
    images: int          # total images containing this class
    boxes: int           # total boxes (optional; can be 0 if not tracked)

@dataclass
class TargetClass:
    class_id: int                 # index in target
    class_name: str
    quota_images: Optional[int] = None  # cap images to balance (None = unlimited)

@dataclass
class MappingEdge:
    source_key: Tuple[str, int]   # (dataset_id, class_id)
    target_id: int                # target class id

@dataclass
class MergeModel:
    # Sources: dataset_id -> list of its classes
    sources: Dict[str, List[SourceClass]] = field(default_factory=dict)
    # Target classes by id
    targets: Dict[int, TargetClass] = field(default_factory=dict)
    # Many-to-one edges
    edges: List[MappingEdge] = field(default_factory=list)

class MergeController:
    """
    Pure model/controller: no Qt imports here.
    Responsible for: target CRUD, wiring edges, live stats, quotas, and balanced allocations.
    """
    def __init__(self):
        self.model = MergeModel()
        self._next_target_id = 0

    # ---------- SOURCE MANAGEMENT ----------
    def upsert_dataset(self, dataset_id: str, classes: List[SourceClass]):
        self.model.sources[dataset_id] = classes

    def remove_dataset(self, dataset_id: str):
        if dataset_id in self.model.sources:
            self.model.sources.pop(dataset_id)
        # drop any edges that referenced it
        self.model.edges = [e for e in self.model.edges if e.source_key[0] != dataset_id]

    # ---------- TARGET MANAGEMENT ----------
    def add_target_class(self, name: str, quota_images: Optional[int] = None) -> int:
        tid = self._next_target_id
        self._next_target_id += 1
        self.model.targets[tid] = TargetClass(class_id=tid, class_name=name, quota_images=quota_images)
        return tid

    def rename_target_class(self, target_id: int, new_name: str):
        if target_id in self.model.targets:
            self.model.targets[target_id].class_name = new_name

    def set_target_quota(self, target_id: int, quota_images: Optional[int]):
        if target_id in self.model.targets:
            self.model.targets[target_id].quota_images = quota_images

    def remove_target_class(self, target_id: int):
        if target_id in self.model.targets:
            self.model.targets.pop(target_id)
        self.model.edges = [e for e in self.model.edges if e.target_id != target_id]

    # ---------- EDGE/WIRING ----------
    def connect(self, dataset_id: str, class_id: int, target_id: int):
        key = (dataset_id, class_id)
        if key not in [(e.source_key[0], e.source_key[1]) for e in self.model.edges if e.target_id == target_id]:
            self.model.edges.append(MappingEdge(source_key=key, target_id=target_id))

    def disconnect(self, dataset_id: str, class_id: int, target_id: int):
        self.model.edges = [e for e in self.model.edges
                            if not (e.source_key == (dataset_id, class_id) and e.target_id == target_id)]

    # ---------- LIVE STATS ----------
    def target_stats(self, target_id: int) -> Dict[str, int]:
        """Aggregate images/boxes flowing to one target."""
        images = 0
        boxes = 0
        for e in self.model.edges:
            if e.target_id != target_id: continue
            ds, cid = e.source_key
            src = self._find_source_class(ds, cid)
            if src:
                images += src.images
                boxes += src.boxes
        return {"images": images, "boxes": boxes}

    def planned_allocation(self, target_id: int) -> Dict[Tuple[str,int], int]:
        """
        If target has a quota_images, split intake across connected sources (by images) as evenly as possible.
        Returns per-source planned image counts.
        """
        tgt = self.model.targets.get(target_id)
        if not tgt:
            return {}
        # gather all sources wired to this target
        entries: List[Tuple[Tuple[str,int], int]] = []  # ((dataset_id, class_id), images)
        for e in self.model.edges:
            if e.target_id == target_id:
                src = self._find_source_class(e.source_key[0], e.source_key[1])
                if src and src.images > 0:
                    entries.append((e.source_key, src.images))
        if not entries:
            return {}

        total_available = sum(img for _, img in entries)
        if tgt.quota_images is None or tgt.quota_images >= total_available:
            # no cap: take all
            return {key: img for key, img in entries}

        # cap exists: distribute as evenly as possible by proportion, then round
        cap = tgt.quota_images
        # naive proportional rounding
        alloc = {key: int(round((img / total_available) * cap)) for key, img in entries}
        # fix rounding drift
        drift = cap - sum(alloc.values())
        # adjust biggest contributors (or smallest) to hit exact cap
        if drift != 0:
            # sort by remainder preference
            from math import modf
            order = sorted(entries, key=lambda kv: -(kv[1]/total_available)) if drift < 0 else \
                    sorted(entries, key=lambda kv: (kv[1]/total_available))
            i = 0
            while drift != 0 and i < len(order):
                key = order[i][0]
                alloc[key] += 1 if drift > 0 else -1
                drift += -1 if drift > 0 else 1
                i = (i + 1) % len(order)
        # ensure non-negative and not exceeding each source
        for key, img in entries:
            alloc[key] = max(0, min(alloc[key], img))
        return alloc

    # ---------- HELPERS ----------
    def _find_source_class(self, dataset_id: str, class_id: int) -> Optional[SourceClass]:
        for sc in self.model.sources.get(dataset_id, []):
            if sc.class_id == class_id:
                return sc
        return None
