from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

from .merge_model import MergePlan, BalanceMode, EdgeKey
from .yolo_io import parse_label_file
from .utils.hashing import stable_int_key

ImgKey = Tuple[str, Path]  # (dataset_id, image_path)

@dataclass
class EdgeGroup:
    edge: EdgeKey
    target: int
    images: List[ImgKey]  # unique images that include this source class (after remap)

@dataclass
class SelectionResult:
    selected_images: Set[ImgKey]
    by_target: Dict[int, Set[ImgKey]]
    by_edge: Dict[EdgeKey, Set[ImgKey]]
    preview_supply: Dict[int, Dict[str, int]]
    preview_edges: Dict[int, List[Tuple[EdgeKey,int,int]]]
    warnings: List[str]

def build_edge_index(plan: MergePlan, sources) -> Dict[int, List[EdgeGroup]]:
    per_edge_imgs: Dict[Tuple[str,int,int], Set[ImgKey]] = defaultdict(set)
    for ds in sources:
        repo = ds.repo
        for split, imgs in repo.splits_map.items():
            for img in imgs:
                rows = parse_label_file(repo.label_path_for(img))
                for (src_cls, x, y, w, h) in rows:
                    tgt = plan.mapping.get((ds.id, src_cls), None)
                    if tgt is None: 
                        continue
                    per_edge_imgs[(ds.id, src_cls, tgt)].add((ds.id, img))
    per_target: Dict[int, List[EdgeGroup]] = defaultdict(list)
    for (dsid, src_cls, tgt), imgs in per_edge_imgs.items():
        per_target[tgt].append(EdgeGroup(edge=(dsid, src_cls), target=tgt, images=list(imgs)))
    return per_target

def select_with_quotas(plan: MergePlan, per_target: Dict[int, List[EdgeGroup]]) -> SelectionResult:
    selected: Set[ImgKey] = set()
    by_target: Dict[int, Set[ImgKey]] = defaultdict(set)
    by_edge: Dict[EdgeKey, Set[ImgKey]] = defaultdict(set)
    preview_supply: Dict[int, Dict[str, int]] = {}
    preview_edges: Dict[int, List[Tuple[EdgeKey,int,int]]] = defaultdict(list)
    warnings: List[str] = []

    for tgt, groups in per_target.items():
        supplies: List[Tuple[EdgeGroup, List[ImgKey]]] = []
        for g in groups:
            imgs = sorted(g.images, key=lambda im: stable_int_key(im[0]+"::"+im[1].as_posix(), seed=plan.random_seed))
            limit = plan.edge_limit.get(g.edge, None)
            if isinstance(limit, int) and limit >= 0:
                imgs = imgs[:limit]
            supplies.append((g, imgs))

        total_supply = sum(len(imgs) for _, imgs in supplies)
        quota = plan.target_quota.get(tgt, total_supply)
        preview_supply[tgt] = {"supply": total_supply, "quota": quota, "selected": 0}
        if total_supply == 0:
            warnings.append(f"Target {tgt}: no available images after mapping/limits.")
            continue

        K = len(supplies)
        want = [0]*K
        if plan.balance_mode == BalanceMode.EQUAL:
            base, rem = divmod(quota, K)
            for i in range(K): want[i] = base + (1 if i < rem else 0)
        else:
            tot = float(total_supply)
            for i, (_, imgs) in enumerate(supplies):
                want[i] = int(round(quota * len(imgs) / tot))

        short = 0
        for i, (_, imgs) in enumerate(supplies):
            if want[i] > len(imgs):
                short += want[i] - len(imgs)
                want[i] = len(imgs)
        if short > 0:
            spare = [(i, len(imgs) - want[i]) for i,(_,imgs) in enumerate(supplies)]
            for i,_ in sorted(spare, key=lambda t: t[1], reverse=True):
                if short == 0: break
                give = min(short, spare[i][1])
                want[i] += give
                short -= give

        taken_total = 0
        for i, (g, imgs) in enumerate(supplies):
            need = want[i]
            chosen: List[ImgKey] = []
            for im in imgs:
                if len(chosen) >= need: break
                if im in selected:
                    continue
                chosen.append(im)
            if len(chosen) < need:
                for im in imgs:
                    if len(chosen) >= need: break
                    if im in selected and im not in chosen:
                        chosen.append(im)

            for im in chosen:
                selected.add(im)
                by_target[tgt].add(im)
                by_edge[g.edge].add(im)
            taken = len(chosen)
            taken_total += taken
            preview_edges[tgt].append((g.edge, len(imgs), taken))

        preview_supply[tgt]["selected"] = taken_total
        if taken_total < quota:
            warnings.append(f"Target {tgt}: selected {taken_total}/{quota} (supply/limits).")

    return SelectionResult(
        selected_images=selected,
        by_target=by_target,
        by_edge=by_edge,
        preview_supply=preview_supply,
        preview_edges=preview_edges,
        warnings=warnings
    )
