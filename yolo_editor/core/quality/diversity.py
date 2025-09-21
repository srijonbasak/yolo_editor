from __future__ import annotations
from typing import List, Sequence
from pathlib import Path

# Simple farthest-point sampling using filename hash distances.
def fps_select(paths: Sequence[Path], k: int) -> List[Path]:
    if k >= len(paths): return list(paths)
    feats = [hash(p.name) & 0xffffffff for p in paths]
    selected = []
    idx0 = max(range(len(feats)), key=lambda i: feats[i])
    selected.append(idx0)
    d = [abs(feats[i] - feats[idx0]) for i in range(len(feats))]
    for _ in range(1, k):
        j = max(range(len(feats)), key=lambda i: d[i])
        selected.append(j)
        for i in range(len(feats)):
            d[i] = min(d[i], abs(feats[i] - feats[j]))
    return [paths[i] for i in selected]
