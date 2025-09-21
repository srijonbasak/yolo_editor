from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import os, yaml

Row = Tuple[int, float, float, float, float]  # cls, cx, cy, w, h
IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}

def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def parse_label_file(label_path: Path) -> List[Row]:
    if not label_path.exists(): return []
    out: List[Row] = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            parts = line.split()
            if len(parts) < 5: continue
            try:
                c = int(float(parts[0]))
                x,y,w,h = map(float, parts[1:5])
                out.append((c,x,y,w,h))
            except Exception:
                continue
    return out

def save_label_file(label_path: Path, rows: List[Row]):
    tmp = Path(str(label_path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for (c,x,y,w,h) in rows:
            f.write(f"{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
    os.replace(tmp, label_path)
