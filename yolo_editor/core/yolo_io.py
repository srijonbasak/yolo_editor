from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import yaml

# Supported image extensions
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# ---------------- Basic helpers ----------------

def is_image(path: Path) -> bool:
    '''Return True if path has a supported image extension.'''
    return path.suffix.lower() in IMG_EXTS


# ---------------- YOLO box ----------------


class Box:
    __slots__ = ("cls", "cx", "cy", "w", "h")

    def __init__(self, cls: int, cx: float, cy: float, w: float, h: float):
        self.cls = int(cls)
        self.cx = float(cx)
        self.cy = float(cy)
        self.w = float(w)
        self.h = float(h)


def read_yolo_txt(txt_path: Path) -> List[Box]:
    out: List[Box] = []
    if not txt_path.exists():
        return out
    try:
        for ln in txt_path.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            if len(parts) < 5:
                continue
            c = int(float(parts[0]))
            cx, cy, w, h = map(float, parts[1:5])
            out.append(Box(c, cx, cy, w, h))
    except Exception:
        # stay resilient even if a label file is malformed
        pass
    return out


def write_yolo_txt(txt_path: Path, boxes: List[Box]) -> None:
    lines = [f"{b.cls} {b.cx:.6f} {b.cy:.6f} {b.w:.6f} {b.h:.6f}" for b in boxes]
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = txt_path.with_suffix(txt_path.suffix + ".tmp")
    tmp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    tmp.replace(txt_path)


def parse_label_file(txt_path: Path) -> List[Tuple[int, float, float, float, float]]:
    '''Return YOLO rows as tuples (cls, cx, cy, w, h).'''
    boxes = read_yolo_txt(txt_path)
    return [(b.cls, b.cx, b.cy, b.w, b.h) for b in boxes]


def save_label_file(txt_path: Path, rows: Iterable[Tuple[int, float, float, float, float]]) -> None:
    boxes = [Box(cls, cx, cy, w, h) for (cls, cx, cy, w, h) in rows]
    write_yolo_txt(txt_path, boxes)


# ---------------- Images & helpers ----------------


def imread_unicode(path: Path):
    '''cv2.imdecode + np.fromfile (Windows-unicode safe).'''
    import cv2

    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def list_images(root: Path) -> List[Path]:
    out = [p for p in root.rglob("*") if p.is_file() and is_image(p)]
    out.sort()
    return out


def labels_for_image(img_path: Path, labels_dir: Optional[Path]) -> Path:
    return (labels_dir / f"{img_path.stem}.txt") if labels_dir else img_path.with_suffix(".txt")


# ---------------- YAML ----------------


def load_yaml(path: Path) -> dict:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def read_yaml(path: Path) -> dict:
    '''Alias retained for legacy callers.'''
    return load_yaml(path)


def normalize_split(name: str) -> str:
    n = (name or "").lower()
    if n == "valid":
        return "val"
    if n == "eval":  # treat eval like val in the editor
        return "val"
    return n
