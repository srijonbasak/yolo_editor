from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import numpy as np
import yaml

# Supported image extensions
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# ---------------- YOLO box ----------------

class Box:
    __slots__ = ("cls", "cx", "cy", "w", "h")
    def __init__(self, cls: int, cx: float, cy: float, w: float, h: float):
        self.cls = int(cls); self.cx = float(cx); self.cy = float(cy); self.w = float(w); self.h = float(h)

def read_yolo_txt(txt_path: Path) -> List[Box]:
    out: List[Box] = []
    if not txt_path.exists():
        return out
    try:
        for ln in txt_path.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            p = ln.split()
            if len(p) < 5:
                continue
            c = int(float(p[0])); cx, cy, w, h = map(float, p[1:5])
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


# ---------------- Images & helpers ----------------

def imread_unicode(path: Path):
    """cv2.imdecode + np.fromfile (Windows-unicode safe)."""
    import cv2
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def list_images(root: Path) -> List[Path]:
    out = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
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

def normalize_split(name: str) -> str:
    n = (name or "").lower()
    if n == "valid": return "val"
    if n == "eval":  return "val"   # treat eval like val in the editor
    return n
