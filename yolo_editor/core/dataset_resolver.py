from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, List
from .yolo_io import list_images, load_yaml, normalize_split

class DatasetModel:
    """
    names: list[str]  (index = class id)
    splits: { split: {images_dir:Path, labels_dir:Path|None, images:list[Path]} }
    yaml_path: Path | None
    """
    def __init__(self):
        self.names: List[str] = []
        self.splits: Dict[str, Dict] = {}
        self.yaml_path: Optional[Path] = None

    def ordered_splits(self) -> List[str]:
        pref = [s for s in ("train", "val", "test") if s in self.splits]
        return pref + [s for s in self.splits.keys() if s not in ("train", "val", "test")]

    def has_any(self) -> bool:
        return any(v["images"] for v in self.splits.values())


def _guess_labels(images_dir: Path) -> Optional[Path]:
    # a) <split>/images -> <split>/labels
    if images_dir.name == "images":
        cand = images_dir.parent / "labels"
        if cand.exists(): return cand
    # b) <root>/images/<split> -> <root>/labels/<split>
    if images_dir.parent.name == "images":
        cand = images_dir.parent.parent / "labels" / images_dir.name
        if cand.exists(): return cand
    # c) <split>/ -> <split>/labels
    cand = images_dir / "labels"
    if cand.exists(): return cand
    return None


def _ensure_images_dir(p: Path) -> Optional[Path]:
    """If p is the split root, auto-try p/'images'. If p is an 'images' dir already, return it."""
    if not p.exists(): return None
    if p.name == "images": return p
    if (p / "images").exists(): return (p / "images")
    # if p contains images directly, accept p
    imgs = [q for q in p.iterdir() if q.is_file() and q.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}]
    if imgs: return p
    return None


def _resolve_from_yaml(yaml_path: Path) -> DatasetModel:
    y = load_yaml(yaml_path)
    dm = DatasetModel()
    dm.yaml_path = yaml_path

    # names may be list or dict
    names = y.get("names")
    if isinstance(names, list):
        dm.names = [str(n) for n in names]
    elif isinstance(names, dict):
        dm.names = [names[k] for k in sorted(names, key=lambda x: int(x))]

    base = yaml_path.parent

    for key in ("train", "val", "valid", "test", "eval"):
        if key not in y: continue
        split = normalize_split(key)
        raw = str(y[key]).strip().replace("\\", "/")
        raw_p = Path(raw)

        # absolute path?
        if raw_p.is_absolute():
            img_dir = _ensure_images_dir(raw_p)
        else:
            img_dir = _ensure_images_dir((base / raw_p).resolve())
            if img_dir is None:
                # Ultralytics style: '../train/images' can be off by one; strip ../ repeatedly
                raw2 = raw
                while raw2.startswith("../"):
                    raw2 = raw2[3:]
                img_dir = _ensure_images_dir((base / raw2).resolve())

        if img_dir:
            labels_dir = _guess_labels(img_dir)
            dm.splits[split] = {
                "images_dir": img_dir,
                "labels_dir": labels_dir,
                "images": list_images(img_dir),
            }

    return dm


def _resolve_from_root(root: Path) -> DatasetModel:
    dm = DatasetModel()

    yml = root / "data.yaml"
    if yml.exists():
        ydm = _resolve_from_yaml(yml)
        if ydm.splits: return ydm

    # Layout A: images/<split> (+ labels/<split>)
    img_root = root / "images"
    lab_root = root / "labels"
    if img_root.exists():
        for sp in ("train", "val", "test"):
            img_dir = img_root / sp
            if img_dir.exists():
                labels_dir = None
                cand = lab_root / sp
                if lab_root.exists() and cand.exists(): labels_dir = cand
                if labels_dir is None: labels_dir = _guess_labels(img_dir)
                dm.splits[sp] = {"images_dir": img_dir, "labels_dir": labels_dir, "images": list_images(img_dir)}

    # Layout B/C: <split>/images + <split>/labels  OR  <split> contains images directly
    for sp in ("train", "val", "test", "valid", "eval"):
        sp_dir = root / sp
        if not sp_dir.exists(): continue
        split = normalize_split(sp)
        img_dir = _ensure_images_dir(sp_dir)
        if img_dir:
            labels_dir = _guess_labels(img_dir)
            dm.splits[split] = {"images_dir": img_dir, "labels_dir": labels_dir, "images": list_images(img_dir)}

    # Optional classes.txt
    names_txt = root / "classes.txt"
    if names_txt.exists():
        dm.names = [ln.strip() for ln in names_txt.read_text(encoding="utf-8").splitlines() if ln.strip()]

    return dm


def resolve_dataset(path: Path) -> DatasetModel:
    """Accepts a dataset ROOT folder (train/val/test/etc) or a data.yaml file."""
    if path.is_file() and path.suffix.lower() in (".yaml", ".yml"):
        dm = _resolve_from_yaml(path)
    else:
        dm = _resolve_from_root(path)
    # keep only splits that actually have images
    dm.splits = {k: v for k, v in dm.splits.items() if v["images"]}
    return dm
