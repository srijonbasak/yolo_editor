from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional
from .yolo_io import is_image, read_yaml
from .splitting import normalize_split

class DatasetRepository:
    """
    Supports your preferred layout and common YOLOv8 layouts.

    Preferred:
      <root>/<split>/{Image,Labels}/...

    Also supported:
      A) images/<split>/*  + labels/<split>/*
      B) <split>/images/*  + <split>/labels/*
      C) <split>/*.jpg     + <split>/labels/*.txt
      D) YAML-driven (train/val/valid/eval/test keys; '../' OK)
    """
    def __init__(self, root: Path, yaml_path: Optional[Path] = None):
        self.root = Path(root)
        self.yaml_path = Path(yaml_path) if yaml_path else self._find_yaml()
        self.names: List[str] = []
        self.nc: int = 0
        self.splits_map: Dict[str, List[Path]] = {"train": [], "val": [], "test": []}
        self._discover()

    # ----------------------- discovery -----------------------

    def _find_yaml(self) -> Optional[Path]:
        for name in ("data.yaml","data.yml","dataset.yaml","dataset.yml"):
            p = self.root / name
            if p.exists(): return p
        return None

    def _read_names(self, cfg: dict):
        names = cfg.get("names")
        if isinstance(names, dict):
            self.names = [v for k,v in sorted(names.items(), key=lambda kv: int(kv[0]))]
        elif isinstance(names, list):
            self.names = [str(n) for n in names]
        self.nc = cfg.get("nc", len(self.names) if self.names else 0)

    def _candidate_paths_from_yaml(self, cfg: dict) -> Dict[str, Path]:
        out = {}
        for key in ("train","val","valid","eval","test"):
            p = cfg.get(key)
            if not p: continue
            base = self.yaml_path.parent if self.yaml_path else self.root
            pp = (base / str(p)).resolve()
            out["val" if key in ("val","valid","eval") else key] = pp
        return out

    def _scan_images_under(self, root: Path) -> List[Path]:
        if not root.exists(): return []
        imgs = []
        for p in root.rglob("*"):
            if is_image(p): imgs.append(p)
        return sorted(imgs)

    def _discover(self):
        # 1) YAML-driven
        if self.yaml_path and self.yaml_path.exists():
            cfg = read_yaml(self.yaml_path)
            self._read_names(cfg)
            cand = self._candidate_paths_from_yaml(cfg)
            if cand:
                norm_map = {}
                for k, p in cand.items():
                    nk = "val" if k in ("val","valid","eval") else k
                    nk = normalize_split(nk)
                    norm_map[nk] = p
                for split, path in norm_map.items():
                    if path.name.lower() in ("image","images"):
                        self.splits_map[split] = self._scan_images_under(path)
                    else:
                        imgA = path / "Image"
                        imgB = path / "images"
                        if imgA.exists():
                            self.splits_map[split] = self._scan_images_under(imgA)
                        elif imgB.exists():
                            self.splits_map[split] = self._scan_images_under(imgB)
                        else:
                            self.splits_map[split] = self._scan_images_under(path)

        # 2) Preferred: <root>/<split>/{Image,Labels}
        if not any(self.splits_map.values()):
            for split_raw in ("train","val","test","valid","eval","evaluation"):
                split = normalize_split(split_raw)
                split_dir = self.root / split_raw
                if split_dir.exists():
                    imgA = split_dir / "Image"
                    if imgA.exists():
                        self.splits_map[split].extend(self._scan_images_under(imgA))

        # 3) A) images/<split> + labels/<split>
        if not any(self.splits_map.values()):
            images = self.root / "images"
            if images.exists():
                for split_raw in ("train","val","test","valid","eval"):
                    split = normalize_split("val" if split_raw in ("val","valid","eval") else split_raw)
                    self.splits_map[split].extend(self._scan_images_under(images / split_raw))

        # 4) B/C) <split>/images or <split>/*.jpg
        if not any(self.splits_map.values()):
            for split_raw in ("train","val","test","valid","eval"):
                split = normalize_split("val" if split_raw in ("val","valid","eval") else split_raw)
                split_dir = self.root / split_raw
                if split_dir.exists():
                    if (split_dir / "images").exists():
                        self.splits_map[split].extend(self._scan_images_under(split_dir / "images"))
                    elif (split_dir / "Image").exists():
                        self.splits_map[split].extend(self._scan_images_under(split_dir / "Image"))
                    else:
                        self.splits_map[split].extend([p for p in split_dir.iterdir() if p.is_file() and is_image(p)])

        # sort & dedup
        for k in list(self.splits_map.keys()):
            uniq, seen = [], set()
            for p in self.splits_map[k]:
                if p not in seen:
                    uniq.append(p); seen.add(p)
            self.splits_map[k] = uniq

    # ----------------------- API -----------------------

    def list_images(self) -> List[Path]:
        out = []
        for imgs in self.splits_map.values():
            out.extend(imgs)
        return out

    def label_path_for(self, image_path: Path) -> Path:
        """
        Map image path -> label path.
        Handles both 'Image'/'Labels' (preferred) and 'images'/'labels'.
        """
        parts = list(image_path.parts)
        for i in range(len(parts)-1, -1, -1):
            if parts[i] == "Image":
                return Path(*parts[:i], "Labels", *parts[i+1:]).with_suffix(".txt")
            if parts[i].lower() == "images":
                return Path(*parts[:i], "labels", *parts[i+1:]).with_suffix(".txt")

        if image_path.parent.name == "Image":
            return image_path.parent.parent / "Labels" / (image_path.stem + ".txt")
        if image_path.parent.name.lower() == "images":
            return image_path.parent.parent / "labels" / (image_path.stem + ".txt")

        cur = image_path.parent
        for _ in range(4):
            cand = cur.parent / "Labels" / (image_path.stem + ".txt")
            if cand.exists(): return cand
            cand2 = cur.parent / "labels" / (image_path.stem + ".txt")
            if cand2.exists(): return cand2
            cur = cur.parent

        return image_path.with_suffix(".txt")
