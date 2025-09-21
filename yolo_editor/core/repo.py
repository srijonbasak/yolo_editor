from pathlib import Path
from typing import List, Dict, Optional, Tuple
import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def _is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS

def _gather_images_under(d: Path) -> List[Path]:
    if not d.exists():
        return []
    return sorted(p for p in d.rglob("*") if _is_image(p))

def _gather_images_here(d: Path) -> List[Path]:
    if not d.exists():
        return []
    return sorted(p for p in d.iterdir() if _is_image(p))

def _to_abs(base: Path, maybe_rel_or_abs: str) -> Path:
    p = Path(maybe_rel_or_abs)
    return (p if p.is_absolute() else (base / p)).resolve()

class DatasetRepository:
    """
    Flexible YOLO repo:
    - images/<split>/*   and   labels/<split>/*
    - <split>/images/*   and   <split>/labels/*
    - <split>/* (images directly under split) with <split>/labels/*
    - root IS a split (./images & ./labels)
    - YAML with train/val(/valid)/test (+ optional 'path')
    """
    def __init__(self, root: Path, yaml_path: Optional[Path] = None):
        self.root = Path(root).resolve()
        self.splits_map: Dict[str, List[Path]] = {}     # split -> image paths
        self.labels_map: Dict[Path, Path] = {}          # image -> label path
        self.class_names: List[str] = []
        self.num_classes: Optional[int] = None
        self.yaml_path: Optional[Path] = Path(yaml_path).resolve() if yaml_path else None
        self._last_debug: Dict[str, List[str]] = {}     # diagnostics
        self._load()

    # ---------- public API ----------
    def list_images(self) -> List[Path]:
        imgs = []
        for s in sorted(self.splits_map):
            imgs.extend(self.splits_map[s])
        return imgs

    def label_path_for(self, image_path: Path) -> Path:
        p = self.labels_map.get(image_path)
        if p is not None:
            return p
        parts = list(Path(image_path).parts)
        try:
            idx = parts.index("images")
            parts[idx] = "labels"
            return Path(*parts).with_suffix(".txt")
        except ValueError:
            if Path(image_path).parent.name == "images":
                return Path(image_path).parent.with_name("labels") / (Path(image_path).stem + ".txt")
            images_dir = self.root / "images"
            labels_dir = self.root / "labels"
            try:
                rel = Path(image_path).relative_to(images_dir)
                return labels_dir / rel.with_suffix(".txt")
            except Exception:
                # If root is a split dir, try sibling 'labels'
                split_dir = Path(image_path).parent
                return split_dir.with_name("labels") / (Path(image_path).stem + ".txt")

    def classes_file(self) -> Path:
        return self.root / "classes.txt"

    def debug_info(self) -> str:
        lines = [f"root: {self.root}"]
        if self.yaml_path:
            lines.append(f"yaml: {self.yaml_path}")
        for k, v in self._last_debug.items():
            lines.append(f"{k}:")
            for line in v:
                lines.append(f"  - {line}")
        for split, imgs in self.splits_map.items():
            lines.append(f"split '{split}': {len(imgs)} images")
        return "\n".join(lines)

    # ---------- internals ----------
    def _load(self):
        self.splits_map.clear()
        self.labels_map.clear()
        self._last_debug.clear()

        # 1) Use preselected YAML if provided or discover one (recursive)
        yaml_file = self.yaml_path or self._find_yaml()
        if yaml_file:
            self.yaml_path = yaml_file
            self._load_from_yaml(yaml_file)
        else:
            # 2a) root IS a split (./images, ./labels)
            if (self.root / "images").exists():
                self._load_as_single_split(self.root)

            # 2b) images/<split> and labels/<split>
            if not self.splits_map:
                self._load_images_labels_splits()

            # 2c) <split>/images and <split>/labels (train/images)
            if not self.splits_map:
                self._load_split_folders()

            # 2d) <split> contains images directly (train/*.jpg) + <split>/labels/*
            if not self.splits_map:
                self._load_split_direct_images()

        # class names fallback
        if not self.class_names:
            self._load_class_names_fallback()
        if self.num_classes is None and self.class_names:
            self.num_classes = len(self.class_names)

    def _find_yaml(self) -> Optional[Path]:
        # 1) common names in root
        candidates = ["data.yaml", "dataset.yaml", "coco128.yaml", "yolo.yaml"]
        for name in candidates:
            p = self.root / name
            if p.exists():
                return p
        # 2) any yaml/yml in root
        for p in self.root.iterdir():
            if p.suffix.lower() in {".yaml", ".yml"} and p.is_file():
                return p
        # 3) recursive search (in case YAML lives in a nested folder)
        for p in self.root.rglob("*.y*ml"):
            if p.is_file():
                return p
        return None

    # ---- YAML loader with smart fallbacks ----
    def _scan_images_with_fallbacks(self, split: str, primary: Path, entry_text: str, yaml_parent: Path) -> Tuple[List[Path], Path]:
        """
        Return (images, chosen_dir). Try primary, then progressively smarter fallbacks
        when the YAML entry like '../train/images' is off by one level.
        """
        tried = []

        def scan_dir(d: Path) -> List[Path]:
            # check 'images' child if present, else scan d; prefer recursive
            if d.name == "images" or (d / "images").exists():
                base = d if d.name == "images" else (d / "images")
                imgs = _gather_images_under(base)
                tried.append(f"{split}: scanned {base} -> {len(imgs)}")
                return imgs
            deep = _gather_images_under(d)
            tried.append(f"{split}: scanned {d} -> {len(deep)}")
            if deep:
                return deep
            here = _gather_images_here(d)
            tried.append(f"{split}: scanned(here) {d} -> {len(here)}")
            return here

        # 1) primary from YAML (our normal resolution)
        imgs = scan_dir(primary)
        if imgs:
            self._last_debug.setdefault("scanned", []).extend(tried)
            return imgs, (primary / "images" if (primary / "images").exists() and primary.name != "images" else primary)

        # 2) If the entry was relative, try relative to the YAML's *own* folder (ignore 'path')
        entry_rel = entry_text.replace("\\", "/")
        if not Path(entry_text).is_absolute():
            alt1 = (yaml_parent / entry_text).resolve()
            imgs = scan_dir(alt1)
            if imgs:
                self._last_debug.setdefault("scanned", []).extend(tried)
                return imgs, (alt1 / "images" if (alt1 / "images").exists() and alt1.name != "images" else alt1)

            # 3) If it starts with ../, also try stripping one or more ../ (common Roboflow quirk)
            while entry_rel.startswith("../"):
                entry_rel = entry_rel[3:]
                alt2 = (yaml_parent / entry_rel).resolve()
                imgs = scan_dir(alt2)
                if imgs:
                    self._last_debug.setdefault("scanned", []).extend(tried)
                    return imgs, (alt2 / "images" if (alt2 / "images").exists() and alt2.name != "images" else alt2)

        # 4) Last resort: try canonical patterns right next to the YAML
        for candidate in [yaml_parent / "train/images",
                          yaml_parent / "val/images",
                          yaml_parent / "valid/images",
                          yaml_parent / "test/images"]:
            imgs = scan_dir(candidate)
            if imgs:
                self._last_debug.setdefault("scanned", []).extend(tried)
                return imgs, (candidate / "images" if (candidate / "images").exists() and candidate.name != "images" else candidate)

        self._last_debug.setdefault("scanned", []).extend(tried)
        return [], primary

    def _load_from_yaml(self, yaml_path: Path):
        text = Path(yaml_path).read_text(encoding="utf-8")
        data = yaml.safe_load(text) or {}
        yaml_parent = Path(yaml_path).parent
        base = _to_abs(yaml_parent, data.get("path", "."))

        def resolve_dirs_with_source(entry) -> List[Tuple[Path, str]]:
            """
            Returns list of (primary_path, original_entry_text).
            primary_path = base + entry (our standard interpretation)
            """
            if not entry:
                return []
            if isinstance(entry, str):
                return [(_to_abs(base, entry), entry)]
            if isinstance(entry, list):
                return [(_to_abs(base, x), x) for x in entry]
            return []

        train_list = resolve_dirs_with_source(data.get("train"))
        val_list   = resolve_dirs_with_source(data.get("val")) or resolve_dirs_with_source(data.get("valid"))
        test_list  = resolve_dirs_with_source(data.get("test"))

        self._last_debug["yaml_dirs"] = [f"train: {p}" for p, _ in train_list] + \
                                        [f"val: {p}" for p, _ in val_list] + \
                                        [f"test: {p}" for p, _ in test_list]

        entries = {"train": train_list, "val": val_list, "test": test_list}

        for split, pairs in entries.items():
            images: List[Path] = []
            for primary, entry_text in pairs:
                found, chosen_dir = self._scan_images_with_fallbacks(split, primary, entry_text, yaml_parent)
                images.extend(found)
            if images:
                self.splits_map[split] = images

        # labels_map: replace .../images/... with .../labels/... or sibling labels
        for split, images in self.splits_map.items():
            for img in images:
                self.labels_map[img] = self._guess_label_from_image(img)

        # names / nc
        names = data.get("names")
        if isinstance(names, list):
            self.class_names = [str(n) for n in names]
        elif isinstance(names, dict):
            try:
                self.class_names = [names[k] for k in sorted(names, key=lambda x: int(x))]
            except Exception:
                self.class_names = [names[k] for k in sorted(names)]
        self.num_classes = data.get("nc")

    # ---- Non-YAML layouts ----
    def _load_as_single_split(self, root: Path):
        img_dir = root / "images"
        lbl_dir = root / "labels"
        imgs = _gather_images_under(img_dir)
        self._last_debug["single_split"] = [f"images: {img_dir}", f"labels: {lbl_dir}", f"found: {len(imgs)}"]
        if imgs:
            self.splits_map["train"] = imgs
            for img in imgs:
                rel = img.relative_to(img_dir)
                self.labels_map[img] = (lbl_dir / rel).with_suffix(".txt")

    def _load_images_labels_splits(self):
        images_root = self.root / "images"
        labels_root = self.root / "labels"
        if not images_root.exists():
            return
        splits = [p.name for p in images_root.iterdir() if p.is_dir()]
        for split in splits:
            img_dir = images_root / split
            imgs = _gather_images_under(img_dir)
            self._last_debug.setdefault("images/<split>", []).append(f"{img_dir} -> {len(imgs)}")
            if not imgs:
                continue
            self.splits_map[split] = imgs
            for img in imgs:
                rel = img.relative_to(images_root)
                self.labels_map[img] = (labels_root / rel).with_suffix(".txt")

    def _load_split_folders(self):
        for split in ("train", "val", "test", "valid"):
            img_dir = self.root / split / "images"
            if img_dir.exists():
                imgs = _gather_images_under(img_dir)
                self._last_debug.setdefault("<split>/images", []).append(f"{img_dir} -> {len(imgs)}")
                if imgs:
                    key = "val" if split == "valid" else split
                    self.splits_map[key] = imgs
                    for img in imgs:
                        lbl = img.parent.parent / "labels" / (img.stem + ".txt")
                        self.labels_map[img] = lbl

    def _load_split_direct_images(self):
        """Handle layout where split dir holds images directly: train/*.jpg with train/labels/*.txt"""
        for split in ("train", "val", "test", "valid"):
            split_dir = self.root / split
            if split_dir.exists() and split_dir.is_dir():
                imgs = _gather_images_here(split_dir)
                if imgs:
                    key = "val" if split == "valid" else split
                    self._last_debug.setdefault("<split> direct", []).append(f"{split_dir} -> {len(imgs)}")
                    self.splits_map[key] = imgs
                    labels_dir = split_dir / "labels"
                    for img in imgs:
                        self.labels_map[img] = labels_dir / (img.stem + ".txt")

    def _guess_label_from_image(self, img: Path) -> Path:
        parts = list(Path(img).parts)
        try:
            idx = parts.index("images")
            parts[idx] = "labels"
            return Path(*parts).with_suffix(".txt")
        except ValueError:
            p = Path(img)
            if p.parent.name == "images":
                return p.parent.with_name("labels") / (p.stem + ".txt")
            # split_dir/labels
            return p.parent.with_name("labels") / (p.stem + ".txt")

    def _load_class_names_fallback(self):
        classes_txt = self.root / "classes.txt"
        if classes_txt.exists():
            self.class_names = [ln.strip() for ln in classes_txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
