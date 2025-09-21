from pathlib import Path
from typing import List

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

class DatasetRepository:
    def __init__(self, root: Path):
        self.root = root
        self.images_dir = root / "images"
        self.labels_dir = root / "labels"

    def list_images(self) -> List[Path]:
        return sorted(p for p in self.images_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)

    def label_path_for(self, image_path: Path) -> Path:
        rel = image_path.relative_to(self.images_dir).with_suffix(".txt")
        return (self.labels_dir / rel)

    def classes_file(self) -> Path:
        return self.root / "classes.txt"
