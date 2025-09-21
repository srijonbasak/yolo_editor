from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
from .repo import DatasetRepository

@dataclass
class LoadedDataset:
    id: str
    name: str
    repo: DatasetRepository

class MultiRepo:
    def __init__(self): self.items: Dict[str, LoadedDataset] = {}
    def add(self, ds_id: str, root: Path, yaml_path: Optional[Path] = None, display_name: Optional[str] = None):
        repo = DatasetRepository(root=root, yaml_path=yaml_path)
        name = display_name or (yaml_path.name if yaml_path else root.name)
        self.items[ds_id] = LoadedDataset(ds_id, name, repo)
    def __iter__(self): return iter(self.items.values())
    def __len__(self): return len(self.items)
