from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple, Optional

DatasetID = str
ClassID   = int
EdgeKey   = Tuple[DatasetID, ClassID]

class SplitStrategy(str, Enum):
    KEEP = "keep"
    FLATTEN = "flatten"

class CopyMode(str, Enum):
    HARDLINK = "hardlink"
    COPY = "copy"

class CollisionPolicy(str, Enum):
    RENAME = "rename"
    SUBDIRS = "subdirs"
    SKIP = "skip"

class BalanceMode(str, Enum):
    EQUAL = "equal"
    PROP  = "prop"

@dataclass
class TargetClass:
    index: int
    name: str

@dataclass
class MergePlan:
    name: str
    output_dir: Path
    target_classes: List[TargetClass]

    mapping: Dict[EdgeKey, Optional[int]]                         # None => DROP
    target_quota: Dict[int, int] = field(default_factory=dict)    # target_index -> desired images
    edge_limit: Dict[EdgeKey, int] = field(default_factory=dict)  # max images from that edge
    balance_mode: BalanceMode = BalanceMode.EQUAL
    random_seed: int = 1337

    split_strategy: SplitStrategy = SplitStrategy.KEEP
    copy_mode: CopyMode = CopyMode.HARDLINK
    collision_policy: CollisionPolicy = CollisionPolicy.RENAME
    drop_empty_images: bool = True

    target_train_name: str = "train"
    target_val_name: str   = "val"
    target_test_name: str  = "test"

    def to_json(self) -> dict:
        d = asdict(self)
        d["output_dir"] = str(self.output_dir)
        return d
