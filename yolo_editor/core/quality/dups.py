from __future__ import annotations
from pathlib import Path
from PIL import Image
import imagehash

def phash_hex(path: Path) -> str:
    with Image.open(path) as im:
        return str(imagehash.phash(im))

def too_similar(h1: str, h2: str, max_dist: int = 6) -> bool:
    return imagehash.hex_to_hash(h1) - imagehash.hex_to_hash(h2) <= max_dist
