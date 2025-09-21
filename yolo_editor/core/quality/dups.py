from __future__ import annotations
from pathlib import Path
from PIL import Image
import imagehash
import logging

logger = logging.getLogger(__name__)

def phash_hex(path: Path) -> str:
    """Generate perceptual hash for an image."""
    try:
        with Image.open(path) as im:
            return str(imagehash.phash(im))
    except Exception as e:
        logger.warning(f"Failed to generate phash for {path}: {e}")
        return ""

def too_similar(h1: str, h2: str, max_dist: int = 6) -> bool:
    """Check if two hashes are too similar."""
    try:
        if not h1 or not h2:
            return False
        return imagehash.hex_to_hash(h1) - imagehash.hex_to_hash(h2) <= max_dist
    except Exception as e:
        logger.warning(f"Failed to compare hashes {h1} and {h2}: {e}")
        return False
