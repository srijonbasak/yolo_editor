from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def blur_score(img_bgr) -> float:
    """Calculate blur score using Laplacian variance."""
    try:
        if img_bgr is None or img_bgr.size == 0:
            return 0.0
        return float(cv2.Laplacian(img_bgr, cv2.CV_64F).var())
    except Exception as e:
        logger.warning(f"Failed to calculate blur score: {e}")
        return 0.0

def exposure_score(img_bgr) -> float:
    """Calculate exposure score (mean brightness)."""
    try:
        if img_bgr is None or img_bgr.size == 0:
            return 0.0
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))
    except Exception as e:
        logger.warning(f"Failed to calculate exposure score: {e}")
        return 0.0

def meets_min_resolution(img_bgr, min_w=320, min_h=240) -> bool:
    """Check if image meets minimum resolution requirements."""
    try:
        if img_bgr is None or img_bgr.size == 0:
            return False
        h, w = img_bgr.shape[:2]
        return (w >= min_w and h >= min_h)
    except Exception as e:
        logger.warning(f"Failed to check resolution: {e}")
        return False

def load_bgr(path: Path):
    """Load image as BGR array (Windows-safe)."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)  # Windows-safe
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.warning(f"Failed to load image {path}: {e}")
        return None
