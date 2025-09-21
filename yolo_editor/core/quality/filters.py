from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path

def blur_score(img_bgr) -> float:
    return float(cv2.Laplacian(img_bgr, cv2.CV_64F).var())

def exposure_score(img_bgr) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))

def meets_min_resolution(img_bgr, min_w=320, min_h=240) -> bool:
    h, w = img_bgr.shape[:2]
    return (w >= min_w and h >= min_h)

def load_bgr(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)  # Windows-safe
    return cv2.imdecode(data, cv2.IMREAD_COLOR)
