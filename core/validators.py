from typing import List
from .models import Annotation

def validate_annotation(a: Annotation) -> List[str]:
    w, h = a.size
    issues = []
    for i, b in enumerate(a.boxes):
        if b.w <= 0 or b.h <= 0:
            issues.append(f"box {i}: zero/neg size")
        if not (0 <= b.cx <= 1 and 0 <= b.cy <= 1 and 0 <= b.w <= 1 and 0 <= b.h <= 1):
            issues.append(f"box {i}: values out of [0,1]")
        left = (b.cx - b.w/2) * w
        top  = (b.cy - b.h/2) * h
        right = left + b.w * w
        bottom = top + b.h * h
        if left < 0 or top < 0 or right > w or bottom > h:
            issues.append(f"box {i}: out of image bounds")
    return issues
