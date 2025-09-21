from pydantic import BaseModel, Field, conint, confloat
from typing import List, Tuple

Norm = confloat(ge=0.0, le=1.0)

class BBox(BaseModel):
    cls: conint(ge=0)
    cx: Norm; cy: Norm; w: Norm; h: Norm

    def as_tuple(self):
        return (self.cls, self.cx, self.cy, self.w, self.h)

class Annotation(BaseModel):
    image_path: str
    size: Tuple[int, int]  # (w,h)
    boxes: List[BBox] = Field(default_factory=list)
