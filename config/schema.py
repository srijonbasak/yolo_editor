from pydantic import BaseModel, Field
from typing import List, Optional

class Keymap(BaseModel):
    next_image: str = "N"
    prev_image: str = "P"
    save: str = "S"
    add_box: str = "A"
    delete_box: str = "Delete"
    change_class: str = "C"
    fit: str = "F"
    zoom_100: str = "1"

class AppConfig(BaseModel):
    theme: str = "dark"
    recent_paths: List[str] = Field(default_factory=list)
    keymap: Keymap = Field(default_factory=Keymap)

class DatasetConfig(BaseModel):
    root: str
    classes: Optional[List[str]] = None
