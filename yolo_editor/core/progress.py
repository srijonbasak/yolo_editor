from dataclasses import dataclass
from typing import Callable

@dataclass
class Progress:
    total: int
    value: int = 0
    def step(self, n=1): self.value += n

ProgressCallback = Callable[[Progress], None]

class CancelToken:
    def __init__(self): self._cancelled = False
    def cancel(self): self._cancelled = True
    def is_cancelled(self): return self._cancelled
