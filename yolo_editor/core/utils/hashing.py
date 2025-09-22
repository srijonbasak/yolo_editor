from __future__ import annotations
import hashlib

def stable_int_key(*parts: str, seed: int = 1337) -> int:
    h = hashlib.blake2s(digest_size=8)
    for p in parts:
        h.update(p.encode())
    h.update(str(seed).encode())
    return int.from_bytes(h.digest(), "big", signed=False)
