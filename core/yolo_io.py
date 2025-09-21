from pathlib import Path
from typing import List, Tuple

def parse_label_file(path: Path) -> List[Tuple[int, float, float, float, float]]:
    if not path.exists():
        return []
    rows = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        if len(parts) != 5:
            continue
        try:
            cls = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:])
            rows.append((cls, x, y, w, h))
        except Exception:
            continue
    return rows

def save_label_file(path: Path, rows: List[Tuple[int, float, float, float, float]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for cls, x, y, w, h in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
