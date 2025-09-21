import hashlib, os, shutil
from pathlib import Path

def safe_mkdirs(p: Path): p.mkdir(parents=True, exist_ok=True)

def hardlink_or_copy(src: Path, dst: Path, prefer_hardlink: bool = True):
    safe_mkdirs(dst.parent)
    if dst.exists(): return
    if prefer_hardlink:
        try:
            os.link(src, dst)
            return
        except Exception:
            pass
    shutil.copy2(src, dst)

def slugify(s: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in s).strip("-")

def hashed_suffix(path: Path, dataset_id: str) -> str:
    h = hashlib.blake2s(f"{dataset_id}:{path.as_posix()}".encode(), digest_size=5).hexdigest()
    return f"__{slugify(dataset_id)}_{h}"

def resolve_collision_name(dst_img: Path, src_path: Path, dataset_id: str) -> Path:
    if not dst_img.exists(): return dst_img
    stem, ext = dst_img.stem, dst_img.suffix
    return dst_img.with_name(f"{stem}{hashed_suffix(src_path, dataset_id)}{ext}")
