from __future__ import annotations
from pathlib import Path
from typing import Iterable, Set, Tuple
import yaml

from .merge_model import MergePlan, SplitStrategy, CopyMode, CollisionPolicy
from .yolo_io import parse_label_file, save_label_file
from .fsops import hardlink_or_copy, resolve_collision_name, safe_mkdirs
from .progress import Progress, ProgressCallback, CancelToken
from .splitting import normalize_split

ImgKey = Tuple[str, Path]  # (dataset_id, image_path)

def _dst_split_name(plan: MergePlan, source_split: str) -> str:
    norm = normalize_split(source_split)
    if plan.split_strategy == SplitStrategy.FLATTEN:
        return plan.target_train_name
    if norm == "train": return plan.target_train_name
    if norm == "val":   return plan.target_val_name   # -> "eval" per spec
    if norm == "test":  return plan.target_test_name
    return plan.target_train_name

def merge_execute(
    plan: MergePlan,
    sources: Iterable,                 
    progress_cb: ProgressCallback | None = None,
    cancel: CancelToken | None = None,
    selection: Set[ImgKey] | None = None, 
):
    out_root = plan.output_dir
    for split in (plan.target_train_name, plan.target_val_name, plan.target_test_name):
        safe_mkdirs(out_root / split / "Image")
        safe_mkdirs(out_root / split / "Labels")

    total = sum(len(ds.repo.list_images()) for ds in sources)
    prog = Progress(total=total)

    for ds in sources:
        repo = ds.repo
        for split, imgs in repo.splits_map.items():
            dst_split = _dst_split_name(plan, split)
            for img in imgs:
                if cancel and cancel.is_cancelled():
                    return
                if selection is not None and (ds.id, img) not in selection:
                    prog.step(); 
                    if progress_cb: progress_cb(prog)
                    continue

                rows = parse_label_file(repo.label_path_for(img))
                mapped_rows = []
                for (cls, x, y, w, h) in rows:
                    tgt = plan.mapping.get((ds.id, cls), None)
                    if tgt is None:
                        continue
                    mapped_rows.append((tgt, x, y, w, h))

                if plan.drop_empty_images and not mapped_rows:
                    prog.step()
                    if progress_cb: progress_cb(prog)
                    continue

                dst_img = out_root / dst_split / "Image" / img.name
                safe_mkdirs(dst_img.parent)
                if plan.collision_policy == CollisionPolicy.RENAME and dst_img.exists():
                    dst_img = resolve_collision_name(dst_img, img, ds.id)
                elif plan.collision_policy == CollisionPolicy.SUBDIRS:
                    dst_img = out_root / dst_split / "Image" / ds.id / img.name
                    safe_mkdirs(dst_img.parent)

                dst_lbl = out_root / dst_split / "Labels" / (dst_img.stem + ".txt")
                safe_mkdirs(dst_lbl.parent)

                hardlink_or_copy(img, dst_img, prefer_hardlink=(plan.copy_mode == CopyMode.HARDLINK))
                save_label_file(dst_lbl, mapped_rows)

                prog.step()
                if progress_cb: progress_cb(prog)

    names = [tc.name for tc in plan.target_classes]
    yaml_obj = {
        "path": ".",
        "train": f"{plan.target_train_name}/Image",
        "val":   f"{plan.target_val_name}/Image",   # Ultralytics expects 'val'
        "test":  f"{plan.target_test_name}/Image",
        "nc": len(names),
        "names": names,
    }
    if plan.target_val_name != "val":
        yaml_obj[plan.target_val_name] = f"{plan.target_val_name}/Image"

    with open(out_root / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_obj, f, sort_keys=False)
