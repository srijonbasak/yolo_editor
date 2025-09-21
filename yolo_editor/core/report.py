from __future__ import annotations
import json, datetime, platform
from pathlib import Path
from typing import Tuple

from .merge_model import MergePlan
from .merge_selector import SelectionResult

ImgKey = Tuple[str, Path]

def write_report(out_root: Path, plan: MergePlan, sel: SelectionResult | None):
    rep = {
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "os": platform.platform(),
        "plan": plan.to_json(),
        "preview_supply": sel.preview_supply if sel else {},
        "preview_edges": {
            str(t): [ [list(edge), supply, taken] for (edge, supply, taken) in rows ]
            for t, rows in (sel.preview_edges.items() if sel else [])
        } if sel else {},
        "warnings": sel.warnings if sel else [],
        "totals": {
            "selected_images": len(sel.selected_images) if sel else None,
            "by_target": { str(t): len(v) for t,v in (sel.by_target.items() if sel else []) }
        } if sel else {},
    }
    reports = out_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "merge_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
