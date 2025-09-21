# YOLO Editor (Offline, Desktop)
Fast offline viewer/editor for YOLOv8 labels. Supports Ultralytics YAML (train/val/test, valid alias) and both directory layouts:
- `images/<split>` + `labels/<split>`
- `<split>/images` + `<split>/labels`

## Run
```bash
python -m venv .venv
# Windows PowerShell:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m yolo_editor.app
