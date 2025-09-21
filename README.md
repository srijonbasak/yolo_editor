# YOLO Editor (Offline Desktop)
*A fast, offline desktop tool to view, edit, validate, and remap YOLOv8 labels — built by **Srijon Basak**.*

![Platform](https://img.shields.io/badge/platform-Windows-blue)
![Python](https://img.shields.io/badge/python-3.11%2B-3776AB)
![License](https://img.shields.io/badge/license-MIT-green)

> **Why this exists**  
> I kept hitting friction during label QA: tools that break on unusual folder layouts, YAMLs that point one level off (`../train/images`), GUIs that only show class IDs (not names), and lag/crashes with large datasets. I wanted a **portable, zero-cloud** app that “just works” with any YOLOv8 dataset.

---

## 📷 Screenshots

> Put your images in `assets/` and update the names if needed.

![Main UI](assets/main_ui.png)
![Editing Boxes](assets/editing_boxes.png)
![Diagnostics](assets/diagnostics.png)

---

## ✨ Highlights

- **Open anything**
  - Open a **dataset root** *or* a **`data.yaml`** directly.
  - Handles all common layouts (and quirky ones):
    - `images/<split>/*` + `labels/<split>/*`
    - `<split>/images/*` + `<split>/labels/*`
    - `<split>/*.(jpg|png|…)` + `<split>/labels/*.txt`
    - `valid` alias for `val`
    - Smart fallbacks if YAML paths are off (e.g., `../train/images` when folders are adjacent to the YAML)
- **Edit labels quickly**
  - Add boxes (drag), **move**, **resize via corner handles**, **delete**.
  - Change selected boxes to the current class.
  - Saves to YOLO txt (normalized **cx cy w h**) safely/atomically.
  - Always shows **class names** (from YAML) next to IDs.
- **Navigate fast**
  - Arrow keys ← / → (or **P / N**) for prev/next.
  - Fit to window (**F**), 100% zoom (**1**), rubber-band selection.
- **Know your data**
  - Per-split image counts.
  - Per-class coverage (#images) and #boxes—spot class imbalance fast.
- **Merge-ready (extension)**
  - Class remapping (e.g., 13 → 7 classes) & export to a clean output dataset (dir layout + updated YAML).

---

## 🧱 Supported Dataset Layouts

All of these work:

```text
A) images/<split> and labels/<split>
root/
  images/
    train/ *.jpg
    val/   *.jpg
    test/  *.jpg
  labels/
    train/ *.txt
    val/   *.txt
    test/  *.txt

B) <split>/images and <split>/labels
root/
  train/
    images/ *.jpg
    labels/ *.txt
  val/
    images/ *.jpg
    labels/ *.txt
  test/
    images/ *.jpg
    labels/ *.txt

C) Images directly inside split, labels inside split/labels
root/
  train/ *.jpg
  train/labels/*.txt
  val/   *.jpg
  val/labels/*.txt

D) YAML-driven (Ultralytics style, including 'valid' alias and odd ../ paths)
root/
  data.yaml
  train/...
  valid/... or val/...
  test/...
YAML example (Ultralytics):

```train: ../train/images
val: ../valid/images
test: ../test/images
nc: 6
names: ['APC-IFV', 'ART', 'CAR', 'MLRS', 'TANK', 'TRUCK']
```
        If those ../ paths don’t match your structure, the app auto-fallbacks to neighbors (e.g., tries train/images next to the YAML). Use File → Show Dataset Diagnostics to see exactly what it resolved and scanned.


⌨️ Shortcuts

Add box: A

Delete selected: Delete

Change selected to current class: C

Save labels: S

Previous/Next image: ← / → or P / N

Fit to window: F

100% zoom: 1

🛠️ Install & Run (from Source)

    Recommended build/runtime: Python 3.11 on Windows 64-bit.
```# 1) Create and activate a venv (PowerShell)
python -m venv .venv
# If activation is blocked:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

# 2) Install deps
pip install -r requirements.txt

# 3) Launch the app
python -m yolo_editor.app
```
If you’re on Python 3.13 and hit a NumPy/OpenCV binary mismatch, pin:
```pip install "opencv-python==4.9.0.80"
```