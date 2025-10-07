# YOLO Editor (Offline Desktop)

YOLO Editor is a Windows desktop app for working with YOLO-style image labels without touching the cloud. Open a dataset, review the images, fix the boxes, and save back to YOLO TXT files.

Built by Srijon Basak.

---

## What It Does (Simple)
- Works offline on Windows with Python 3.11+ and PySide6.
- Opens a YOLO dataset folder or a `data.yaml` file and finds the splits automatically.
- Shows a file tree, the active image, and every YOLO box with class names.
- Lets you add, move, resize, or delete boxes with the mouse and keeps TXT files in sync.
- Displays dataset stats and quick warnings so you spot missing labels early.
- Includes a merge designer tab for remapping classes between datasets and exporting a clean result.

---

## Supported Dataset Layouts
YOLO Editor understands the common YOLOv5/YOLOv8 layouts:
- `root/images/<split>/*` with `root/labels/<split>/*`
- `<split>/images/*` with `<split>/labels/*`
- `<split>/*.jpg` with `<split>/labels/*.txt`
- A `data.yaml` that points to the folders (supports `val`/`valid`, `../train/images`, and similar paths)

If a YAML path is off, the app tries nearby folders and shows what it resolved in the diagnostics window.

---

## Requirements
- Windows 10 or newer (64-bit recommended)
- Python 3.11+
- A GPU is **not** required

---

## Setup (PowerShell)
```powershell
# 1. Create and activate a virtual environment
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
python -m yolo_editor.app
```

The first launch creates a log file under `%USERPROFILE%\.yolo-editor\logs`.

---

## How to Use
1. Launch the app: `python -m yolo_editor.app`.
2. Click **File -> Open Dataset Root...** to pick a folder that contains `train`, `val`/`valid`, or `test`.  
   Or select **File -> Open Dataset YAML...** and choose a `data.yaml`.
3. Select a split, then pick an image from the left tree.
4. Edit boxes in the image view. Use the class dropdown before or after you draw.
5. Press **Save** or hit **S** to write the YOLO TXT file.
6. Use **Tools -> Show Diagnostics...** if the dataset looks wrong.

The right panel shows the YOLO rows for the current image plus per-class counts so you can spot class imbalance.

---

## Handy Shortcuts
- `A`: add a new box (click and drag)
- `Delete`: remove the selected box
- `C`: set selected boxes to the current class
- `S`: save labels
- Left / Right arrow keys or `P` / `N`: previous / next image
- `F`: fit the image to the window
- `1`: zoom to 100%

---

## Merge Designer (Optional)
The merge tab helps you combine multiple datasets or remap classes:
1. Drop in source datasets.
2. Map source classes onto target classes or mark them to drop.
3. Run an export to build a new dataset folder with updated TXT files and a fresh `data.yaml`.
4. Review the manifest file to see what was copied, skipped, or remapped.

Use this when you need to clean up overlapping datasets or reduce classes before training.

---

## Packaging (PyInstaller)
The project ships with PyInstaller settings so you can build a distributable folder:
```powershell
.\.venv\Scripts\Activate.ps1
pyinstaller yolo_editor.spec
```

The onedir build lives in `dist\YOLO Editor\`. You can wrap it with Inno Setup if you want a simple installer.

---

## Troubleshooting Tips
- **App does not start**: run `python -m yolo_editor.app` from PowerShell to see the traceback.
- **No images detected**: open the diagnostics window to check which folders were scanned.
- **PySide6 plugin error in the packaged build**: include `--collect-all PySide6` when running PyInstaller.
- **Execution policy blocks the venv**: use `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.

---

## Cite This Project
If YOLO Editor helps your research, presentations, or product, please cite it as:

> Basak, S. (2025). *YOLO Editor: Offline Desktop Tool for YOLO Labels* (Version 1.0.0) [Software]. https://github.com/srijonbasak/yolo_editor

### BibTeX
```bibtex
@software{basak_yolo_editor_2025,
  author  = {Srijon Basak},
  title   = {YOLO Editor: Offline Desktop Tool for YOLO Labels},
  year    = {2025},
  month   = {01},
  version = {1.0.0},
  url     = {https://github.com/srijonbasak/yolo_editor},
  license = {MIT}
}
```

---

## License
Copyright (c) 2025 Srijon Basak

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

---

## Contact
Questions or bugs? Reach out at `srijonbasak76@gmail.com`.
