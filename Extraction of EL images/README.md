# Panel Grid Imaging Pipeline

Tools for processing a batch of fisheye photos of a gridded panel (e.g. an
EL/PL-style module made of rows × columns of cells) into clean, individually
extracted cell images, a per-shot stitched grid, and one big composite mosaic
of every shot in the set.

Each capture goes through the same general stages:

1. **Undistort** the fisheye photo using fixed camera intrinsics.
2. **Crop** to just the panel region (ratio table keyed by grid `rows × cols`,
   with an optional OCR read of on-image "rows=/cols=" text to auto-detect
   the grid size).
3. **Classify** the cropped panel as a "bright" or "dark" image (mean
   brightness / contrast / bright-pixel ratio).
4. **Detect and extract cells** using whichever algorithm matches that
   classification:
   - *Bright*: denoise → illumination correction → CLAHE → Otsu threshold →
     morphology → row/column darkness-valley zone detection.
   - *Dark*: CLAHE/gamma/sharpen → Sobel edge filtering → gradient-based
     grid-line detection → Hough line intersections → uniform-block
     selection.
5. **Stitch** the extracted cells back into one grid image per shot.
6. **Composite** all per-shot grids into a single mosaic, normalized to a
   common pixel range and laid out in segments/columns.

As with the other pipeline, these files are an evolution rather than
independent tools — each script/notebook built on the one before it.

## File-by-file (in development order)

| File | What it is |
|------|------------|
| `bright_panel.ipynb` | Prototype notebook for the **bright-image** cell-detection algorithm: panel crop → denoise/illumination/CLAHE → Otsu binarize → darkness-valley zone detection → cell stitch. Worked out against a single test image. |
| `dark_panel.ipynb` | A much larger, messier prototype notebook for **low-contrast / dark-image** cell detection — a different approach using column/row projection profiles, peak detection, blob removal, gap-filling, and Hough-line grid intersection. Contains many iterative experiment cells (including dead/empty ones) — this is the R&D ground for the harder case the bright-panel algorithm can't handle. |
| `differentiate.ipynb` | A small classifier notebook: given a cropped panel image, computes mean brightness, contrast (std), and the fraction of bright pixels, then labels the image **"DARK IMAGE"** or **"BRIGHT IMAGE"** against fixed thresholds. Also holds the shared fisheye camera intrinsics/undistortion config. This is the decision logic that later determines which of the two algorithms above to run. |
| `pipeline.py` | First consolidated script. Wraps the **bright-panel** notebook logic into a full folder-batch pipeline: undistort → OCR/ratio-table panel crop → preprocess → binarize → connected components → zone-based cell extraction → stitch. Saves a numbered debug PNG for every step (`0_original.png` … `8_original_stitched.png`) per image, sequential (no dark-image branch yet), into `Output2/<folder>/`. |
| `pipeline1.py` | Major upgrade — merges everything above into one script. Adds a **Step 0 classifier** (from `differentiate.ipynb`) that routes each image to either `bright_pipeline()` or `dark_pipeline()` (the full Sobel/Hough grid-detection logic from `dark_panel.ipynb`). Also auto-detects whether a folder holds 12 or 16 images and adjusts the final composite layout accordingly. This is the most algorithmically complete version, but still sequential and still writes full debug PNGs per step to `Output2/`. |
| `api.py` | Production/importable rewrite (`pipeline_api.py`, exposes `process_folder()` + a CLI). Drops the per-step debug images, processes all images **in parallel** with a `ThreadPoolExecutor`, and returns an in-memory result dict (composite array, paths, timing, success count) instead of just printing to console. **Note:** as written it only implements the *bright-image* code path from `pipeline1.py` — the dark-image branch hasn't been ported over yet. Output goes to `Output/<folder>/`. |
| `User_Friendly_GUI.py` | A Tkinter desktop GUI (`pipeline_gui.py`) wrapping `api.process_folder()`. Tab 1 shows the per-image stitched grids as thumbnails; Tab 2 shows the final composite in a zoomable/pannable canvas (touchpad pan/zoom, +/- buttons). Includes a folder picker, progress feedback, and an export/save-as for the composite. **Imports `api_final`** — rename/alias `api.py` to `api_final.py` (or add an `import api as api_final`) for the GUI to find it. |

**Practical takeaway:**
- For the most accurate cell detection on a mix of bright and dark captures, use **`pipeline1.py`** — it's the only version with both algorithms wired together.
- For speed and easy integration into other code (or the GUI), use **`api.py`** — but first confirm your captures are all "bright"-classified, since the dark-image branch isn't in there yet.
- The GUI (`User_Friendly_GUI.py`) is the easiest way to run things end-to-end once `api.py` covers your image type.

## Requirements

```bash
pip install numpy opencv-python pillow scipy matplotlib
# optional, only used for OCR-based grid-size auto-detection:
pip install pytesseract   # plus the system tesseract binary
# GUI only:
# tkinter ships with most Python installs; install python3-tk on Linux if missing
```

## Usage

```bash
# Most complete (bright + dark branching), writes full debug steps
python pipeline1.py /path/to/folder --dark-rows 3 --dark-cols 2 --bright-rows 3 --bright-cols 3

# Fast / importable, bright-image path only
python api.py /path/to/folder --rows 3 --cols 3 --workers 8
```

```python
from api import process_folder
result = process_folder(folder_path="/path/to/folder", rows=3, cols=3)
```

```bash
# GUI
python User_Friendly_GUI.py
```

Input folders are expected to contain numbered images (`1.jpeg … 16.jpeg`,
or `12.jpeg` for the smaller layout) — one photo per panel position.

## Output

- `pipeline.py` / `pipeline1.py` → `Output2/<folder_name>/<img_num>/...` (full debug steps) + `Output2/<folder_name>/final_image.png`
- `api.py` (and the GUI, via `api.process_folder`) → `Output/<folder_name>/<img_num>/final_image.jpeg` + `cropped_cell*.jpeg`, plus `Output/<folder_name>/final_image.jpeg` for the composite.

## Key config knobs

- `K`, `DIST`, `UNDISTORT_BALANCE` — fisheye camera calibration / undistortion strength
- `CROP_RATIOS` — per-(rows, cols) panel crop box, as fractions of image width/height
- `MEAN_THRESH`, `STD_THRESH`, `BRIGHT_RATIO_THRESH` — dark vs bright classification thresholds (`differentiate.ipynb`, `pipeline1.py`)
- `DENOISE_KERNEL`, `BG_SIGMA`, `CLAHE_CLIP_LIMIT/TILE_GRID` — bright-path preprocessing
- `ZONE_*` — bright-path row/column zone (cell) detection
- `NUM_SUB_SEGMENT`, `NUM_SEGMENT`, `MID_SUB_SEGMENT_FIRST`, `INTER_CELL_GAP`, `SEGMENT_GAP`, `COLUMN_GAP` — final composite mosaic layout