# Topcon SEG Image Pipeline

Tools for turning a set of three split scan segments (`SEG_1.png`, `SEG_2.png`,
`SEG_3.png`) into a single clean, stitched panorama. The pipeline:

1. **Extract** the bright strip of interest from each segment image.
2. **Detect column borders** inside that strip and crop to content.
3. **Stretch** the cropped panel horizontally/vertically if needed.
4. **Trim** a per-segment top margin (hand-tuned per dataset).
5. **Stitch** the three panels side by side.

The files in this folder are not separate tools — they're successive
iterations of the same pipeline, going from rough notebook experiments to a
parallelized, production-style script. They're kept together so you can see
how each version built on the last.

## File-by-file (in development order)

| # | File | What it is |
|---|------|------------|
| 1 | `app.ipynb` | **Earliest prototype**, exploratory Jupyter cells. Works out the bright-strip extraction logic, the gradient/valley-based column-border detection, and a simple Lanczos stretch — each as a standalone cell against a single test image. No folder structure, no functions shared between cells. |
| 2 | `app.py` | First "real" script. Wraps the notebook logic into reusable functions (`extract_strip`, `detect_and_crop`, `stitch_panels`) and runs all 3 segments through a `main()`. Saves every intermediate step (original, grayscale, profile plots, cropped strip, etc.) into a numbered `Output/<folder>/1_original/ … 7_stitched/` tree for debugging. |
| 3 | `final_app.py` | Same as `app.py`, plus hardcoded per-segment top-crop amounts (`TOP_CUT_SEG1/2/3`) applied just before stitching, to remove a few rows of misalignment at the top of each panel. |
| 4 | `normalized.ipynb` | A **side-experiment**, not merged into the main scripts. Explores column-by-column intensity normalization/flattening on an already-extracted strip (reads from `Output/.../4_strip_extracted/SEG_1.png`). Useful reference if you want to add intensity normalization as a future stage. |
| 5 | `app1.py` | Restructures the output layout to one folder per segment (`SEG_1/`, `SEG_2/`, `SEG_3/`, `8_stitched/`) instead of step-named folders. Adds a real **Stage-3 stretch step** (`SCALE_X`/`SCALE_Y` with PIL Lanczos) and produces two stitched results: a normal left-to-right stitch and an "inverted" version (each panel mirrored, with SEG_1/SEG_3 positions swapped). |
| 6 | `optimised.py` | **Performance rewrite** of `app1.py`. Drops all the intermediate-step disk I/O (no per-stage PNGs), loads images once with `cv2.imread` instead of PIL, and processes all three segments **in parallel** with a `ThreadPoolExecutor`. Stretching is done with `cv2.resize` (Lanczos) instead of PIL. Includes a `Timer` class that prints a timing breakdown at the end. Only the final normal/inverted stitched images + an overview plot are written to disk. |
| 7 | `optimesd_final.py` | `optimised.py` plus a new **Stage-0: white-border removal** (`remove_white_border`), run before grayscale conversion and strip extraction. It detects rows/columns that are mostly white (above `WHITE_BORDER_THRESH`) and trims them from all four edges, with a safety guard so it won't over-crop tiny images. This is the most complete/current version. |

**Practical takeaway:** if you just want to run the pipeline, use
**`optimesd_final.py`** — it's the newest, fastest, and most feature-complete
version. The others are kept for reference/history.

## Requirements

```bash
pip install numpy pillow opencv-python scipy matplotlib
```

## Usage

```bash
python optimesd_final.py /path/to/Images
# or rely on the FOLDER_PATH default set at the top of the script
```

The input folder must contain `SEG_1.png`, `SEG_2.png`, `SEG_3.png` (missing
segments are skipped with a warning).

## Output (optimesd_final.py / optimised.py)

```
Output/<input_folder_name>/8_stitched/
├── stitched_normal.png      # SEG_1 | SEG_2 | SEG_3, left to right
├── stitched_inverted.png    # each panel mirrored, SEG_1 ↔ SEG_3 swapped
└── overview.png             # side-by-side summary plot
```

(`app.py` / `final_app.py` / `app1.py` instead write a full step-by-step debug
tree under `Output/<input_folder_name>/...` — see each script's header
comment for the exact layout.)

## Key config knobs (top of each script)

- `BRIGHT_THRESH`, `SMOOTH_WINDOW`, `VALLEY_SEARCH` — Stage 1 strip extraction
- `GAUSS_KERNEL`, `GRADIENT_SIGMA_MULT`, `PEAK_DISTANCE` — Stage 2 column-border detection
- `SCALE_X`, `SCALE_Y` — Stage 3 stretch factor
- `TOP_CUT_SEG1/2/3` — per-segment top-row trim before stitching
- `WHITE_BORDER_THRESH`, `WHITE_BORDER_COL_RATIO`, `WHITE_BORDER_ROW_RATIO` — Stage 0 white-border removal (only in `optimesd_final.py`)

Tune these against your own images if detection looks off — the notebooks
(`app.ipynb`, `normalized.ipynb`) are good places to test changes on a single
image before updating the main script.