#!/usr/bin/env python3
"""
Full Image Processing Pipeline
================================
Input : folder path containing 1.jpeg … 16.jpeg
Output: Output2/<folder_name>/<img_num>/0_original.png … 8_original_stitched.png
        Output2/<folder_name>/final_image.png

Usage:
    python pipeline.py /path/to/folder
    python pipeline.py /path/to/folder --rows 3 --cols 3
"""

import argparse
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless – no display needed
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CONFIGURATION  (edit these as needed)
# ══════════════════════════════════════════════════════════════════════════════

NUM_IMAGES = 16

# ── Camera intrinsics ─────────────────────────────────────────────────────────
K = np.array([
    [1184.8667983506784, 0.0,                939.9321230074511],
    [0.0,                1182.0885688815326, 604.8672689311147],
    [0.0,                0.0,                1.0]
], dtype=np.float64)

DIST = np.array([
    [ 0.20332565713132572],
    [ 0.042418077818231245],
    [ 0.5625598381701257],
    [-0.6590457476513796]
], dtype=np.float64)

UNDISTORT_BALANCE = 0.6

# ── Crop-ratio table (rows, cols) → (left_r, right_r, top_r, bottom_r) ──────
CROP_RATIOS = {
    (1, 1): (0.30, 0.70, 0.35, 0.68),
    (1, 2): (0.18, 0.50, 0.23, 0.70),
    (1, 3): (0.18, 0.50, 0.15, 0.78),
    (1, 4): (0.18, 0.50, 0.03, 0.78),
    (2, 1): (0.18, 0.70, 0.30, 0.68),
    (2, 2): (0.18, 0.75, 0.15, 0.70),
    (2, 3): (0.18, 0.85, 0.15, 0.66),
    (2, 4): (0.18, 0.75, 0.03, 0.78),
    (3, 1): (0.18, 0.85, 0.30, 0.62),
    (3, 2): (0.18, 0.73, 0.15, 0.78),
    (3, 3): (0.18, 0.85, 0.16, 0.78),
    (3, 4): (0.18, 0.90, 0.17, 0.95),
    (4, 1): (0.01, 0.98, 0.38, 0.64),
    (4, 2): (0.01, 0.98, 0.13, 0.80),
    (4, 3): (0.01, 0.98, 0.15, 0.78),
    (4, 4): (0.01, 0.98, 0.05, 0.78),
}

# ── Pipeline defaults ─────────────────────────────────────────────────────────
DEFAULT_COLS                  = 3
DEFAULT_ROWS                  = 3
BRIGHT_MARGIN                 = 2
BRIGHT_VALLEY_MIN_PX          = 20
BRIGHT_CELL_H_PAD             = 7
BRIGHT_CELL_V_PAD_TOP         = 12
BRIGHT_CELL_V_PAD_BOTTOM      = 13

# ── Final stitching layout ────────────────────────────────────────────────────
INTER_CELL_GAP        = 2
SEGMENT_GAP           = 15
COLUMN_GAP            = 12
NUM_SUB_SEGMENT       = 8
NUM_SEGMENT           = 2
MID_SUB_SEGMENT_FIRST = 4


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def save(path: str, img) -> None:
    """Save an OpenCV image, creating parent dirs as needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(path, img)
    print(f"    [SAVE] {path}")


def pil_save(path: str, img_pil: Image.Image) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img_pil.save(path)
    print(f"    [SAVE] {path}")


def log(msg: str) -> None:
    print(f"  {msg}")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 – FISHEYE UNDISTORTION
# ══════════════════════════════════════════════════════════════════════════════

def undistort(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, DIST, (w, h), np.eye(3), balance=UNDISTORT_BALANCE)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, DIST, np.eye(3), new_K, (w, h), cv2.CV_16SC2)
    return cv2.remap(img_bgr, map1, map2,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 – PANEL CROP
# ══════════════════════════════════════════════════════════════════════════════

def _ocr_grid(img_bgr):
    """Try to read grid dimensions from image header via OCR."""
    try:
        import pytesseract
        h = img_bgr.shape[0]
        header = img_bgr[:int(h * 0.12), :]
        text = pytesseract.image_to_string(header)
        r = re.search(r'rows?\s*[=:]\s*(\d+)', text, re.IGNORECASE)
        c = re.search(r'col(?:umns?)?\s*[=:]\s*(\d+)', text, re.IGNORECASE)
        if r and c:
            return int(r.group(1)), int(c.group(1))
    except ImportError:
        pass
    return None, None


def crop_panel(img_bgr: np.ndarray, rows: int, cols: int) -> np.ndarray:
    rows = max(1, min(4, rows))
    cols = max(1, min(4, cols))
    left_r, right_r, top_r, bottom_r = CROP_RATIOS[(rows, cols)]
    h, w = img_bgr.shape[:2]
    return img_bgr[int(h * top_r):int(h * bottom_r),
                   int(w * left_r):int(w * right_r)]


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE  (Steps 3-8)
# ══════════════════════════════════════════════════════════════════════════════

def pipeline(img_bgr: np.ndarray, out_dir: str,
             rows: int = DEFAULT_ROWS,
             cols: int = DEFAULT_COLS) -> np.ndarray:
    """
    Process a panel image.
    Saves intermediate outputs and returns the final stitched image (BGR).
    """
    log(f"Running pipeline … (rows={rows}, cols={cols})")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape

    # ── Step 3: Denoise → illumination correction → CLAHE ────────────────────
    denoised   = cv2.GaussianBlur(gray, (5, 5), 0)
    background = cv2.GaussianBlur(denoised, (0, 0), 25)
    normalized = cv2.divide(denoised, background, scale=255)
    clahe      = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    preprocessed = clahe.apply(normalized)
    save(f"{out_dir}/3_preprocessed.png", preprocessed)
    log("Step 3 done – Denoise / Illumination / CLAHE")

    # ── Step 4: Binarize (Otsu) ───────────────────────────────────────────────
    blurred = cv2.GaussianBlur(preprocessed, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    save(f"{out_dir}/4_binary.png", binary)
    log("Step 4 done – Binary (Otsu)")

    # ── Step 5: Morphological clean-up ───────────────────────────────────────
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph  = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    morph  = cv2.morphologyEx(morph,  cv2.MORPH_OPEN,  kernel, iterations=1)
    save(f"{out_dir}/5_morph.png", morph)
    log("Step 5 done – Morphological")

    # ── Step 6: Connected components (colour-coded) ───────────────────────────
    num_labels, labels = cv2.connectedComponents(morph)
    rng = np.random.default_rng(42)
    comp_vis = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for lbl in range(1, num_labels):
        comp_vis[labels == lbl] = rng.integers(0, 255, size=3, dtype=np.uint8)
    save(f"{out_dir}/6_components.png", comp_vis)
    log(f"Step 6 done – {num_labels - 1} connected components")

    # ── Step 7: Cell extraction + stitching ───────────────────────────────────
    x0 = int(W * 0.20); x1 = int(W * 0.80)
    v_profile = binary[:, x0:x1].astype(float).mean(axis=1)
    row_zones  = _find_zones(v_profile, rows)

    ry0, ry1   = row_zones[0][0], row_zones[-1][1]
    ry_mid0    = ry0 + int((ry1 - ry0) * 0.20)
    ry_mid1    = ry0 + int((ry1 - ry0) * 0.80)
    h_profile  = binary[ry_mid0:ry_mid1, :].astype(float).mean(axis=0)
    col_zones  = _find_zones(h_profile, cols)

    all_prof  = binary[:, :].astype(float).mean(axis=1)
    side_prof = binary[ry0:ry1, :].astype(float).mean(axis=0)
    top_w    = _valley_width(all_prof,  left=True)
    bottom_w = _valley_width(all_prof,  left=False)
    left_w   = _valley_width(side_prof, left=True)
    right_w  = _valley_width(side_prof, left=False)

    add_top    = top_w    >= BRIGHT_VALLEY_MIN_PX
    add_bottom = bottom_w >= BRIGHT_VALLEY_MIN_PX
    add_left   = left_w   >= BRIGHT_VALLEY_MIN_PX
    add_right  = right_w  >= BRIGHT_VALLEY_MIN_PX

    orig_pil   = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    binary_pil = Image.fromarray(binary)

    bin_cells, orig_cells = [], []
    for r, (ry_lo, ry_hi) in enumerate(row_zones):
        bin_row, orig_row = [], []
        for c, (cx_lo, cx_hi) in enumerate(col_zones):
            crl, crh = ry_lo, ry_hi
            ccl, cch = cx_lo, cx_hi
            if add_top    and r == 0:        crl = max(0, crl - 1 - BRIGHT_CELL_V_PAD_TOP)
            if add_bottom and r == rows - 1: crh = min(H, crh + 1 + BRIGHT_CELL_V_PAD_BOTTOM)
            if add_left   and c == 0:        ccl = max(0, ccl - 1)
            if add_right  and c == cols - 1: cch = min(W, cch + 1)
            ccl = max(0, ccl - BRIGHT_CELL_H_PAD)
            cch = min(W, cch + BRIGHT_CELL_H_PAD)
            box = (ccl, crl, cch, crh)
            bin_row.append(binary_pil.crop(box))
            orig_row.append(orig_pil.crop(box))
        bin_cells.append(bin_row)
        orig_cells.append(orig_row)

    bin_stitched  = _stitch_grid(bin_cells,  rows, cols)
    orig_stitched = _stitch_grid(orig_cells, rows, cols)

    pil_save(f"{out_dir}/7_binary_stitched.png", bin_stitched)
    pil_save(f"{out_dir}/8_original_stitched.png", orig_stitched)
    log("Step 7 done – Cells extracted & stitched")

    # Return as BGR numpy array
    return cv2.cvtColor(np.array(orig_stitched), cv2.COLOR_RGB2BGR)


def _find_zones(profile: np.ndarray, n_cells: int,
                min_cell_size: int = 30) -> list:
    DROP_RATIO = 0.20
    SMOOTH_WIN = 11
    LOCAL_WIN  = 80
    n = len(profile)
    kernel   = np.ones(SMOOTH_WIN) / SMOOTH_WIN
    smoothed = np.convolve(profile.astype(float), kernel, mode="same")
    local_peak = np.empty(n, dtype=float)
    for i in range(n):
        lo = max(0, i - LOCAL_WIN); hi = min(n, i + LOCAL_WIN + 1)
        local_peak[i] = smoothed[lo:hi].max()
    dark_mask = smoothed < local_peak * (1.0 - DROP_RATIO)
    zones, start = [], None
    for i, is_dark in enumerate(dark_mask):
        if not is_dark and start is None:
            start = i
        elif is_dark and start is not None:
            if i - start >= min_cell_size:
                zones.append((start, i))
            start = None
    if start is not None and n - start >= min_cell_size:
        zones.append((start, n))
    if len(zones) < n_cells:
        raise ValueError(f"Expected {n_cells} zones, found {len(zones)}")
    if len(zones) > n_cells:
        zones = sorted(zones, key=lambda z: z[1]-z[0], reverse=True)[:n_cells]
        zones = sorted(zones, key=lambda z: z[0])
    return zones


def _valley_width(profile: np.ndarray, left: bool,
                  dark_threshold: float = 30.0) -> int:
    seq = profile if left else list(reversed(profile))
    w = 0
    for v in seq:
        if v < dark_threshold:
            w += 1
        else:
            break
    return w


def _stitch_grid(cells: list, rows: int, cols: int) -> Image.Image:
    col_widths  = [max(cells[r][c].width  for r in range(rows)) for c in range(cols)]
    row_heights = [max(cells[r][c].height for c in range(cols)) for r in range(rows)]
    out_w = sum(col_widths)  + BRIGHT_MARGIN * (cols + 1)
    out_h = sum(row_heights) + BRIGHT_MARGIN * (rows + 1)
    canvas = Image.new("RGB", (out_w, out_h), (0, 0, 0))
    y = BRIGHT_MARGIN
    for r in range(rows):
        x = BRIGHT_MARGIN
        for c in range(cols):
            canvas.paste(cells[r][c], (x, y))
            x += col_widths[c] + BRIGHT_MARGIN
        y += row_heights[r] + BRIGHT_MARGIN
    return canvas


# ══════════════════════════════════════════════════════════════════════════════
#  PROCESS ONE IMAGE
# ══════════════════════════════════════════════════════════════════════════════

def process_image(img_path: str, out_dir: str,
                  rows: int, cols: int) -> bool:
    """
    Full pipeline for a single image.
    Returns True on success, False on error.
    """
    print(f"\n[IMG] {img_path}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Load
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"  [ERROR] Cannot read image: {img_path}")
        return False

    # Step 0 – save original
    save(f"{out_dir}/0_original.png", img_bgr)

    # Step 1 – undistort
    try:
        undist = undistort(img_bgr)
    except Exception as e:
        log(f"[WARN] Undistortion failed ({e}) – using original.")
        undist = img_bgr
    save(f"{out_dir}/1_undistorted.png", undist)

    # Step 2 – OCR override + panel crop
    ocr_r, ocr_c = _ocr_grid(undist)
    final_rows = ocr_r or rows
    final_cols = ocr_c or cols
    cropped = crop_panel(undist, final_rows, final_cols)
    save(f"{out_dir}/2_cropped.png", cropped)
    log(f"Step 2 done – Cropped to {final_rows}×{final_cols} panel")

    # Steps 3-8 – pipeline
    try:
        pipeline(cropped, out_dir, final_rows, final_cols)
    except Exception as e:
        import traceback
        log(f"[ERROR] Pipeline failed: {e}")
        traceback.print_exc()
        return False

    return True


# ══════════════════════════════════════════════════════════════════════════════
#  FINAL STITCHING  (16-image composite)
# ══════════════════════════════════════════════════════════════════════════════

def build_final_stitched(img_list: list,
                          min_pixel_value: float,
                          max_pixel_value: float) -> np.ndarray:
    sub_h = img_list[0].shape[0]
    sub_w = img_list[0].shape[1]

    gap_first  = INTER_CELL_GAP * (MID_SUB_SEGMENT_FIRST - 1)
    gap_second = INTER_CELL_GAP * (NUM_SUB_SEGMENT - MID_SUB_SEGMENT_FIRST - 1)
    total_h    = sub_h * NUM_SUB_SEGMENT + gap_first + gap_second + SEGMENT_GAP
    total_w    = sub_w * NUM_SEGMENT + COLUMN_GAP
    final      = np.zeros((total_h, total_w), dtype=np.uint8)

    y_positions = []
    y = 0
    for i in range(MID_SUB_SEGMENT_FIRST):
        y_positions.append(y)
        y += sub_h + INTER_CELL_GAP
    y -= INTER_CELL_GAP
    y += SEGMENT_GAP
    for i in range(MID_SUB_SEGMENT_FIRST, NUM_SUB_SEGMENT):
        y_positions.append(y)
        y += sub_h + INTER_CELL_GAP

    for count, img in enumerate(img_list):
        img_norm = (img - min_pixel_value) / (max_pixel_value - min_pixel_value + 1e-6)
        img_norm = np.clip(img_norm * 255, 0, 255).astype(np.uint8)
        img_norm = np.flip(img_norm, axis=0)

        if count < NUM_SUB_SEGMENT:
            y_start = y_positions[count]
            final[y_start:y_start + sub_h, 0:sub_w] = img_norm
        else:
            col_index = count - NUM_SUB_SEGMENT
            y_start   = y_positions[col_index]
            x_start   = sub_w + COLUMN_GAP
            final[y_start:y_start + sub_h, x_start:x_start + sub_w] = img_norm

    return final


def build_and_save_final(folder_out: str) -> bool:
    print(f"\n[FINAL] Building composite image …")
    img_list: list = []
    missing:  list = []
    ref_shape = None

    for j in range(1, NUM_IMAGES + 1):
        stitched_path = os.path.join(folder_out, str(j), "8_original_stitched.png")
        if not os.path.exists(stitched_path):
            print(f"  [WARN] Missing {stitched_path} – will use blank.")
            missing.append(j)
            img_list.append(None)
            continue
        img_bgr = cv2.imread(stitched_path)
        if img_bgr is None:
            print(f"  [WARN] Cannot read {stitched_path} – will use blank.")
            missing.append(j)
            img_list.append(None)
            continue
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        if ref_shape is None:
            ref_shape = gray.shape
        img_list.append(gray)
        print(f"  [OK] Loaded image {j}/{NUM_IMAGES}  shape={gray.shape}")

    if ref_shape is None:
        print("  [ERROR] No stitched images found at all – skipping final build.")
        return False

    # Fill blanks
    blank = np.zeros(ref_shape, dtype=np.uint8)
    resized = []
    for img in img_list:
        if img is None:
            resized.append(blank)
        elif img.shape != ref_shape:
            resized.append(cv2.resize(img, (ref_shape[1], ref_shape[0])))
        else:
            resized.append(img)

    all_pixels      = np.concatenate([im.ravel() for im in resized])
    min_pv          = float(all_pixels.min())
    max_pv          = float(all_pixels.max())
    print(f"  [INFO] Pixel range [{min_pv:.1f}, {max_pv:.1f}]  "
          f"missing: {missing if missing else 'none'}")

    final_arr  = build_final_stitched(resized, min_pv, max_pv)
    save_path  = os.path.join(folder_out, "final_image.png")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(save_path, final_arr)
    print(f"  [FINAL] Saved → {save_path}")
    return True


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Process a folder of 1-16.jpeg images through the full pipeline.")
    parser.add_argument("folder", help="Input folder containing 1.jpeg … 16.jpeg")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS,
                        help=f"Grid rows (default {DEFAULT_ROWS})")
    parser.add_argument("--cols", type=int, default=DEFAULT_COLS,
                        help=f"Grid cols (default {DEFAULT_COLS})")
    args = parser.parse_args()

    folder = Path(args.folder).resolve()
    if not folder.is_dir():
        print(f"[ERROR] Not a directory: {folder}")
        sys.exit(1)

    folder_name = folder.name
    folder_out  = os.path.join("Output2", folder_name)

    print(f"\n{'='*60}")
    print(f"  Input folder : {folder}")
    print(f"  Output root  : {os.path.abspath(folder_out)}")
    print(f"  Grid         : {args.rows}×{args.cols}")
    print(f"{'='*60}\n")

    success_count = 0
    for j in range(1, NUM_IMAGES + 1):
        img_path = None
        for ext in (".jpeg", ".jpg", ".JPEG", ".JPG"):
            candidate = folder / f"{j}{ext}"
            if candidate.exists():
                img_path = str(candidate)
                break

        if img_path is None:
            print(f"\n[SKIP] Image {j} not found in {folder}")
            continue

        out_dir = os.path.join(folder_out, str(j))
        ok = process_image(img_path, out_dir, args.rows, args.cols)
        if ok:
            success_count += 1

    print(f"\n{'='*60}")
    print(f"  Processed {success_count}/{NUM_IMAGES} images successfully.")
    print(f"{'='*60}")

    build_and_save_final(folder_out)

    print(f"\n✅ All done. Output at: {os.path.abspath(folder_out)}")


if __name__ == "__main__":
    main()