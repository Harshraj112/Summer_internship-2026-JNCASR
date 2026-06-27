#!/usr/bin/env python3
"""
pipeline_api.py  –  Importable image-processing pipeline
=========================================================

Usage (from another file):
    from pipeline_api import process_folder

    result = process_folder(
        folder_path = "/path/to/images",   # contains 1.jpeg … 16.jpeg
        rows        = 3,
        cols        = 3,
    )
    # result["final_image"]        → np.ndarray  – composite of all 16 images
    # result["final_image_path"]   → str          – Output/<folder>/final_image.jpeg
    # result["stitched_paths"]     → list[str]    – Output/<folder>/1/final_image.jpeg … 16/final_image.jpeg
    # result["cropped_cell_paths"] → list[list]   – Output/<folder>/1/cropped_cell1.jpeg … etc.
    # result["elapsed_sec"]        → float
    # result["success_count"]      → int

CLI:
    python pipeline_api.py /path/to/folder --rows 3 --cols 3
"""

from __future__ import annotations

import argparse
import os
import sys
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

# ══════════════════════════════════════════════════════════════════════════════
#  ❶  ALL TUNEABLE CONSTANTS  –  edit here if something goes wrong
# ══════════════════════════════════════════════════════════════════════════════

# ── I/O ───────────────────────────────────────────────────────────────────────
NUM_IMAGES          = 16        # images expected per folder  (1.jpeg … 16.jpeg)
OUTPUT_ROOT         = "Output"  # top-level output directory
CELL_JPEG_QUALITY   = 92        # JPEG quality for cropped cells + stitched (0-100)
FINAL_JPEG_QUALITY  = 92        # JPEG quality for the 16-image composite

# ── Default grid ─────────────────────────────────────────────────────────────
DEFAULT_ROWS = 3
DEFAULT_COLS = 3

# ── Parallelism ───────────────────────────────────────────────────────────────
MAX_WORKERS = min(NUM_IMAGES, os.cpu_count() or 4)
# Set to 1 to disable threading (easier to debug)

# ── Camera intrinsics (fisheye undistortion) ──────────────────────────────────
K = np.array([
    [1184.8667983506784, 0.0,                939.9321230074511],
    [0.0,                1182.0885688815326, 604.8672689311147],
    [0.0,                0.0,                1.0],
], dtype=np.float64)

DIST = np.array([
    [ 0.20332565713132572],
    [ 0.042418077818231245],
    [ 0.5625598381701257],
    [-0.6590457476513796],
], dtype=np.float64)

UNDISTORT_BALANCE = 0.6         # 0 = no black border, 1 = keep full FOV

# ── Crop-ratio table  (rows, cols) → (left_r, right_r, top_r, bottom_r) ──────
CROP_RATIOS: dict[tuple[int, int], tuple[float, float, float, float]] = {
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

# ── Preprocessing ─────────────────────────────────────────────────────────────
DENOISE_KERNEL       = (5, 5)   # GaussianBlur kernel for denoising
BG_SIGMA             = 25       # sigma for background/illumination estimation
CLAHE_CLIP_LIMIT     = 2.0
CLAHE_TILE_GRID      = (8, 8)

# ── Binarisation ──────────────────────────────────────────────────────────────
BINARIZE_BLUR_KERNEL = (5, 5)

# ── Morphology ────────────────────────────────────────────────────────────────
MORPH_KERNEL_SIZE        = (3, 3)
MORPH_CLOSE_ITERATIONS   = 2
MORPH_OPEN_ITERATIONS    = 1

# ── Zone detection ────────────────────────────────────────────────────────────
ZONE_H_BAND_LEFT   = 0.20   # fraction of width used for the vertical profile
ZONE_H_BAND_RIGHT  = 0.80
ZONE_MID_TOP       = 0.20   # fraction of row height used for horizontal profile
ZONE_MID_BOTTOM    = 0.80
ZONE_DROP_RATIO    = 0.20   # darkness threshold relative to local peak
ZONE_SMOOTH_WIN    = 11     # moving-average window
ZONE_LOCAL_WIN     = 80     # half-window for local-max computation
ZONE_MIN_CELL_PX   = 30     # minimum cell size in pixels

# ── Cell padding ──────────────────────────────────────────────────────────────
BRIGHT_MARGIN            = 2
BRIGHT_VALLEY_MIN_PX     = 20
BRIGHT_CELL_H_PAD        = 7
BRIGHT_CELL_V_PAD_TOP    = 12
BRIGHT_CELL_V_PAD_BOTTOM = 13
VALLEY_DARK_THRESHOLD    = 30.0   # pixel value below which an edge is "dark"

# ── Final 16-image composite layout ──────────────────────────────────────────
INTER_CELL_GAP        = 2
SEGMENT_GAP           = 15
COLUMN_GAP            = 1
NUM_SUB_SEGMENT       = 8
NUM_SEGMENT           = 2
MID_SUB_SEGMENT_FIRST = 4   # rows that go in the first column-segment


# ══════════════════════════════════════════════════════════════════════════════
#  ❷  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _mkdir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _pil_save_jpeg(path: str, img: Image.Image, quality: int) -> None:
    _mkdir(str(Path(path).parent))
    img.save(path, "JPEG", quality=quality)


# ══════════════════════════════════════════════════════════════════════════════
#  ❸  FISHEYE UNDISTORTION  (remap cached per resolution)
# ══════════════════════════════════════════════════════════════════════════════

_remap_cache: dict[tuple[int, int], tuple] = {}


def _get_remap(h: int, w: int) -> tuple:
    key = (h, w)
    if key not in _remap_cache:
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, DIST, (w, h), np.eye(3), balance=UNDISTORT_BALANCE)
        m1, m2 = cv2.fisheye.initUndistortRectifyMap(
            K, DIST, np.eye(3), new_K, (w, h), cv2.CV_16SC2)
        _remap_cache[key] = (m1, m2)
    return _remap_cache[key]


def _undistort(img: np.ndarray) -> np.ndarray:
    m1, m2 = _get_remap(*img.shape[:2])
    return cv2.remap(img, m1, m2,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT)


# ══════════════════════════════════════════════════════════════════════════════
#  ❹  PANEL CROP
# ══════════════════════════════════════════════════════════════════════════════

def _ocr_grid(img: np.ndarray):
    try:
        import pytesseract
        h      = img.shape[0]
        text   = pytesseract.image_to_string(img[:int(h * 0.12), :])
        r = re.search(r'rows?\s*[=:]\s*(\d+)', text, re.IGNORECASE)
        c = re.search(r'col(?:umns?)?\s*[=:]\s*(\d+)', text, re.IGNORECASE)
        if r and c:
            return int(r.group(1)), int(c.group(1))
    except ImportError:
        pass
    return None, None


def _crop_panel(img: np.ndarray, rows: int, cols: int) -> np.ndarray:
    rows = max(1, min(4, rows))
    cols = max(1, min(4, cols))
    lr, rr, tr, br = CROP_RATIOS[(rows, cols)]
    h, w = img.shape[:2]
    return img[int(h * tr):int(h * br), int(w * lr):int(w * rr)]


# ══════════════════════════════════════════════════════════════════════════════
#  ❺  ZONE / CELL DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def _find_zones(profile: np.ndarray, n_cells: int) -> list[tuple[int, int]]:
    n        = len(profile)
    kernel   = np.ones(ZONE_SMOOTH_WIN) / ZONE_SMOOTH_WIN
    smoothed = np.convolve(profile.astype(float), kernel, mode="same")

    local_peak = np.empty(n, dtype=float)
    for i in range(n):
        lo = max(0, i - ZONE_LOCAL_WIN)
        hi = min(n, i + ZONE_LOCAL_WIN + 1)
        local_peak[i] = smoothed[lo:hi].max()

    dark = smoothed < local_peak * (1.0 - ZONE_DROP_RATIO)
    zones, start = [], None
    for i, is_dark in enumerate(dark):
        if not is_dark and start is None:
            start = i
        elif is_dark and start is not None:
            if i - start >= ZONE_MIN_CELL_PX:
                zones.append((start, i))
            start = None
    if start is not None and n - start >= ZONE_MIN_CELL_PX:
        zones.append((start, n))

    if len(zones) < n_cells:
        raise ValueError(f"Expected {n_cells} zones, found {len(zones)}")
    if len(zones) > n_cells:
        zones = sorted(zones, key=lambda z: z[1] - z[0], reverse=True)[:n_cells]
        zones = sorted(zones, key=lambda z: z[0])
    return zones


def _valley_width(profile: np.ndarray, left: bool) -> int:
    seq = profile if left else profile[::-1]
    w = 0
    for v in seq:
        if v < VALLEY_DARK_THRESHOLD:
            w += 1
        else:
            break
    return w


def _stitch_grid(cells: list[list[Image.Image]], rows: int, cols: int) -> Image.Image:
    col_w  = [max(cells[r][c].width  for r in range(rows)) for c in range(cols)]
    row_h  = [max(cells[r][c].height for c in range(cols)) for r in range(rows)]
    out_w  = sum(col_w) + BRIGHT_MARGIN * (cols + 1)
    out_h  = sum(row_h) + BRIGHT_MARGIN * (rows + 1)
    canvas = Image.new("RGB", (out_w, out_h), (0, 0, 0))
    y = BRIGHT_MARGIN
    for r in range(rows):
        x = BRIGHT_MARGIN
        for c in range(cols):
            canvas.paste(cells[r][c], (x, y))
            x += col_w[c] + BRIGHT_MARGIN
        y += row_h[r] + BRIGHT_MARGIN
    return canvas


# ══════════════════════════════════════════════════════════════════════════════
#  ❻  CORE PIPELINE  (steps 3–8, returns cells + stitched)
# ══════════════════════════════════════════════════════════════════════════════

def _run_pipeline(
    img_bgr: np.ndarray,
    rows: int,
    cols: int,
) -> tuple[Image.Image, list[list[Image.Image]]]:
    """
    Process one cropped panel image.

    Returns
    -------
    stitched_pil  : PIL Image – full grid stitched (the "final_image" for this frame)
    orig_cells    : 2-D list [row][col] of PIL cell images  (for cropped_cell saves)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape

    # Step 3 – Denoise / illumination correction / CLAHE
    denoised     = cv2.GaussianBlur(gray, DENOISE_KERNEL, 0)
    background   = cv2.GaussianBlur(denoised, (0, 0), BG_SIGMA)
    normalized   = cv2.divide(denoised, background, scale=255)
    clahe        = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT,
                                   tileGridSize=CLAHE_TILE_GRID)
    preprocessed = clahe.apply(normalized)

    # Step 4 – Binarise (Otsu)
    blurred = cv2.GaussianBlur(preprocessed, BINARIZE_BLUR_KERNEL, 0)
    _, binary = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 5 – Morphological clean-up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KERNEL_SIZE)
    morph  = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel,
                               iterations=MORPH_CLOSE_ITERATIONS)
    morph  = cv2.morphologyEx(morph,  cv2.MORPH_OPEN,  kernel,
                               iterations=MORPH_OPEN_ITERATIONS)

    # Step 6 – Zone / cell detection
    x0 = int(W * ZONE_H_BAND_LEFT);  x1 = int(W * ZONE_H_BAND_RIGHT)
    row_zones = _find_zones(morph[:, x0:x1].astype(float).mean(axis=1), rows)

    ry0, ry1  = row_zones[0][0], row_zones[-1][1]
    ry_mid0   = ry0 + int((ry1 - ry0) * ZONE_MID_TOP)
    ry_mid1   = ry0 + int((ry1 - ry0) * ZONE_MID_BOTTOM)
    col_zones = _find_zones(morph[ry_mid0:ry_mid1, :].astype(float).mean(axis=0), cols)

    all_prof  = morph.astype(float).mean(axis=1)
    side_prof = morph[ry0:ry1, :].astype(float).mean(axis=0)
    add_top    = _valley_width(all_prof,  left=True)  >= BRIGHT_VALLEY_MIN_PX
    add_bottom = _valley_width(all_prof,  left=False) >= BRIGHT_VALLEY_MIN_PX
    add_left   = _valley_width(side_prof, left=True)  >= BRIGHT_VALLEY_MIN_PX
    add_right  = _valley_width(side_prof, left=False) >= BRIGHT_VALLEY_MIN_PX

    orig_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    # Step 7 – Extract cells
    orig_cells: list[list[Image.Image]] = []
    for r, (ry_lo, ry_hi) in enumerate(row_zones):
        orig_row: list[Image.Image] = []
        for c, (cx_lo, cx_hi) in enumerate(col_zones):
            crl, crh = ry_lo, ry_hi
            ccl, cch = cx_lo, cx_hi
            if add_top    and r == 0:        crl = max(0, crl - 1 - BRIGHT_CELL_V_PAD_TOP)
            if add_bottom and r == rows - 1: crh = min(H, crh + 1 + BRIGHT_CELL_V_PAD_BOTTOM)
            if add_left   and c == 0:        ccl = max(0, ccl - 1)
            if add_right  and c == cols - 1: cch = min(W, cch + 1)
            ccl = max(0, ccl - BRIGHT_CELL_H_PAD)
            cch = min(W, cch + BRIGHT_CELL_H_PAD)
            orig_row.append(orig_pil.crop((ccl, crl, cch, crh)))
        orig_cells.append(orig_row)

    # Step 8 – Stitch
    stitched_pil = _stitch_grid(orig_cells, rows, cols)
    return stitched_pil, orig_cells


# ══════════════════════════════════════════════════════════════════════════════
#  ❼  PROCESS ONE IMAGE
#     Output/<folder>/<img_num>/final_image.jpeg          ← stitched grid
#     Output/<folder>/<img_num>/cropped_cell1.jpeg …      ← individual cells
# ══════════════════════════════════════════════════════════════════════════════

def _process_single(
    img_num: int,
    folder: Path,
    folder_out: str,
    rows: int,
    cols: int,
) -> tuple[int, Optional[np.ndarray], Optional[str], list[str]]:
    """
    Returns (img_num, stitched_bgr | None, stitched_path | None, cell_paths).
    """
    # Locate source file
    img_path = None
    for ext in (".jpeg", ".jpg", ".JPEG", ".JPG"):
        candidate = folder / f"{img_num}{ext}"
        if candidate.exists():
            img_path = str(candidate)
            break
    if img_path is None:
        print(f"  [SKIP] {img_num}.jpeg not found")
        return img_num, None, None, []

    print(f"  [START] Image {img_num}")
    t0 = time.perf_counter()

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"  [ERROR] Cannot read image {img_num}")
        return img_num, None, None, []

    # Undistort
    try:
        undist = _undistort(img_bgr)
    except Exception as e:
        print(f"  [WARN] Image {img_num}: undistortion failed ({e}) – using original.")
        undist = img_bgr

    # Crop panel
    ocr_r, ocr_c = _ocr_grid(undist)
    final_rows = ocr_r or rows
    final_cols = ocr_c or cols
    cropped = _crop_panel(undist, final_rows, final_cols)

    # Run core pipeline
    try:
        stitched_pil, orig_cells = _run_pipeline(cropped, final_rows, final_cols)
    except Exception as e:
        import traceback
        print(f"  [ERROR] Image {img_num} pipeline failed: {e}")
        traceback.print_exc()
        return img_num, None, None, []

    img_out_dir = os.path.join(folder_out, str(img_num))
    _mkdir(img_out_dir)

    # ── Save stitched grid → final_image.jpeg ────────────────────────────────
    stitched_path = os.path.join(img_out_dir, "final_image.jpeg")
    _pil_save_jpeg(stitched_path, stitched_pil, CELL_JPEG_QUALITY)

    # ── Save individual cells → cropped_cell1.jpeg, cropped_cell2.jpeg … ─────
    cell_paths: list[str] = []
    cell_idx = 1
    for row in orig_cells:
        for cell_img in row:
            cell_path = os.path.join(img_out_dir, f"cropped_cell{cell_idx}.jpeg")
            _pil_save_jpeg(cell_path, cell_img, CELL_JPEG_QUALITY)
            cell_paths.append(cell_path)
            cell_idx += 1

    stitched_bgr = cv2.cvtColor(np.array(stitched_pil), cv2.COLOR_RGB2BGR)

    elapsed = time.perf_counter() - t0
    print(f"  [DONE]  Image {img_num}  →  {final_rows}×{final_cols} grid  "
          f"({cell_idx - 1} cells)  {elapsed:.2f}s")
    return img_num, stitched_bgr, stitched_path, cell_paths


# ══════════════════════════════════════════════════════════════════════════════
#  ❽  FINAL COMPOSITE  (all 16 stitched images → one array)
# ══════════════════════════════════════════════════════════════════════════════

def _build_composite(img_list: list[Optional[np.ndarray]]) -> np.ndarray:
    ref_shape = next((im.shape[:2] for im in img_list if im is not None), None)
    if ref_shape is None:
        raise RuntimeError("No valid stitched images to composite.")

    blank = np.zeros(ref_shape, dtype=np.uint8)

    grays: list[np.ndarray] = []
    for im in img_list:
        if im is None:
            grays.append(blank)
        else:
            g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if im.ndim == 3 else im
            grays.append(cv2.resize(g, (ref_shape[1], ref_shape[0]))
                         if g.shape != ref_shape else g)

    all_px = np.concatenate([g.ravel() for g in grays])
    mn, mx = float(all_px.min()), float(all_px.max())
    print(f"  [COMPOSITE] Pixel range [{mn:.1f}, {mx:.1f}]")

    sub_h, sub_w = grays[0].shape
    gap1    = INTER_CELL_GAP * (MID_SUB_SEGMENT_FIRST - 1)
    gap2    = INTER_CELL_GAP * (NUM_SUB_SEGMENT - MID_SUB_SEGMENT_FIRST - 1)
    total_h = sub_h * NUM_SUB_SEGMENT + gap1 + gap2 + SEGMENT_GAP
    total_w = sub_w * NUM_SEGMENT + COLUMN_GAP
    canvas  = np.zeros((total_h, total_w), dtype=np.uint8)

    y_pos: list[int] = []
    y = 0
    for i in range(MID_SUB_SEGMENT_FIRST):
        y_pos.append(y);  y += sub_h + INTER_CELL_GAP
    y -= INTER_CELL_GAP;  y += SEGMENT_GAP
    for i in range(MID_SUB_SEGMENT_FIRST, NUM_SUB_SEGMENT):
        y_pos.append(y);  y += sub_h + INTER_CELL_GAP

    for idx, gray in enumerate(grays):
        norm = np.clip((gray.astype(float) - mn) / (mx - mn + 1e-6) * 255,
                       0, 255).astype(np.uint8)
        norm = np.flip(norm, axis=0)
        if idx < NUM_SUB_SEGMENT:
            ys = y_pos[idx]
            canvas[ys:ys + sub_h, 0:sub_w] = norm
        else:
            ci = idx - NUM_SUB_SEGMENT
            ys = y_pos[ci];  xs = sub_w + COLUMN_GAP
            canvas[ys:ys + sub_h, xs:xs + sub_w] = norm

    return canvas


# ══════════════════════════════════════════════════════════════════════════════
#  ❾  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def process_folder(
    folder_path: str,
    rows: int = DEFAULT_ROWS,
    cols: int = DEFAULT_COLS,
    output_root: str = OUTPUT_ROOT,
    max_workers: int = MAX_WORKERS,
) -> dict:
    """
    Run the full pipeline on a folder of 1.jpeg … 16.jpeg.

    Output layout
    -------------
    Output/<folder>/
        1/
            final_image.jpeg          ← stitched grid for image 1
            cropped_cell1.jpeg
            cropped_cell2.jpeg
            …
        2/
            final_image.jpeg
            cropped_cell1.jpeg
            …
        …
        16/
            …
        final_image.jpeg              ← composite of all 16

    Returns
    -------
    {
        "final_image"        : np.ndarray           – 16-image composite (grayscale)
        "final_image_path"   : str                  – path to composite JPEG
        "stitched_paths"     : list[str | None]     – per-image final_image.jpeg paths
        "cropped_cell_paths" : list[list[str]]      – per-image cell JPEG paths
        "elapsed_sec"        : float
        "success_count"      : int
    }
    """
    t_start = time.perf_counter()

    folder = Path(folder_path).resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Not a directory: {folder}")

    folder_out = os.path.join(output_root, folder.name)
    _mkdir(folder_out)

    print(f"\n{'='*60}")
    print(f"  Input   : {folder}")
    print(f"  Output  : {os.path.abspath(folder_out)}")
    print(f"  Grid    : {rows}×{cols}   Workers: {max_workers}")
    print(f"{'='*60}")

    # ── Parallel processing ───────────────────────────────────────────────────
    results_map: dict[int, tuple] = {}
    success_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = {
            exe.submit(_process_single, j, folder, folder_out, rows, cols): j
            for j in range(1, NUM_IMAGES + 1)
        }
        for fut in as_completed(futures):
            img_num, stitched_bgr, stitched_path, cell_paths = fut.result()
            results_map[img_num] = (stitched_bgr, stitched_path, cell_paths)
            if stitched_bgr is not None:
                success_count += 1

    # ── Ordered lists (index 0 = image 1) ────────────────────────────────────
    stitched_list:      list[Optional[np.ndarray]] = []
    stitched_paths:     list[Optional[str]]         = []
    cropped_cell_paths: list[list[str]]             = []

    for j in range(1, NUM_IMAGES + 1):
        bgr, sp, cp = results_map.get(j, (None, None, []))
        stitched_list.append(bgr)
        stitched_paths.append(sp)
        cropped_cell_paths.append(cp)

    print(f"\n  Processed {success_count}/{NUM_IMAGES} images successfully.")

    # ── Build & save composite ────────────────────────────────────────────────
    print("\n[COMPOSITE] Building final composite …")
    try:
        composite  = _build_composite(stitched_list)
        final_path = os.path.join(folder_out, "final_image.jpeg")
        cv2.imwrite(final_path, composite,
                    [cv2.IMWRITE_JPEG_QUALITY, FINAL_JPEG_QUALITY])
        print(f"  [SAVED] {final_path}")
    except RuntimeError as e:
        print(f"  [ERROR] {e}")
        composite  = np.zeros((1, 1), dtype=np.uint8)
        final_path = ""

    elapsed = time.perf_counter() - t_start
    print(f"\n{'='*60}")
    print(f"  ✅  Total execution time : {elapsed:.2f}s")
    print(f"  Output at               : {os.path.abspath(folder_out)}")
    print(f"{'='*60}\n")

    return {
        "final_image":        composite,
        "final_image_path":   final_path,
        "stitched_paths":     stitched_paths,
        "cropped_cell_paths": cropped_cell_paths,
        "elapsed_sec":        elapsed,
        "success_count":      success_count,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process a folder of 1-16.jpeg images through the full pipeline.")
    parser.add_argument("folder",
                        help="Input folder containing 1.jpeg … 16.jpeg")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS,
                        help=f"Grid rows (default {DEFAULT_ROWS})")
    parser.add_argument("--cols", type=int, default=DEFAULT_COLS,
                        help=f"Grid cols (default {DEFAULT_COLS})")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS,
                        help=f"Parallel workers (default {MAX_WORKERS})")
    args = parser.parse_args()

    result = process_folder(
        folder_path = args.folder,
        rows        = args.rows,
        cols        = args.cols,
        max_workers = args.workers,
    )
    sys.exit(0 if result["success_count"] > 0 else 1)


if __name__ == "__main__":
    main()