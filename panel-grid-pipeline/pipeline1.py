#!/usr/bin/env python3
"""
Full Image Processing Pipeline
================================
Input : folder path containing 1.jpeg … 16.jpeg  (or 12.jpeg)
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
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

NUM_IMAGES = 16   # will be auto-detected per folder (12 or 16)

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

# ── Dark / Bright classification thresholds ───────────────────────────────────
MEAN_THRESH         = 50
STD_THRESH          = 20
BRIGHT_RATIO_THRESH = 0.15

# ── Crop-ratio tables (rows, cols) → (left_r, right_r, top_r, bottom_r) ─────
CROP_RATIOS_DARK = {
    (1, 1): (0.30, 0.70, 0.35, 0.68),
    (1, 2): (0.18, 0.50, 0.23, 0.70),
    (1, 3): (0.18, 0.50, 0.15, 0.78),
    (1, 4): (0.18, 0.50, 0.03, 0.78),
    (2, 1): (0.18, 0.70, 0.30, 0.68),
    (2, 2): (0.18, 0.75, 0.15, 0.70),
    (2, 3): (0.18, 0.73, 0.15, 0.78),
    (2, 4): (0.18, 0.75, 0.03, 0.78),
    (3, 1): (0.18, 0.85, 0.30, 0.62),
    (3, 2): (0.15, 0.85, 0.11, 0.85),
    (3, 3): (0.12, 0.85, 0.16, 0.78),
    (3, 4): (0.18, 0.90, 0.17, 0.95),
    (4, 1): (0.01, 0.98, 0.38, 0.64),
    (4, 2): (0.01, 0.98, 0.13, 0.80),
    (4, 3): (0.01, 0.98, 0.15, 0.78),
    (4, 4): (0.01, 0.98, 0.05, 0.78),
}

CROP_RATIOS_BRIGHT = {
    (1, 1): (0.30, 0.70, 0.35, 0.68),
    (1, 2): (0.18, 0.50, 0.23, 0.70),
    (1, 3): (0.18, 0.50, 0.15, 0.78),
    (1, 4): (0.18, 0.50, 0.03, 0.78),
    (2, 1): (0.18, 0.70, 0.30, 0.68),
    (2, 2): (0.18, 0.75, 0.15, 0.70),
    (2, 3): (0.18, 0.73, 0.15, 0.78),
    (2, 4): (0.18, 0.75, 0.03, 0.78),
    (3, 1): (0.18, 0.85, 0.30, 0.62),
    (3, 2): (0.18, 0.85, 0.15, 0.66),
    (3, 3): (0.18, 0.85, 0.16, 0.78),
    (3, 4): (0.18, 0.90, 0.17, 0.95),
    (4, 1): (0.01, 0.98, 0.38, 0.64),
    (4, 2): (0.01, 0.98, 0.13, 0.80),
    (4, 3): (0.01, 0.98, 0.15, 0.78),
    (4, 4): (0.01, 0.98, 0.05, 0.78),
}

# ── Dark pipeline parameters ──────────────────────────────────────────────────
DARK_WANT_ROWS    = 2
DARK_WANT_COLS    = 3
DARK_INSET        = 2
DARK_EXTRA_TOP    = -2
DARK_EXTRA_BOTTOM = -2
DARK_EXTRA_LEFT   = -2
DARK_EXTRA_RIGHT  = -2
DARK_SIZE_TOL     = 0.10

# Edge detection thresholds
DARK_THRESHOLD_VERTICAL   = 90
DARK_THRESHOLD_HORIZONTAL = 70
DARK_MIN_CC_SIZE          = 200
DARK_MIN_VERTICAL_HEIGHT  = 80
DARK_PEAK_TOLERANCE       = 20

# ── Bright pipeline parameters ────────────────────────────────────────────────
BRIGHT_ROWS               = 2
BRIGHT_COLS               = 2
BRIGHT_MARGIN             = 2
BRIGHT_VALLEY_MIN_PX      = 20
BRIGHT_CELL_H_PAD         = 7
BRIGHT_CELL_V_PAD_TOP     = 12
BRIGHT_CELL_V_PAD_BOTTOM  = 13

# ── Stitching parameters ──────────────────────────────────────────────────────
INTER_CELL_GAP = 2
SEGMENT_GAP    = 15
COLUMN_GAP     = 2

# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def save(path: str, img) -> None:
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
#  STEP 0 – CLASSIFY (DARK vs BRIGHT)
# ══════════════════════════════════════════════════════════════════════════════

def classify_image(img_gray: np.ndarray) -> str:
    mean_val     = float(np.mean(img_gray))
    std_val      = float(np.std(img_gray))
    bright_ratio = float(np.sum(img_gray > 100) / img_gray.size)
    is_dark = (mean_val < MEAN_THRESH and
               std_val  < STD_THRESH  and
               bright_ratio < BRIGHT_RATIO_THRESH)
    label = "DARK" if is_dark else "BRIGHT"
    log(f"Classification → {label}  "
        f"(mean={mean_val:.1f}, std={std_val:.1f}, bright_ratio={bright_ratio:.3f})")
    return label


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


def crop_panel(img_bgr: np.ndarray, rows: int, cols: int,
               crop_table: dict) -> np.ndarray:
    rows = max(1, min(4, rows))
    cols = max(1, min(4, cols))
    left_r, right_r, top_r, bottom_r = crop_table[(rows, cols)]
    h, w = img_bgr.shape[:2]
    return img_bgr[int(h * top_r):int(h * bottom_r),
                   int(w * left_r):int(w * right_r)]


# ══════════════════════════════════════════════════════════════════════════════
#  DARK PIPELINE — STAGE A: Edge Filtering (from Script 1)
#  Gaussian → Sobel → Threshold → Connected Components → Size + V-peak filter
# ══════════════════════════════════════════════════════════════════════════════

def _dark_edge_filter(sharpened_gray: np.ndarray, out_dir: str) -> np.ndarray:
    """
    Produces a clean edge mask from a sharpened grayscale image.
    Replicates the full Script-1 pipeline:
      Gaussian → Sobel(V+H) → separate thresholds → CCA → size filter
      → remove short vertical components unless near a projection peak.
    """
    # Gaussian smoothing
    blur = cv2.GaussianBlur(sharpened_gray, (0, 0), 4.0)

    # Sobel edges
    sx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)

    sx_norm = cv2.normalize(np.abs(sx), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    sy_norm = cv2.normalize(np.abs(sy), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Separate thresholds for vertical / horizontal edges
    _, edge_vertical   = cv2.threshold(sx_norm, DARK_THRESHOLD_VERTICAL,   255, cv2.THRESH_BINARY)
    _, edge_horizontal = cv2.threshold(sy_norm, DARK_THRESHOLD_HORIZONTAL, 255, cv2.THRESH_BINARY)
    edge_binary = cv2.bitwise_or(edge_vertical, edge_horizontal)

    # Connected-component size filter
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        edge_binary, connectivity=8)
    filtered = np.zeros_like(edge_binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= DARK_MIN_CC_SIZE:
            filtered[labels == i] = 255

    # Column-projection peak detection on size-filtered image
    col_proj = np.sum(filtered > 0, axis=0).astype(np.float64)
    col_proj_smooth = uniform_filter1d(col_proj, size=10)
    v_peaks, _ = find_peaks(
        col_proj_smooth,
        height=col_proj_smooth.max() * 0.10,
        distance=30
    )

    # Re-label and remove short vertical components not near any peak
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        filtered, connectivity=8)
    filtered_v = np.zeros_like(filtered)
    for i in range(1, num_labels):
        x  = stats[i, cv2.CC_STAT_LEFT]
        w  = stats[i, cv2.CC_STAT_WIDTH]
        h  = stats[i, cv2.CC_STAT_HEIGHT]
        cx = x + w // 2
        near_peak = any(abs(cx - vp) <= DARK_PEAK_TOLERANCE for vp in v_peaks)
        if h > w:   # vertical component
            if h >= DARK_MIN_VERTICAL_HEIGHT or near_peak:
                filtered_v[labels == i] = 255
        else:       # horizontal component — always keep
            filtered_v[labels == i] = 255

    save(f"{out_dir}/7a_edge_filtered.png", filtered_v)
    log(f"Edge filter done – {len(v_peaks)} column peaks, "
        f"{np.sum(filtered_v > 0)} edge pixels retained")
    return filtered_v


# ══════════════════════════════════════════════════════════════════════════════
#  DARK PIPELINE — STAGE B: Gradient-Based Grid Detection (from Script 2)
#  Sobel projections → peak detection → span filter → gap fill → unit cells
#  → draw lines/rectangles on black panel
# ══════════════════════════════════════════════════════════════════════════════

# ── Shared low-level helpers ──────────────────────────────────────────────────

def _longest_run(mask_1d: np.ndarray) -> int:
    mask_1d = np.asarray(mask_1d).astype(np.uint8)
    if mask_1d.size == 0:
        return 0
    padded  = np.concatenate([[0], mask_1d, [0]])
    changes = np.diff(padded)
    starts  = np.where(changes ==  1)[0]
    ends    = np.where(changes == -1)[0]
    if not len(starts) or not len(ends):
        return 0
    n = min(len(starts), len(ends))
    return int((ends[:n] - starts[:n]).max())


def _measure_span_grad(grad_map: np.ndarray, pos: int,
                       orientation: str = "h", band: int = 3,
                       thr_ratio: float = 0.08):
    h, w = grad_map.shape
    if orientation == "h":
        y1 = max(0, pos - band); y2 = min(h, pos + band + 1)
        strip = grad_map[y1:y2, :]
    else:
        x1 = max(0, pos - band); x2 = min(w, pos + band + 1)
        strip = grad_map[:, x1:x2]
    if strip.max() == 0:
        return 0, 0
    thresh  = strip.max() * thr_ratio
    profile = (np.max(strip, axis=0 if orientation == "h" else 1) > thresh).astype(np.uint8)
    return _longest_run(profile), int(profile.sum())


def _dominant_span(spans: np.ndarray) -> float:
    spans = np.asarray(spans, dtype=float)
    if spans.size == 0:
        return 0.0
    top_n = max(2, min(spans.size // 4, spans.size))
    return float(np.median(np.sort(spans)[-top_n:]))


def _suppress_small_near_large(peaks: np.ndarray, proj: np.ndarray,
                                radius: int = 28, ratio: float = 0.55) -> np.ndarray:
    if len(peaks) == 0:
        return peaks
    peaks   = np.array(sorted(peaks.tolist()), dtype=int)
    changed = True
    while changed:
        changed = False
        amps = proj[np.clip(peaks, 0, len(proj) - 1)]
        keep = np.ones(len(peaks), dtype=bool)
        for i in range(len(peaks)):
            if not keep[i]:
                continue
            for j in range(len(peaks)):
                if i == j or not keep[j]:
                    continue
                if abs(int(peaks[i]) - int(peaks[j])) <= radius:
                    if amps[i] < amps[j] * ratio:
                        keep[i] = False
                        changed  = True
                        break
        peaks = peaks[keep]
    return peaks


def _find_grid_peaks(proj: np.ndarray, distance: int = 30,
                     min_h_ratio: float = 0.04, prom_ratio: float = 0.02,
                     sup_radius: int = 28, sup_ratio: float = 0.55) -> np.ndarray:
    max_val = proj.max()
    if max_val == 0:
        return np.array([], dtype=int)
    peaks, _ = find_peaks(proj,
                          height=min_h_ratio * max_val,
                          distance=distance,
                          prominence=prom_ratio * max_val)
    if len(peaks) == 0:
        return peaks
    return _suppress_small_near_large(peaks, proj, radius=sup_radius, ratio=sup_ratio)


def _filter_by_span_and_spacing(peaks: np.ndarray, spans, coverages,
                                 tol: float = 0.30,
                                 span_ratio: float = 0.55) -> np.ndarray:
    peaks     = np.asarray(peaks, dtype=int)
    spans     = np.asarray(spans, dtype=float)
    coverages = np.asarray(coverages, dtype=float)
    if len(peaks) == 0:
        return peaks
    order  = np.argsort(peaks)
    peaks, spans, coverages = peaks[order], spans[order], coverages[order]
    dom = _dominant_span(spans)
    if dom > 0:
        keep  = spans >= dom * span_ratio
        peaks, spans, coverages = peaks[keep], spans[keep], coverages[keep]
    if len(peaks) <= 2:
        return peaks
    diffs = np.diff(peaks); diffs = diffs[diffs > 2]
    if len(diffs) == 0:
        return peaks
    step = float(np.median(diffs))
    if step <= 0:
        return peaks
    good = np.ones(len(peaks), dtype=bool)
    span_median = float(np.median(spans)) if len(spans) else 0.0
    for i in range(1, len(peaks) - 1):
        left  = peaks[i]     - peaks[i - 1]
        right = peaks[i + 1] - peaks[i]
        if (abs(left  - step) > tol * step and
            abs(right - step) > tol * step and
                spans[i] < 0.92 * max(spans.max(), span_median)):
            good[i] = False
    return peaks[good]


def _final_uniform_cleanup(lines: np.ndarray) -> np.ndarray:
    lines = np.array(sorted(set(map(int, lines))))
    if len(lines) <= 2:
        return lines
    diffs = np.diff(lines); diffs = diffs[diffs > 2]
    if len(diffs) == 0:
        return lines
    step = float(np.median(diffs))
    keep = [True]
    for i in range(1, len(lines) - 1):
        if (abs(lines[i] - lines[i - 1] - step) > 0.35 * step and
                abs(lines[i + 1] - lines[i]     - step) > 0.35 * step):
            keep.append(False)
        else:
            keep.append(True)
    keep.append(True)
    return lines[np.array(keep, dtype=bool)]


def _snap_to_proj(pos: int, proj: np.ndarray, img_size: int, radius: int) -> int:
    lo = max(0, pos - radius); hi = min(img_size, pos + radius + 1)
    if lo >= hi:
        return max(0, min(img_size - 1, pos))
    return lo + int(np.argmax(proj[lo:hi]))


def _fill_grid_gaps(peaks: np.ndarray, proj: np.ndarray, img_size: int,
                    fill_tol: float = 0.30, snap_radius: int = 8) -> np.ndarray:
    peaks = np.array(sorted(set(map(int, peaks))))
    if len(peaks) < 2:
        return peaks
    diffs = np.diff(peaks)
    step  = float(np.median(diffs[diffs > 2])) if np.any(diffs > 2) else 0.0
    if step <= 0:
        return peaks
    filled = list(peaks)
    for i in range(len(peaks) - 1):
        gap    = peaks[i + 1] - peaks[i]
        n_miss = round(gap / step) - 1
        if n_miss < 1:
            continue
        if abs(gap - (n_miss + 1) * step) > fill_tol * step:
            continue
        for k in range(1, n_miss + 1):
            cand = int(round(peaks[i] + k * step))
            filled.append(_snap_to_proj(cand, proj, img_size, snap_radius))
    for ext in [int(round(peaks[0] - step)), int(round(peaks[-1] + step))]:
        if 0 <= ext < img_size:
            snapped = _snap_to_proj(ext, proj, img_size, snap_radius)
            if proj[snapped] > proj.max() * 0.03:
                filled.append(snapped)
    return np.array(sorted(set(filled)))


def _dark_gradient_grid(filtered_v_bgr: np.ndarray,
                        out_dir: str) -> np.ndarray:
    """
    Replicates Script 2: gradient-based grid detection on the edge-filtered image.
    Returns a black panel (BGR) with coloured grid lines drawn on it.
    """
    gray  = cv2.cvtColor(filtered_v_bgr, cv2.COLOR_BGR2GRAY)
    H, W  = gray.shape

    sx    = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy    = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    abs_sx = np.abs(sx)
    abs_sy = np.abs(sy)

    row_proj = uniform_filter1d(np.sum(abs_sy, axis=1).astype(float), size=7)
    col_proj = uniform_filter1d(np.sum(abs_sx, axis=0).astype(float), size=5)

    # ── Horizontal lines ──────────────────────────────────────────────────────
    row_ys_raw = _find_grid_peaks(row_proj,
                                  distance=40, min_h_ratio=0.0, prom_ratio=0.00,
                                  sup_radius=30, sup_ratio=0.55)
    if len(row_ys_raw):
        spans_r, covs_r = zip(*[_measure_span_grad(abs_sy, y, "h") for y in row_ys_raw])
        row_ys = _filter_by_span_and_spacing(
            row_ys_raw, list(spans_r), list(covs_r), tol=0.30, span_ratio=0.45)
    else:
        row_ys = row_ys_raw

    # ── Vertical lines ────────────────────────────────────────────────────────
    col_xs_raw = _find_grid_peaks(col_proj,
                                  distance=250, min_h_ratio=0.04, prom_ratio=0.02,
                                  sup_radius=30, sup_ratio=0.2)
    if len(col_xs_raw):
        spans_c, covs_c = zip(*[_measure_span_grad(abs_sx, x, "v") for x in col_xs_raw])
        col_xs = _filter_by_span_and_spacing(
            col_xs_raw, list(spans_c), list(covs_c), tol=0.50, span_ratio=0.45)
    else:
        col_xs = col_xs_raw

    row_ys = _final_uniform_cleanup(row_ys)
    col_xs = _final_uniform_cleanup(col_xs)

    log(f"Gradient grid (before gap-fill) → rows: {len(row_ys)}  cols: {len(col_xs)}")

    row_ys = _fill_grid_gaps(row_ys, row_proj, H, fill_tol=0.35, snap_radius=10)
    col_xs = _fill_grid_gaps(col_xs, col_proj, W, fill_tol=0.35, snap_radius=10)

    log(f"Gradient grid (after  gap-fill) → rows: {len(row_ys)}  cols: {len(col_xs)}")

    # ── Build unit cells ──────────────────────────────────────────────────────
    sorted_rows = sorted(map(int, row_ys))
    sorted_cols = sorted(map(int, col_xs))

    unit_cells = []
    for i in range(len(sorted_rows) - 1):
        for j in range(len(sorted_cols) - 1):
            y1, y2 = sorted_rows[i], sorted_rows[i + 1]
            x1, x2 = sorted_cols[j], sorted_cols[j + 1]
            wr, hr = x2 - x1, y2 - y1
            if wr > 10 and hr > 10:
                unit_cells.append((x1, y1, wr, hr))

    SIZE_TOL = 0.35
    if unit_cells:
        widths  = np.array([c[2] for c in unit_cells], dtype=float)
        heights = np.array([c[3] for c in unit_cells], dtype=float)
        med_w   = float(np.median(widths))
        med_h   = float(np.median(heights))
        rectangles = [
            cell for cell, wr, hr in zip(unit_cells, widths, heights)
            if (abs(wr - med_w) <= SIZE_TOL * med_w and
                abs(hr - med_h) <= SIZE_TOL * med_h)
        ]
    else:
        rectangles = []
        med_w = med_h = 0

    log(f"Gradient grid – {len(rectangles)} rectangles  (med {med_w:.0f}×{med_h:.0f})")

    # ── Draw on black panel ───────────────────────────────────────────────────
    colors = [(0,255,0),(0,200,255),(255,100,0),(200,0,255),(0,165,255),(180,0,255)]
    black_panel = np.zeros_like(filtered_v_bgr)

    used_ys = sorted({int(y) for (x, y, wr, hr) in rectangles for y in [y, y + hr]})
    used_xs = sorted({int(x) for (x, y, wr, hr) in rectangles for x in [x, x + wr]})

    if used_xs and used_ys:
        x_min, x_max = min(used_xs), max(used_xs)
        y_min, y_max = min(used_ys), max(used_ys)
        for y in used_ys:
            cv2.line(black_panel, (x_min, int(y)), (x_max, int(y)), (255, 255, 255), 2)
        for x in used_xs:
            cv2.line(black_panel, (int(x), y_min), (int(x), y_max), (255, 255, 255), 2)
    for i, (x, y, wr, hr) in enumerate(rectangles):
        c = colors[i % len(colors)]
        cv2.rectangle(black_panel, (x, y), (x + wr, y + hr), c, 2)
        cv2.putText(black_panel, f"{wr}x{hr}", (x + 5, y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)

    save(f"{out_dir}/11_lines_on_black.png", black_panel)
    return black_panel


# ══════════════════════════════════════════════════════════════════════════════
#  DARK PIPELINE — STAGE C: Intersection + Cell Extraction (from Script 3)
# ══════════════════════════════════════════════════════════════════════════════

def _get_intersections(black_panel: np.ndarray) -> list:
    H, W = black_panel.shape[:2]
    hsv  = cv2.cvtColor(black_panel, cv2.COLOR_BGR2HSV)
    color_ranges = [
        cv2.inRange(hsv, (100, 80, 60), (130, 255, 255)),
        cv2.inRange(hsv, ( 40, 80, 60), ( 90, 255, 255)),
        cv2.inRange(hsv, ( 10, 80, 60), ( 25, 255, 255)),
        cv2.inRange(hsv, (140, 80, 60), (170, 255, 255)),
        cv2.inRange(hsv, ( 85, 80, 60), (100, 255, 255)),
    ]
    mask = np.zeros((H, W), dtype=np.uint8)
    for m in color_ranges:
        mask = cv2.bitwise_or(mask, m)

    edges = cv2.Canny(mask, 30, 100)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                             threshold=60, minLineLength=50, maxLineGap=15)
    if lines is None:
        return []

    h_segs, v_segs = [], []
    for x1, y1, x2, y2 in lines[:, 0]:
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle < 20 or angle > 160:
            h_segs.append((x1, y1, x2, y2))
        elif 70 < angle < 110:
            v_segs.append((x1, y1, x2, y2))

    def cluster(segs, axis, gap=18):
        if not segs:
            return []
        coord   = [((s[1]+s[3])//2) if axis == 'h' else ((s[0]+s[2])//2)
                   for s in segs]
        order   = sorted(range(len(coord)), key=lambda i: coord[i])
        clusters, cur = [], [order[0]]
        for i in order[1:]:
            if coord[i] - coord[cur[-1]] < gap:
                cur.append(i)
            else:
                clusters.append(cur); cur = [i]
        clusters.append(cur)
        result = []
        for cl in clusters:
            pos = int(np.median([coord[i] for i in cl]))
            if axis == 'h':
                xs = [segs[i][0] for i in cl] + [segs[i][2] for i in cl]
                result.append((min(xs), pos, max(xs), pos))
            else:
                ys = [segs[i][1] for i in cl] + [segs[i][3] for i in cl]
                result.append((pos, min(ys), pos, max(ys)))
        return result

    h_rep = cluster(h_segs, 'h')
    v_rep = cluster(v_segs, 'v')
    points = [((v[0]+v[2])//2, (h[1]+h[3])//2) for h in h_rep for v in v_rep]

    log(f"  HoughLinesP – H lines: {len(h_rep)}  V lines: {len(v_rep)}"
        f"  → {len(points)} intersections")
    return points


def _build_all_cells(pts: list):
    xs = sorted(set(p[0] for p in pts))
    ys = sorted(set(p[1] for p in pts))
    cells = []
    for r in range(len(ys) - 1):
        for c in range(len(xs) - 1):
            x1, y1 = xs[c],   ys[r]
            x2, y2 = xs[c+1], ys[r+1]
            cells.append({"r": r, "c": c,
                           "x1": x1, "y1": y1,
                           "x2": x2, "y2": y2,
                           "w": x2-x1, "h": y2-y1})
    return cells, xs, ys


def _select_grid_block(all_cells: list, xs: list, ys: list,
                       want_rows: int, want_cols: int, tol: float) -> dict:
    cell_map   = {(c["r"], c["c"]): c for c in all_cells}
    total_rows = len(ys) - 1
    total_cols = len(xs) - 1

    def cells_uniform(cells):
        if not cells:
            return False
        ws = np.array([c["w"] for c in cells], float)
        hs = np.array([c["h"] for c in cells], float)
        mw, mh = np.median(ws), np.median(hs)
        return (np.all(np.abs(ws - mw) / (mw + 1e-6) <= tol) and
                np.all(np.abs(hs - mh) / (mh + 1e-6) <= tol))

    # Try top-left want_rows × want_cols first
    top_cells = [c for c in all_cells if c["r"] < want_rows and c["c"] < want_cols]
    if cells_uniform(top_cells) and top_cells:
        sel = top_cells
        log(f"Uniform grid — selected top-left {want_rows}×{want_cols} block.")
    else:
        # Find largest uniform rectangular block
        best_area, best_cells = 0, []
        for r0 in range(total_rows):
            for c0 in range(total_cols):
                anchor = cell_map.get((r0, c0))
                if not anchor:
                    continue
                rw, rh = anchor["w"], anchor["h"]
                max_cols = 0
                for dc in range(total_cols - c0):
                    cell = cell_map.get((r0, c0 + dc))
                    if (not cell or
                            abs(cell["w"] - rw) / (rw + 1e-6) > tol or
                            abs(cell["h"] - rh) / (rh + 1e-6) > tol):
                        break
                    max_cols = dc + 1
                if not max_cols:
                    continue
                max_rows = 0
                for dr in range(total_rows - r0):
                    ok = all(
                        cell_map.get((r0+dr, c0+dc)) is not None and
                        abs(cell_map[(r0+dr, c0+dc)]["w"] - rw) / (rw + 1e-6) <= tol and
                        abs(cell_map[(r0+dr, c0+dc)]["h"] - rh) / (rh + 1e-6) <= tol
                        for dc in range(max_cols)
                    )
                    if not ok:
                        break
                    max_rows = dr + 1
                area = max_rows * max_cols
                if area > best_area:
                    best_area  = area
                    best_cells = [cell_map[(r0+dr, c0+dc)]
                                  for dr in range(max_rows)
                                  for dc in range(max_cols)]
        sel = best_cells if best_cells else all_cells
        log(f"Non-uniform grid — largest uniform block: {len(sel)} cells.")

    sel_xs = [v for c in sel for v in (c["x1"], c["x2"])]
    sel_ys = [v for c in sel for v in (c["y1"], c["y2"])]
    return {"tl": (min(sel_xs), min(sel_ys)),
            "br": (max(sel_xs), max(sel_ys))}


# ══════════════════════════════════════════════════════════════════════════════
#  DARK PIPELINE  (combined Steps 3-8)
# ══════════════════════════════════════════════════════════════════════════════

def dark_pipeline(img_bgr: np.ndarray, out_dir: str) -> np.ndarray:
    log("Running DARK pipeline …")

    # ── Step 3: CLAHE + Gamma + Blur + Sharpen ───────────────────────────────
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    clahe     = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(gray)
    save(f"{out_dir}/3_clahe.png", img_clahe)

    lut       = np.array([((i / 255) ** 0.4) * 255 for i in range(256)], dtype=np.uint8)
    img_gamma = cv2.LUT(img_clahe, lut)
    save(f"{out_dir}/4_gamma.png", img_gamma)

    blur      = cv2.GaussianBlur(img_gamma, (0, 0), sigmaX=2)
    sharpened = cv2.addWeighted(img_gamma, 1.8, blur, -0.8, 0)
    save(f"{out_dir}/5_blur.png", blur)
    save(f"{out_dir}/6_sharpened.png", sharpened)
    log("Step 3 done – CLAHE / Gamma / Blur / Sharpen")

    # ── Step 4 (Script-1): Sobel edge filter → size/V-peak filtered mask ─────
    filtered_v_gray = _dark_edge_filter(sharpened, out_dir)
    filtered_v_bgr  = cv2.cvtColor(filtered_v_gray, cv2.COLOR_GRAY2BGR)
    save(f"{out_dir}/7b_filtered_v.png", filtered_v_gray)
    log("Step 4 done – Edge / size / V-peak filter")

    # ── Step 5 (Script-2): Gradient-based grid → black panel ─────────────────
    black_panel = _dark_gradient_grid(filtered_v_bgr, out_dir)
    log("Step 5 done – Gradient grid detection → black panel")

    # ── Step 6: HoughLinesP intersections ────────────────────────────────────
    intersections = _get_intersections(black_panel)
    log(f"Step 6 done – {len(intersections)} intersections via HoughLinesP")

    if len(intersections) < 4:
        log("[WARN] Too few intersections – saving cropped image as final.")
        save(f"{out_dir}/8_original_stitched.png", img_bgr)
        return img_bgr

    # ── Step 7 (Script-3): Build cells → select uniform block ────────────────
    all_cells, xs_g, ys_g = _build_all_cells(intersections)
    log(f"Step 7 – Grid: {len(ys_g)-1} rows × {len(xs_g)-1} cols  "
        f"({len(all_cells)} cells)")

    if not all_cells:
        log("[WARN] No cells built – saving cropped image as final.")
        save(f"{out_dir}/8_original_stitched.png", img_bgr)
        return img_bgr

    corner_pts = _select_grid_block(all_cells, xs_g, ys_g,
                                    DARK_WANT_ROWS, DARK_WANT_COLS,
                                    DARK_SIZE_TOL)

    # ── Step 8: Full-grid crop with independent padding ───────────────────────
    # Positive EXTRA_* → expand outward (add padding).
    # Negative EXTRA_* → shrink inward (extra crop).
    img_h, img_w = img_bgr.shape[:2]
    l = max(0,     corner_pts["tl"][0] + DARK_INSET - DARK_EXTRA_LEFT)
    t = max(0,     corner_pts["tl"][1] + DARK_INSET - DARK_EXTRA_TOP)
    r = min(img_w, corner_pts["br"][0] - DARK_INSET + DARK_EXTRA_RIGHT)
    b = min(img_h, corner_pts["br"][1] - DARK_INSET + DARK_EXTRA_BOTTOM)
    full_grid_crop = img_bgr[t:b, l:r].copy()
    save(f"{out_dir}/13_full_grid_crop.png", full_grid_crop)
    save(f"{out_dir}/8_original_stitched.png", full_grid_crop)
    log(f"Step 8 done – Full grid crop shape: {full_grid_crop.shape}")
    return full_grid_crop


# ══════════════════════════════════════════════════════════════════════════════
#  BRIGHT PIPELINE  (Steps 3-8)
# ══════════════════════════════════════════════════════════════════════════════

def bright_pipeline(img_bgr: np.ndarray, out_dir: str) -> np.ndarray:
    log("Running BRIGHT pipeline …")
    rows, cols = BRIGHT_ROWS, BRIGHT_COLS

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape

    # ── Step 3: Denoise → illumination correction → CLAHE ────────────────────
    denoised     = cv2.GaussianBlur(gray, (5, 5), 0)
    background   = cv2.GaussianBlur(denoised, (0, 0), 25)
    normalized   = cv2.divide(denoised, background, scale=255)
    clahe        = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    preprocessed = clahe.apply(normalized)
    save(f"{out_dir}/3_preprocessed.png", preprocessed)
    log("Step 3 done – Denoise / Illumination / CLAHE")

    # ── Step 4: Binarize (Otsu) ───────────────────────────────────────────────
    blurred = cv2.GaussianBlur(preprocessed, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    save(f"{out_dir}/4_binary.png", binary)
    log("Step 4 done – Binary (Otsu)")

    # ── Step 5: Morphological clean-up ───────────────────────────────────────
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph  = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    morph  = cv2.morphologyEx(morph,  cv2.MORPH_OPEN,  kernel, iterations=1)
    save(f"{out_dir}/5_morph.png", morph)
    log("Step 5 done – Morphological")

    # ── Step 6: Connected components ─────────────────────────────────────────
    num_labels, labels = cv2.connectedComponents(morph)
    rng      = np.random.default_rng(42)
    comp_vis = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for lbl in range(1, num_labels):
        comp_vis[labels == lbl] = rng.integers(0, 255, size=3, dtype=np.uint8)
    save(f"{out_dir}/6_components.png", comp_vis)
    log(f"Step 6 done – {num_labels - 1} connected components")

    # ── Step 7: Cell extraction + stitching ───────────────────────────────────
    x0 = int(W * 0.20); x1 = int(W * 0.80)
    v_profile = binary[:, x0:x1].astype(float).mean(axis=1)
    row_zones = _find_zones(v_profile, rows)

    ry0, ry1   = row_zones[0][0], row_zones[-1][1]
    ry_mid0    = ry0 + int((ry1 - ry0) * 0.20)
    ry_mid1    = ry0 + int((ry1 - ry0) * 0.80)
    h_profile  = binary[ry_mid0:ry_mid1, :].astype(float).mean(axis=0)
    col_zones  = _find_zones(h_profile, cols)

    all_prof = binary[:, :].astype(float).mean(axis=1)
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

            # Vertical top pad/crop (positive = expand, negative = always crop)
            if r == 0:
                if BRIGHT_CELL_V_PAD_TOP > 0 and add_top:
                    crl = max(0, crl - 1 - BRIGHT_CELL_V_PAD_TOP)
                elif BRIGHT_CELL_V_PAD_TOP < 0:
                    crl = min(H, crl - BRIGHT_CELL_V_PAD_TOP)

            # Vertical bottom pad/crop
            if r == rows - 1:
                if BRIGHT_CELL_V_PAD_BOTTOM > 0 and add_bottom:
                    crh = min(H, crh + 1 + BRIGHT_CELL_V_PAD_BOTTOM)
                elif BRIGHT_CELL_V_PAD_BOTTOM < 0:
                    crh = max(0, crh + BRIGHT_CELL_V_PAD_BOTTOM)

            # Horizontal left/right single-pixel expand (valley-gated)
            if add_left  and c == 0:        ccl = max(0, ccl - 1)
            if add_right and c == cols - 1: cch = min(W, cch + 1)

            # Horizontal pad/crop (positive = expand, negative = crop)
            ccl = max(0, ccl - BRIGHT_CELL_H_PAD)
            cch = min(W, cch + BRIGHT_CELL_H_PAD)
            box = (ccl, crl, cch, crh)
            bin_row.append(binary_pil.crop(box))
            orig_row.append(orig_pil.crop(box))
        bin_cells.append(bin_row)
        orig_cells.append(orig_row)

    bin_stitched  = _stitch_grid(bin_cells,  rows, cols)
    orig_stitched = _stitch_grid(orig_cells, rows, cols)

    pil_save(f"{out_dir}/7_binary_stitched.png",   bin_stitched)
    pil_save(f"{out_dir}/8_original_stitched.png", orig_stitched)
    log("Step 7 done – Cells extracted & stitched")

    return cv2.cvtColor(np.array(orig_stitched), cv2.COLOR_RGB2BGR)


def _find_zones(profile: np.ndarray, n_cells: int,
                min_cell_size: int = 30) -> list:
    DROP_RATIO = 0.20
    SMOOTH_WIN = 11
    LOCAL_WIN  = 80
    n       = len(profile)
    kernel  = np.ones(SMOOTH_WIN) / SMOOTH_WIN
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
    out_w  = sum(col_widths)  + BRIGHT_MARGIN * (cols + 1)
    out_h  = sum(row_heights) + BRIGHT_MARGIN * (rows + 1)
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
                  dark_rows: int, dark_cols: int,
                  bright_rows: int, bright_cols: int) -> bool:
    print(f"\n[IMG] {img_path}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"  [ERROR] Cannot read image: {img_path}")
        return False

    save(f"{out_dir}/0_original.png", img_bgr)

    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    label = classify_image(gray)

    try:
        undist = undistort(img_bgr)
    except Exception as e:
        log(f"[WARN] Undistortion failed ({e}) – using original.")
        undist = img_bgr
    save(f"{out_dir}/1_undistorted.png", undist)

    if label == "DARK":
        ocr_r, ocr_c = _ocr_grid(undist)
        rows = ocr_r or dark_rows
        cols = ocr_c or dark_cols
        cropped = crop_panel(undist, rows, cols, CROP_RATIOS_DARK)
    else:
        ocr_r, ocr_c = _ocr_grid(undist)
        rows = ocr_r or bright_rows
        cols = ocr_c or bright_cols
        cropped = crop_panel(undist, rows, cols, CROP_RATIOS_BRIGHT)
    save(f"{out_dir}/2_cropped.png", cropped)
    log(f"Step 2 done – Cropped to {rows}×{cols} panel")

    try:
        if label == "DARK":
            dark_pipeline(cropped, out_dir)
        else:
            bright_pipeline(cropped, out_dir)
    except Exception as e:
        import traceback
        log(f"[ERROR] Pipeline failed: {e}")
        traceback.print_exc()
        return False

    return True


# ══════════════════════════════════════════════════════════════════════════════
#  FINAL STITCHING  — auto-adapts to 12-image (6+6) or 16-image (8+8) folders
# ══════════════════════════════════════════════════════════════════════════════

def _stitch_layout(num_images: int):
    """
    Return (num_sub_segment, mid_first) for the two supported layouts:
      16 images → 2 columns of 8, split 4 | 4  (segment gap in the middle)
      12 images → 2 columns of 6, split 3 | 3
    For any other count we fall back to a single-column layout.
    """
    if num_images == 16:
        return 8, 4
    if num_images == 12:
        return 6, 3
    # generic fallback: one column of num_images, no mid-split
    half = num_images // 2
    return num_images, half


def build_final_stitched(img_list: list,
                          min_pixel_value: float,
                          max_pixel_value: float,
                          num_images: int) -> np.ndarray:

    num_sub_segment, mid_first = _stitch_layout(num_images)
    num_segment = 2   # always two columns

    sub_h = img_list[0].shape[0]
    sub_w = img_list[0].shape[1]

    gap_first  = INTER_CELL_GAP * (mid_first - 1)
    gap_second = INTER_CELL_GAP * (num_sub_segment - mid_first - 1)
    total_h    = sub_h * num_sub_segment + gap_first + gap_second + SEGMENT_GAP
    total_w    = sub_w * num_segment + COLUMN_GAP
    final      = np.zeros((total_h, total_w), dtype=np.uint8)

    # Build Y positions for one column
    y_positions = []
    y = 0
    for i in range(mid_first):
        y_positions.append(y)
        y += sub_h + INTER_CELL_GAP
    y -= INTER_CELL_GAP
    y += SEGMENT_GAP
    for i in range(mid_first, num_sub_segment):
        y_positions.append(y)
        y += sub_h + INTER_CELL_GAP

    for count, img in enumerate(img_list):
        img_norm = (img - min_pixel_value) / (max_pixel_value - min_pixel_value + 1e-6)
        img_norm = np.clip(img_norm * 255, 0, 255).astype(np.uint8)
        img_norm = np.flip(img_norm, axis=0)

        if count < num_sub_segment:
            y_start = y_positions[count]
            final[y_start:y_start + sub_h, 0:sub_w] = img_norm
        else:
            col_index = count - num_sub_segment
            y_start   = y_positions[col_index]
            x_start   = sub_w + COLUMN_GAP
            final[y_start:y_start + sub_h, x_start:x_start + sub_w] = img_norm

    return final


def build_and_save_final(folder_out: str, num_images: int) -> bool:
    print(f"\n[FINAL] Building composite image for {num_images} images …")
    img_list: list = []
    missing:  list = []
    ref_shape = None

    for j in range(1, num_images + 1):
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
        print(f"  [OK] Loaded image {j}/{num_images}  shape={gray.shape}")

    if ref_shape is None:
        print("  [ERROR] No stitched images found – skipping final build.")
        return False

    blank   = np.zeros(ref_shape, dtype=np.uint8)
    resized = []
    for img in img_list:
        if img is None:
            resized.append(blank)
        elif img.shape != ref_shape:
            resized.append(cv2.resize(img, (ref_shape[1], ref_shape[0])))
        else:
            resized.append(img)

    all_pixels = np.concatenate([im.ravel() for im in resized])
    min_pv     = float(all_pixels.min())
    max_pv     = float(all_pixels.max())
    print(f"  [INFO] Pixel range [{min_pv:.1f}, {max_pv:.1f}]  "
          f"missing: {missing if missing else 'none'}")

    final_arr = build_final_stitched(resized, min_pv, max_pv, num_images)
    save_path = os.path.join(folder_out, "final_image.png")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(save_path, final_arr)
    print(f"  [FINAL] Saved → {save_path}")
    return True


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def _count_images_in_folder(folder: Path) -> int:
    """Auto-detect how many numbered images (1.jpeg … N.jpeg) are present."""
    count = 0
    for j in range(1, 17):          # look for up to 16
        for ext in (".jpeg", ".jpg", ".JPEG", ".JPG"):
            if (folder / f"{j}{ext}").exists():
                count = j
                break
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Process a folder of numbered images through the full pipeline.")
    parser.add_argument("folder", help="Input folder containing 1.jpeg … N.jpeg")
    parser.add_argument("--dark-rows",   type=int, default=DARK_WANT_ROWS)
    parser.add_argument("--dark-cols",   type=int, default=DARK_WANT_COLS)
    parser.add_argument("--bright-rows", type=int, default=BRIGHT_ROWS)
    parser.add_argument("--bright-cols", type=int, default=BRIGHT_COLS)
    parser.add_argument("--num-images",  type=int, default=None,
                        help="Override auto-detected image count (12 or 16)")
    args = parser.parse_args()

    folder = Path(args.folder).resolve()
    if not folder.is_dir():
        print(f"[ERROR] Not a directory: {folder}")
        sys.exit(1)

    num_images = args.num_images or _count_images_in_folder(folder)
    if num_images == 0:
        print(f"[ERROR] No numbered images found in {folder}")
        sys.exit(1)

    folder_name = folder.name
    folder_out  = os.path.join("Output2", folder_name)

    layout_str = "8+8" if num_images == 16 else ("6+6" if num_images == 12 else f"{num_images//2}+{num_images - num_images//2}")

    print(f"\n{'='*60}")
    print(f"  Input folder : {folder}")
    print(f"  Output root  : {os.path.abspath(folder_out)}")
    print(f"  Images found : {num_images}  (stitch layout: {layout_str})")
    print(f"  Dark  grid   : {args.dark_rows}×{args.dark_cols}")
    print(f"  Bright grid  : {args.bright_rows}×{args.bright_cols}")
    print(f"{'='*60}\n")

    success_count = 0
    for j in range(1, num_images + 1):
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
        ok = process_image(img_path, out_dir,
                           args.dark_rows, args.dark_cols,
                           args.bright_rows, args.bright_cols)
        if ok:
            success_count += 1

    print(f"\n{'='*60}")
    print(f"  Processed {success_count}/{num_images} images successfully.")
    print(f"{'='*60}")

    build_and_save_final(folder_out, num_images)

    print(f"\n✅ All done. Output at: {os.path.abspath(folder_out)}")


if __name__ == "__main__":
    main()