import os
import sys
import time
import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
import cv2
from concurrent.futures import ThreadPoolExecutor

# ══════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════

FOLDER_PATH = "topcon/Images"

# Stage-0 (White Border Removal)
WHITE_BORDER_THRESH      = 200   # pixel value considered "white" (0-255)
WHITE_BORDER_COL_RATIO   = 0.85  # a column is "white" if this fraction of pixels exceed thresh
WHITE_BORDER_ROW_RATIO   = 0.85  # a row    is "white" if this fraction of pixels exceed thresh
WHITE_BORDER_MIN_CONTENT = 10    # minimum px of content to keep (safety guard)

# Stage-1
BRIGHT_THRESH        = 100
SMOOTH_WINDOW        = 7
VALLEY_SMOOTH_WINDOW = 25
VALLEY_SEARCH        = 60
EXPAND_LEFT          = 0
EXPAND_RIGHT         = 50

# Stage-2
GAUSS_KERNEL         = 31
GRADIENT_SIGMA_MULT  = 2
PEAK_DISTANCE        = 20
VALLEY_WINDOW        = 2
EXTRA_LEFT           = 0
EXTRA_RIGHT          = 0

# Stage-3
SCALE_X              = 1.4
SCALE_Y              = 1

# Stage-4
TOP_CUT_SEG1         = 24
TOP_CUT_SEG2         = 12
TOP_CUT_SEG3         = 25

# Stage-5
STITCH_MARGIN        = 2

# ══════════════════════════════════════════════════════════════════
#  TIMING UTILITY
# ══════════════════════════════════════════════════════════════════

class Timer:
    def __init__(self):
        self._laps  = []
        self._start = time.perf_counter()

    def lap(self, label):
        t       = time.perf_counter()
        elapsed = t - (self._laps[-1][1] if self._laps else self._start)
        self._laps.append((label, t, elapsed))
        return elapsed

    def total(self):
        return time.perf_counter() - self._start

    def report(self):
        print("\n" + "═" * 52)
        print(f"{'TIMING REPORT':^52}")
        print("═" * 52)
        for label, _, dur in self._laps:
            bar = "█" * int(dur / self.total() * 30)
            print(f"  {label:<28} {dur:>6.3f}s  {bar}")
        print("─" * 52)
        print(f"  {'TOTAL':<28} {self.total():>6.3f}s")
        print("═" * 52)


# ══════════════════════════════════════════════════════════════════
#  STAGE-0 : white border removal  (no I/O)
# ══════════════════════════════════════════════════════════════════

def remove_white_border(rgb: np.ndarray) -> np.ndarray:
    """
    Detects and removes white/bright borders from all four sides of an RGB image.

    Strategy
    ────────
    1. Convert to grayscale for analysis only.
    2. For each row:  if ≥ WHITE_BORDER_ROW_RATIO of its pixels are above
       WHITE_BORDER_THRESH → consider it a "white row".
    3. For each column: same logic with WHITE_BORDER_COL_RATIO.
    4. Walk inward from each edge to find the first non-white row/column.
    5. Crop and return; original array is unchanged outside the crop.

    Parameters
    ----------
    rgb : np.ndarray  shape (H, W, 3), dtype uint8

    Returns
    -------
    np.ndarray  cropped RGB image (may equal the input if no border detected)
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)  # (H, W)
    H, W = gray.shape

    # ── per-row and per-column "white" masks ──────────────────────
    # white_row[i]  = True  if row i is predominantly white
    # white_col[j]  = True  if col j is predominantly white
    white_row = (gray > WHITE_BORDER_THRESH).mean(axis=1) >= WHITE_BORDER_ROW_RATIO  # (H,)
    white_col = (gray > WHITE_BORDER_THRESH).mean(axis=0) >= WHITE_BORDER_COL_RATIO  # (W,)

    # ── find crop boundaries by walking inward ────────────────────
    top    = 0
    bottom = H - 1
    left   = 0
    right  = W - 1

    # top edge
    while top < H and white_row[top]:
        top += 1

    # bottom edge
    while bottom > top and white_row[bottom]:
        bottom -= 1

    # left edge
    while left < W and white_col[left]:
        left += 1

    # right edge
    while right > left and white_col[right]:
        right -= 1

    # ── safety: keep at least WHITE_BORDER_MIN_CONTENT pixels ─────
    if (bottom - top + 1) < WHITE_BORDER_MIN_CONTENT:
        print("  ⚠  remove_white_border: content height too small after crop – skipping.")
        return rgb
    if (right - left + 1) < WHITE_BORDER_MIN_CONTENT:
        print("  ⚠  remove_white_border: content width  too small after crop – skipping.")
        return rgb

    cropped = rgb[top : bottom + 1, left : right + 1]

    removed_rows = (top) + (H - 1 - bottom)
    removed_cols = (left) + (W - 1 - right)
    if removed_rows > 0 or removed_cols > 0:
        print(
            f"  ✂  White border removed: "
            f"top={top}px  bottom={H-1-bottom}px  "
            f"left={left}px  right={W-1-right}px  "
            f"→  {cropped.shape[1]}×{cropped.shape[0]}px"
        )

    return cropped


# ══════════════════════════════════════════════════════════════════
#  STAGE-1 : bright-strip extraction  (no I/O)
# ══════════════════════════════════════════════════════════════════

def _find_bright_runs(col_means):
    smoothed = uniform_filter1d(col_means, size=SMOOTH_WINDOW)
    bright   = smoothed > BRIGHT_THRESH

    runs, start = [], None
    for i, v in enumerate(bright):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(bright) - 1))

    if not runs:
        raise ValueError(
            f"No bright strip found (threshold={BRIGHT_THRESH}). "
            "Try lowering BRIGHT_THRESH."
        )

    top2 = sorted(runs, key=lambda r: r[1] - r[0], reverse=True)[:2]
    top2.sort()
    (x0, x1)     = top2[0]
    (x0_2, x1_2) = top2[1] if len(top2) > 1 else top2[0]
    return x0, x1, x0_2, x1_2


def _nearest_valley(valley_smoothed, anchor, direction, window):
    n = len(valley_smoothed)
    if direction == "left":
        lo, prev = max(0, anchor - window), valley_smoothed[anchor]
        for i in range(anchor - 1, lo - 1, -1):
            curr = valley_smoothed[i]
            if curr > prev:
                return i + 1
            prev = curr
        return lo
    else:
        hi, prev = min(n - 1, anchor + window), valley_smoothed[anchor]
        for i in range(anchor + 1, hi + 1):
            curr = valley_smoothed[i]
            if curr > prev:
                return i - 1
            prev = curr
        return hi


def extract_strip(rgb, gray):
    """Stage-1: returns strip (RGB ndarray) — no disk I/O."""
    col_means = gray.mean(axis=0).astype(np.float32)

    x0, x1, x0_2, x1_2 = _find_bright_runs(col_means)

    valley_smoothed = uniform_filter1d(col_means, size=VALLEY_SMOOTH_WINDOW)
    left_edge  = _nearest_valley(valley_smoothed, x0,   "left",  VALLEY_SEARCH)
    right_edge = _nearest_valley(valley_smoothed, x1_2, "right", VALLEY_SEARCH)
    left_edge  = int(np.clip(left_edge  - EXPAND_LEFT,  0, rgb.shape[1] - 1))
    right_edge = int(np.clip(right_edge + EXPAND_RIGHT, 0, rgb.shape[1] - 1))

    return rgb[:, left_edge:right_edge + 1]


# ══════════════════════════════════════════════════════════════════
#  STAGE-2 : column-border detection & crop  (no I/O)
# ══════════════════════════════════════════════════════════════════

def detect_and_crop(strip_rgb):
    """Stage-2: returns cropped (RGB ndarray) — no disk I/O."""
    gray    = cv2.cvtColor(strip_rgb, cv2.COLOR_RGB2GRAY)
    profile = gray.mean(axis=0).astype(np.float32)

    profile_smooth = cv2.GaussianBlur(
        profile.reshape(1, -1), (1, GAUSS_KERNEL), 0
    ).flatten()

    gradient  = np.abs(np.gradient(profile_smooth))
    threshold = gradient.mean() + GRADIENT_SIGMA_MULT * gradient.std() - 1

    change_points, _ = find_peaks(
        gradient, height=threshold, distance=PEAK_DISTANCE
    )

    valleys = []
    for peak in change_points:
        lo     = max(0, peak - VALLEY_WINDOW)
        hi     = min(len(profile_smooth), peak + VALLEY_WINDOW)
        valley = lo + np.argmin(profile_smooth[lo:hi])
        valleys.append(valley)

    if len(valleys) < 2:
        return strip_rgb

    x_left  = max(0,                      min(valleys) + EXTRA_LEFT)
    x_right = min(strip_rgb.shape[1] - 1, max(valleys) - EXTRA_RIGHT)
    return strip_rgb[:, x_left:x_right]


# ══════════════════════════════════════════════════════════════════
#  STAGE-3 : stretch  (no I/O)
# ══════════════════════════════════════════════════════════════════

def stretch_panel(cropped_rgb):
    """Stage-3: returns stretched (RGB ndarray) — no disk I/O."""
    if SCALE_X == 1 and SCALE_Y == 1:
        return cropped_rgb
    h, w     = cropped_rgb.shape[:2]
    new_w, new_h = int(w * SCALE_X), int(h * SCALE_Y)
    return cv2.resize(
        cropped_rgb, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4
    )


# ══════════════════════════════════════════════════════════════════
#  PROCESS ONE SEGMENT  (used by thread pool)
# ══════════════════════════════════════════════════════════════════

TOP_CUTS = {"SEG_1": TOP_CUT_SEG1, "SEG_2": TOP_CUT_SEG2, "SEG_3": TOP_CUT_SEG3}


def process_segment(args):
    """
    Full per-segment pipeline  S0 → S1 → S2 → S3.
      S0: white border removal
      S1: bright-strip extraction
      S2: column-border detection & crop
      S3: stretch

    Returns: (stem, stretched_rgb)  or raises on error.
    """
    path, stem = args
    t0 = time.perf_counter()

    # ── Load ──────────────────────────────────────────────────────
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    print(f"\n  [{stem}]  original size: {rgb.shape[1]}×{rgb.shape[0]}px")

    # ── Stage-0 : remove white border BEFORE any grayscale work ──
    rgb = remove_white_border(rgb)

    # ── Re-derive grayscale from border-free RGB ──────────────────
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # ── Stage-1 : bright-strip extraction ─────────────────────────
    strip = extract_strip(rgb, gray)

    # ── Stage-2 : column crop ─────────────────────────────────────
    cropped = detect_and_crop(strip)

    # ── Stage-4 : top cut (segment-specific) ─────────────────────
    cut = TOP_CUTS.get(stem, 0)
    if cut > 0:
        cropped = cropped[cut:, :, :]

    # ── Stage-3 : stretch ─────────────────────────────────────────
    stretched = stretch_panel(cropped)

    elapsed = time.perf_counter() - t0
    print(f"  ✔  {stem}  →  {stretched.shape[1]}×{stretched.shape[0]}px  "
          f"[{elapsed:.3f}s]")
    return stem, stretched


# ══════════════════════════════════════════════════════════════════
#  STITCHING
# ══════════════════════════════════════════════════════════════════

def stitch_panels(panels, margin=STITCH_MARGIN):
    """Horizontally stitch panels with a black gap of `margin` px."""
    max_h = max(p.shape[0] for p in panels)
    parts = []
    gap   = np.zeros((max_h, margin, 3), dtype=np.uint8)

    for i, p in enumerate(panels):
        if p.shape[0] < max_h:
            pad = np.zeros((max_h - p.shape[0], p.shape[1], 3), dtype=np.uint8)
            p   = np.vstack([p, pad])
        parts.append(p)
        if i < len(panels) - 1:
            parts.append(gap)

    return np.hstack(parts)


def mirror_panel(panel):
    return np.fliplr(panel)


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def main(folder):
    timer  = Timer()
    folder = os.path.abspath(folder)
    parent = os.path.basename(folder)

    base_out   = os.path.join("Output", parent)
    stitch_dir = os.path.join(base_out, "8_stitched")
    os.makedirs(stitch_dir, exist_ok=True)

    print(f"\n📁 Input  : {folder}")
    print(f"📁 Output : {os.path.abspath(stitch_dir)}\n")

    # ── Collect existing segment paths ────────────────────────────
    seg_names = ["SEG_1.png", "SEG_2.png", "SEG_3.png"]
    tasks = []
    for name in seg_names:
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            tasks.append((path, os.path.splitext(name)[0]))
        else:
            print(f"  WARNING: '{name}' not found – skipping.")

    if not tasks:
        print("\nNo images processed. Check FOLDER_PATH.")
        return

    # ── Parallel segment processing ───────────────────────────────
    print(f"▶  Processing {len(tasks)} segment(s) in parallel …\n")
    results = {}
    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        for stem, stretched in pool.map(process_segment, tasks):
            results[stem] = stretched

    # Restore original order
    ordered_stems    = [t[1] for t in tasks if t[1] in results]
    stretched_panels = [results[s] for s in ordered_stems]

    timer.lap("Segment processing (parallel)")

    # ── Normal stitch ─────────────────────────────────────────────
    normal_stitched = stitch_panels(stretched_panels)
    normal_path = os.path.join(stitch_dir, "stitched_normal.png")
    cv2.imwrite(normal_path, cv2.cvtColor(normal_stitched, cv2.COLOR_RGB2BGR))
    print(f"\n▶  Normal  saved → {normal_path}  "
          f"({normal_stitched.shape[1]}×{normal_stitched.shape[0]}px)")
    timer.lap("Normal stitch + save")

    # ── Inverted stitch ───────────────────────────────────────────
    mirrored = [mirror_panel(p) for p in stretched_panels]
    if len(mirrored) >= 3:
        mirrored[0], mirrored[-1] = mirrored[-1], mirrored[0]

    inv_stitched = stitch_panels(mirrored)
    inv_path = os.path.join(stitch_dir, "stitched_inverted.png")
    cv2.imwrite(inv_path, cv2.cvtColor(inv_stitched, cv2.COLOR_RGB2BGR))
    print(f"▶  Inverted saved → {inv_path}  "
          f"({inv_stitched.shape[1]}×{inv_stitched.shape[0]}px)")
    timer.lap("Inverted stitch + save")

    # ── Timing report ─────────────────────────────────────────────
    timer.report()
    print(f"\n✅  Done.\n")


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else FOLDER_PATH
    main(folder)