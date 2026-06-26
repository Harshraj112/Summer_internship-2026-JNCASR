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
        self._laps = []
        self._start = time.perf_counter()

    def lap(self, label):
        t = time.perf_counter()
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
    gray = cv2.cvtColor(strip_rgb, cv2.COLOR_RGB2GRAY)
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
    h, w = cropped_rgb.shape[:2]
    new_w, new_h = int(w * SCALE_X), int(h * SCALE_Y)
    # cv2.resize is faster than PIL LANCZOS for large arrays
    return cv2.resize(
        cropped_rgb, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4
    )


# ══════════════════════════════════════════════════════════════════
#  PROCESS ONE SEGMENT  (used by thread pool)
# ══════════════════════════════════════════════════════════════════

TOP_CUTS = {"SEG_1": TOP_CUT_SEG1, "SEG_2": TOP_CUT_SEG2, "SEG_3": TOP_CUT_SEG3}


def process_segment(args):
    """
    Full per-segment pipeline (S1→S3).
    Returns: (stem, stretched_rgb)  or raises on error.
    """
    path, stem = args
    t0 = time.perf_counter()

    # Load once with cv2 (faster than PIL for large images)
    bgr  = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    strip    = extract_strip(rgb, gray)
    cropped  = detect_and_crop(strip)

    cut = TOP_CUTS.get(stem, 0)
    if cut > 0:
        cropped = cropped[cut:, :, :]

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
    ordered_stems   = [t[1] for t in tasks if t[1] in results]
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