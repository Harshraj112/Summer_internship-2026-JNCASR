import os
import sys
import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
import matplotlib
matplotlib.use("Agg")          # headless – comment out if you want live windows
import matplotlib.pyplot as plt
import cv2

# ══════════════════════════════════════════════════════════════════
#  CONFIG  — tweak these to your images
# ══════════════════════════════════════════════════════════════════

FOLDER_PATH = "topcon/Images"          # ← change or pass as CLI arg

# ── Stage-1 : bright-strip extraction ────────────────────────────
BRIGHT_THRESH        = 100
SMOOTH_WINDOW        = 7
VALLEY_SMOOTH_WINDOW = 25
VALLEY_SEARCH        = 60
EXPAND_LEFT          = 0
EXPAND_RIGHT         = 50

# ── Stage-2 : column-border detection ────────────────────────────
GAUSS_KERNEL         = 31     # must be odd
GRADIENT_SIGMA_MULT  = 2      # threshold = mean + N*std − 1
PEAK_DISTANCE        = 20
VALLEY_WINDOW        = 2      # px each side when snapping to valley
EXTRA_LEFT           = 0      # +ve trims left  | -ve adds back
EXTRA_RIGHT          = 0      # +ve trims right | -ve adds back

# ── Stage-3 : stitching ───────────────────────────────────────────
STITCH_MARGIN        = 2      # white gap px between panels
TOP_CUT_SEG1         = 24      # px to cut from top of SEG_1 before stitching
TOP_CUT_SEG2         = 12     # px to cut from top of SEG_2 before stitching
TOP_CUT_SEG3         = 24      # px to cut from top of SEG_3 before stitching

# ══════════════════════════════════════════════════════════════════
#  OUTPUT FOLDER STRUCTURE
#
#  Output/
#   └─ <parent_folder>/           e.g. "Images"
#       ├─ 1_original/
#       ├─ 2_grayscale/
#       ├─ 3_smoothed_profile/    (PNG plot saved per image)
#       ├─ 4_strip_extracted/
#       ├─ 5_intensity_profile/   (PNG plot of Stage-2 analysis)
#       ├─ 6_column_cropped/
#       └─ 7_stitched/
# ══════════════════════════════════════════════════════════════════

def make_output_dirs(base_out):
    steps = [
        "1_original",
        "2_grayscale",
        "3_smoothed_profile",
        "4_strip_extracted",
        "5_intensity_profile",
        "6_column_cropped",
        "7_stitched",
    ]
    dirs = {}
    for s in steps:
        d = os.path.join(base_out, s)
        os.makedirs(d, exist_ok=True)
        dirs[s] = d
    return dirs


def save_img(arr_rgb, path):
    """Save an RGB numpy array as PNG."""
    cv2.imwrite(path, cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR))


def save_gray(arr_gray, path):
    cv2.imwrite(path, arr_gray)


# ──────────────────────────────────────────────────────────────────
#  STAGE-1 HELPERS
# ──────────────────────────────────────────────────────────────────

def find_bright_runs(col_means, smooth_window, threshold):
    smoothed = uniform_filter1d(col_means, size=smooth_window)
    bright   = smoothed > threshold

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
            f"No bright strip found (threshold={threshold}). "
            "Try lowering BRIGHT_THRESH."
        )

    top2 = sorted(runs, key=lambda r: r[1] - r[0], reverse=True)[:2]
    top2.sort()
    (x0, x1)     = top2[0]
    (x0_2, x1_2) = top2[1] if len(top2) > 1 else top2[0]
    return x0, x1, x0_2, x1_2, smoothed, bright


def find_nearest_valley(valley_smoothed, anchor, direction, window):
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


def extract_strip(image_path, dirs, stem):
    """
    Stage-1 pipeline.
    Saves: 1_original, 2_grayscale, 3_smoothed_profile, 4_strip_extracted.
    Returns: strip (RGB ndarray).
    """
    # ── load ──────────────────────────────────────────────────────
    pil  = Image.open(image_path)
    gray = np.array(pil.convert("L"))
    rgb  = np.array(pil.convert("RGB"))

    # 1. original
    save_img(rgb, os.path.join(dirs["1_original"], f"{stem}.png"))

    # 2. grayscale
    save_gray(gray, os.path.join(dirs["2_grayscale"], f"{stem}.png"))

    # ── processing ───────────────────────────────────────────────
    col_means = gray.mean(axis=0).astype(float)

    x0, x1, x0_2, x1_2, smoothed, bright = find_bright_runs(
        col_means, SMOOTH_WINDOW, BRIGHT_THRESH
    )

    valley_smoothed = uniform_filter1d(col_means, size=VALLEY_SMOOTH_WINDOW)
    left_edge  = find_nearest_valley(valley_smoothed, x0,   "left",  VALLEY_SEARCH)
    right_edge = find_nearest_valley(valley_smoothed, x1_2, "right", VALLEY_SEARCH)
    left_edge  = int(np.clip(left_edge  - EXPAND_LEFT,  0, rgb.shape[1] - 1))
    right_edge = int(np.clip(right_edge + EXPAND_RIGHT, 0, rgb.shape[1] - 1))

    # 3. smoothed-profile plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    axes[0].plot(col_means,      label="raw means")
    axes[0].axhline(BRIGHT_THRESH, linestyle="--", label=f"threshold={BRIGHT_THRESH}")
    axes[0].set_title("Raw mean intensity per column")
    axes[0].legend(fontsize=8)

    axes[1].plot(smoothed, label=f"light-smoothed (w={SMOOTH_WINDOW})")
    axes[1].axhline(BRIGHT_THRESH, linestyle="--", label=f"threshold={BRIGHT_THRESH}")
    axes[1].set_title("Light-smoothed → run detection")
    axes[1].legend(fontsize=8)

    axes[2].plot(valley_smoothed, color="orange",
                 label=f"valley-smoothed (w={VALLEY_SMOOTH_WINDOW})")
    axes[2].axvline(left_edge,  color="red",   linestyle="--",
                    label=f"left={left_edge}")
    axes[2].axvline(right_edge, color="green", linestyle="--",
                    label=f"right={right_edge}")
    axes[2].set_title("Heavy-smoothed → valley detection")
    axes[2].legend(fontsize=8)

    plt.suptitle(f"{stem} – Stage-1 profiles", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(dirs["3_smoothed_profile"], f"{stem}.png"), dpi=120)
    plt.close(fig)

    # 4. extracted strip
    strip = rgb[:, left_edge:right_edge + 1]
    save_img(strip, os.path.join(dirs["4_strip_extracted"], f"{stem}.png"))

    print(f"  [S1] {stem}: cols {left_edge}→{right_edge}  "
          f"(w={right_edge - left_edge + 1}px)")
    return strip


# ──────────────────────────────────────────────────────────────────
#  STAGE-2 HELPERS
# ──────────────────────────────────────────────────────────────────

def detect_and_crop(strip_rgb, dirs, stem):
    """
    Stage-2 pipeline.
    Saves: 5_intensity_profile, 6_column_cropped.
    Returns: cropped (RGB ndarray).
    """
    gray = cv2.cvtColor(strip_rgb, cv2.COLOR_RGB2GRAY)

    profile        = gray.mean(axis=0)
    profile_smooth = cv2.GaussianBlur(
        profile.reshape(1, -1).astype(np.float32),
        (1, GAUSS_KERNEL), 0
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
    valleys = np.array(valleys)

    # 5. intensity-profile plot (4 sub-plots)
    fig, axes = plt.subplots(4, 1, figsize=(12, 14))

    axes[0].imshow(gray, cmap="gray", aspect="auto")
    axes[0].set_title("Extracted strip (grayscale)")
    axes[0].axis("off")

    axes[1].plot(profile)
    axes[1].set_title("Raw vertical intensity profile")
    axes[1].grid(True)

    axes[2].plot(profile_smooth)
    axes[2].set_title("Smoothed intensity profile")
    axes[2].grid(True)

    axes[3].plot(gradient, label="gradient")
    axes[3].axhline(threshold, color="r", linestyle="--", label="threshold")
    for x in change_points:
        axes[3].axvline(x, color="g", alpha=0.6)
    for x in valleys:
        axes[3].axvline(x, color="red", linewidth=2, alpha=0.8)
    axes[3].set_title("Gradient + change points (green) + valley borders (red)")
    axes[3].legend(fontsize=8)
    axes[3].grid(True)

    plt.suptitle(f"{stem} – Stage-2 analysis", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(dirs["5_intensity_profile"], f"{stem}.png"), dpi=120)
    plt.close(fig)

    # crop
    if len(valleys) < 2:
        print(f"  [S2] {stem}: not enough borders – keeping full strip")
        cropped = strip_rgb
    else:
        x_left  = int(valleys.min()) + EXTRA_LEFT
        x_right = int(valleys.max()) - EXTRA_RIGHT
        x_left  = max(0, x_left)
        x_right = min(strip_rgb.shape[1] - 1, x_right)
        cropped = strip_rgb[:, x_left:x_right]
        print(f"  [S2] {stem}: borders {valleys}  →  crop {x_left}→{x_right}  "
              f"(w={x_right - x_left}px)")

    # 6. column-cropped
    save_img(cropped, os.path.join(dirs["6_column_cropped"], f"{stem}.png"))
    return cropped


# ──────────────────────────────────────────────────────────────────
#  STAGE-3 : STITCH
# ──────────────────────────────────────────────────────────────────

def stitch_panels(panels, margin=2):
    max_h = max(p.shape[0] for p in panels)
    padded = []
    for p in panels:
        if p.shape[0] < max_h:
            pad = np.full((max_h - p.shape[0], p.shape[1], 3), 255, dtype=np.uint8)
            p   = np.vstack([p, pad])
        padded.append(p)

    gap   = np.full((max_h, margin, 3), 0, dtype=np.uint8)
    parts = []
    for i, p in enumerate(padded):
        parts.append(p)
        if i < len(padded) - 1:
            parts.append(gap)
    return np.hstack(parts)


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def main(folder):
    folder     = os.path.abspath(folder)
    parent     = os.path.basename(folder)          # e.g. "Images"
    base_out   = os.path.join("Output", parent)    # e.g. "Output/Images"
    dirs       = make_output_dirs(base_out)

    print(f"\n📁 Input  : {folder}")
    print(f"📁 Output : {os.path.abspath(base_out)}\n")

    seg_names = ["SEG_1.png", "SEG_2.png", "SEG_3.png"]
    panels    = []

    for name in seg_names:
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            print(f"  WARNING: '{name}' not found – skipping.")
            continue

        stem = os.path.splitext(name)[0]           # "SEG_1"
        print(f"▶  {name}")

        strip   = extract_strip(path, dirs, stem)
        cropped = detect_and_crop(strip, dirs, stem)
        panels.append(cropped)

    if not panels:
        print("No images processed. Check FOLDER_PATH.")
        return

    # ── hardcoded top cuts for SEG_1 and SEG_3 ───────────────────
    found_stems = [os.path.splitext(n)[0] for n in seg_names
                   if os.path.isfile(os.path.join(folder, n))]
    top_cuts = {"SEG_1": TOP_CUT_SEG1, "SEG_2": TOP_CUT_SEG2, "SEG_3": TOP_CUT_SEG3}
    for i, stem in enumerate(found_stems):
        cut = top_cuts.get(stem, 0)
        if cut > 0:
            panels[i] = panels[i][cut:, :, :]
            print(f"  [top-cut] {stem}: removed {cut}px from top  "
                  f"(new h={panels[i].shape[0]}px)")

    # stitch
    print(f"\n▶  Stitching {len(panels)} panel(s) with {STITCH_MARGIN}px margin …")
    stitched = stitch_panels(panels, margin=STITCH_MARGIN)

    stitch_path = os.path.join(dirs["7_stitched"], "stitched_result.png")
    save_img(stitched, stitch_path)
    print(f"   Saved → {stitch_path}  ({stitched.shape[1]}×{stitched.shape[0]}px)")

    # summary overview plot
    fig, axes = plt.subplots(1, len(panels) + 1,
                             figsize=(6 * (len(panels) + 1), 10))
    if len(panels) == 1:
        axes = list(axes) + [axes]
    for i, (ax, panel) in enumerate(zip(axes[:-1], panels)):
        ax.imshow(panel)
        ax.set_title(f"SEG_{i + 1} – cropped")
        ax.axis("off")
    axes[-1].imshow(stitched)
    axes[-1].set_title("Stitched result")
    axes[-1].axis("off")
    plt.tight_layout()
    fig.savefig(os.path.join(dirs["7_stitched"], "overview.png"), dpi=120)
    plt.close(fig)
    print(f"   Overview → {os.path.join(dirs['7_stitched'], 'overview.png')}")

    print("\n✅  Done.\n")
    print("Output tree:")
    for key, d in dirs.items():
        files = os.listdir(d)
        print(f"  {key}/  ({len(files)} file{'s' if len(files) != 1 else ''})")


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else FOLDER_PATH
    main(folder)