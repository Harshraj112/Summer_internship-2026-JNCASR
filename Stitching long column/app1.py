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
#  CONFIG  — tweak these to suit your images
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

# ── Stage-3 : stretch ─────────────────────────────────────────────
SCALE_X              = 2      # horizontal stretch multiplier
SCALE_Y              = 1      # vertical stretch multiplier

# ── Stage-4 : top-cuts before stitching ──────────────────────────
TOP_CUT_SEG1         = 24     # px to cut from top of SEG_1
TOP_CUT_SEG2         = 12     # px to cut from top of SEG_2
TOP_CUT_SEG3         = 24     # px to cut from top of SEG_3

# ── Stage-5 : stitching ───────────────────────────────────────────
STITCH_MARGIN        = 2      # black gap px between panels

# ══════════════════════════════════════════════════════════════════
#  OUTPUT FOLDER STRUCTURE
#
#  Output/
#   └─ <input_folder_name>/          e.g. "Images"
#       ├─ SEG_1/
#       │   ├─ 1_original.png
#       │   ├─ 2_grayscale.png
#       │   ├─ 3_smoothed_profile.png
#       │   ├─ 4_strip_extracted.png
#       │   ├─ 5_intensity_profile.png
#       │   ├─ 6_column_cropped.png
#       │   └─ 7_stretched.png
#       ├─ SEG_2/  (same sub-files)
#       ├─ SEG_3/  (same sub-files)
#       └─ 8_stitched/
#           ├─ stitched_normal.png
#           ├─ stitched_inverted.png
#           └─ overview.png
# ══════════════════════════════════════════════════════════════════


def make_seg_dir(base_out, stem):
    """Create per-image output folder and return its path."""
    d = os.path.join(base_out, stem)
    os.makedirs(d, exist_ok=True)
    return d


def make_stitch_dir(base_out):
    d = os.path.join(base_out, "8_stitched")
    os.makedirs(d, exist_ok=True)
    return d


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


# ──────────────────────────────────────────────────────────────────
#  STAGE-1 : bright-strip extraction
# ──────────────────────────────────────────────────────────────────

def extract_strip(image_path, seg_dir, stem):
    """
    Stage-1 pipeline.
    Saves: 1_original, 2_grayscale, 3_smoothed_profile, 4_strip_extracted.
    Returns: strip (RGB ndarray).
    """
    pil  = Image.open(image_path)
    gray = np.array(pil.convert("L"))
    rgb  = np.array(pil.convert("RGB"))

    save_img(rgb,  os.path.join(seg_dir, "1_original.png"))
    save_gray(gray, os.path.join(seg_dir, "2_grayscale.png"))

    col_means = gray.mean(axis=0).astype(float)

    x0, x1, x0_2, x1_2, smoothed, bright = find_bright_runs(
        col_means, SMOOTH_WINDOW, BRIGHT_THRESH
    )

    valley_smoothed = uniform_filter1d(col_means, size=VALLEY_SMOOTH_WINDOW)
    left_edge  = find_nearest_valley(valley_smoothed, x0,   "left",  VALLEY_SEARCH)
    right_edge = find_nearest_valley(valley_smoothed, x1_2, "right", VALLEY_SEARCH)
    left_edge  = int(np.clip(left_edge  - EXPAND_LEFT,  0, rgb.shape[1] - 1))
    right_edge = int(np.clip(right_edge + EXPAND_RIGHT, 0, rgb.shape[1] - 1))

    # 3 – smoothed-profile plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    axes[0].plot(col_means, label="raw means")
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
    fig.savefig(os.path.join(seg_dir, "3_smoothed_profile.png"), dpi=120)
    plt.close(fig)

    # 4 – extracted strip
    strip = rgb[:, left_edge:right_edge + 1]
    save_img(strip, os.path.join(seg_dir, "4_strip_extracted.png"))

    print(f"  [S1] {stem}: cols {left_edge}→{right_edge}  "
          f"(w={right_edge - left_edge + 1}px)")
    return strip


# ──────────────────────────────────────────────────────────────────
#  STAGE-2 : column-border detection & crop
# ──────────────────────────────────────────────────────────────────

def detect_and_crop(strip_rgb, seg_dir, stem):
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

    # 5 – intensity-profile plot
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
    fig.savefig(os.path.join(seg_dir, "5_intensity_profile.png"), dpi=120)
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

    save_img(cropped, os.path.join(seg_dir, "6_column_cropped.png"))
    return cropped


# ──────────────────────────────────────────────────────────────────
#  STAGE-3 : stretch
# ──────────────────────────────────────────────────────────────────

def stretch_panel(cropped_rgb, seg_dir, stem):
    """
    Stage-3 pipeline.
    Applies SCALE_X / SCALE_Y resize using LANCZOS resampling.
    Saves: 7_stretched.png
    Returns: stretched (RGB ndarray).
    """
    pil = Image.fromarray(cropped_rgb)
    new_w = int(pil.width  * SCALE_X)
    new_h = int(pil.height * SCALE_Y)
    stretched_pil = pil.resize((new_w, new_h), Image.LANCZOS)
    stretched = np.array(stretched_pil)

    save_img(stretched, os.path.join(seg_dir, "7_stretched.png"))
    print(f"  [S3] {stem}: {pil.width}×{pil.height} → {new_w}×{new_h} "
          f"(×{SCALE_X} horiz, ×{SCALE_Y} vert)")
    return stretched


# ──────────────────────────────────────────────────────────────────
#  STAGE-4 : stitching helpers
# ──────────────────────────────────────────────────────────────────

def stitch_panels(panels, margin=2):
    """Horizontally stitch panels with a black gap of `margin` px."""
    max_h = max(p.shape[0] for p in panels)
    padded = []
    for p in panels:
        if p.shape[0] < max_h:
            pad = np.full(
                (max_h - p.shape[0], p.shape[1], 3), 0, dtype=np.uint8
            )
            p = np.vstack([p, pad])
        padded.append(p)

    gap   = np.full((max_h, margin, 3), 0, dtype=np.uint8)
    parts = []
    for i, p in enumerate(padded):
        parts.append(p)
        if i < len(padded) - 1:
            parts.append(gap)
    return np.hstack(parts)


def mirror_panel(panel):
    """Flip a panel horizontally (left-right mirror)."""
    return np.fliplr(panel)


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def main(folder):
    folder   = os.path.abspath(folder)
    parent   = os.path.basename(folder)          # e.g. "Images"
    base_out = os.path.join("Output", parent)    # e.g. "Output/Images"
    os.makedirs(base_out, exist_ok=True)

    print(f"\n📁 Input  : {folder}")
    print(f"📁 Output : {os.path.abspath(base_out)}\n")

    seg_names = ["SEG_1.png", "SEG_2.png", "SEG_3.png"]
    top_cuts  = {
        "SEG_1": TOP_CUT_SEG1,
        "SEG_2": TOP_CUT_SEG2,
        "SEG_3": TOP_CUT_SEG3,
    }

    stretched_panels = []   # final processed panels (after top-cut + stretch)
    found_stems      = []

    for name in seg_names:
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            print(f"  WARNING: '{name}' not found – skipping.")
            continue

        stem    = os.path.splitext(name)[0]       # "SEG_1"
        seg_dir = make_seg_dir(base_out, stem)    # Output/Images/SEG_1/

        print(f"\n▶  {name}")

        # S1 – extract bright strip
        strip = extract_strip(path, seg_dir, stem)

        # S2 – detect borders & crop columns
        cropped = detect_and_crop(strip, seg_dir, stem)

        # Apply top-cut before stretching
        cut = top_cuts.get(stem, 0)
        if cut > 0:
            cropped = cropped[cut:, :, :]
            print(f"  [top-cut] {stem}: removed {cut}px from top  "
                  f"(new h={cropped.shape[0]}px)")

        # S3 – stretch
        stretched = stretch_panel(cropped, seg_dir, stem)

        stretched_panels.append(stretched)
        found_stems.append(stem)

    if not stretched_panels:
        print("\nNo images processed. Check FOLDER_PATH.")
        return

    stitch_dir = make_stitch_dir(base_out)

    # ── Normal stitch (SEG_1 | SEG_2 | SEG_3) ────────────────────
    print(f"\n▶  Stitching NORMAL ({len(stretched_panels)} panel(s)) …")
    normal_stitched = stitch_panels(stretched_panels, margin=STITCH_MARGIN)
    normal_path = os.path.join(stitch_dir, "stitched_normal.png")
    save_img(normal_stitched, normal_path)
    print(f"   Saved → {normal_path}  "
          f"({normal_stitched.shape[1]}×{normal_stitched.shape[0]}px)")

    # ── Inverted stitch ───────────────────────────────────────────
    #   • mirror every panel horizontally
    #   • swap positions: panel[0] ↔ panel[-1]  (SEG_1 ↔ SEG_3)
    mirrored_panels = [mirror_panel(p) for p in stretched_panels]

    if len(mirrored_panels) >= 3:
        # Swap first and last panel
        inv_panels = mirrored_panels.copy()
        inv_panels[0], inv_panels[-1] = inv_panels[-1], inv_panels[0]
        swap_msg = f"positions swapped: {found_stems[0]} ↔ {found_stems[-1]}"
    else:
        inv_panels = mirrored_panels
        swap_msg   = "fewer than 3 panels – no position swap applied"

    print(f"\n▶  Stitching INVERTED (mirrored + {swap_msg}) …")
    inv_stitched = stitch_panels(inv_panels, margin=STITCH_MARGIN)
    inv_path = os.path.join(stitch_dir, "stitched_inverted.png")
    save_img(inv_stitched, inv_path)
    print(f"   Saved → {inv_path}  "
          f"({inv_stitched.shape[1]}×{inv_stitched.shape[0]}px)")

    # ── Overview plot ─────────────────────────────────────────────
    n_panels = len(stretched_panels)
    fig, axes = plt.subplots(
        1, n_panels + 2,
        figsize=(5 * (n_panels + 2), 8)
    )
    axes = list(axes)

    for i, (ax, panel, stem) in enumerate(
        zip(axes[:n_panels], stretched_panels, found_stems)
    ):
        ax.imshow(panel)
        ax.set_title(f"{stem}\n(stretched)")
        ax.axis("off")

    axes[n_panels].imshow(normal_stitched)
    axes[n_panels].set_title("Stitched – Normal")
    axes[n_panels].axis("off")

    axes[n_panels + 1].imshow(inv_stitched)
    axes[n_panels + 1].set_title("Stitched – Inverted\n(mirrored + swapped)")
    axes[n_panels + 1].axis("off")

    plt.tight_layout()
    overview_path = os.path.join(stitch_dir, "overview.png")
    fig.savefig(overview_path, dpi=120)
    plt.close(fig)
    print(f"   Overview → {overview_path}")

    # ── Summary ───────────────────────────────────────────────────
    print("\n✅  Done.\n")
    print("Output tree:")
    print(f"  Output/{parent}/")
    for stem in found_stems:
        seg_dir = os.path.join(base_out, stem)
        files   = sorted(os.listdir(seg_dir))
        print(f"    {stem}/  ({len(files)} files)")
        for f in files:
            print(f"      {f}")
    stitch_files = sorted(os.listdir(stitch_dir))
    print(f"    8_stitched/  ({len(stitch_files)} files)")
    for f in stitch_files:
        print(f"      {f}")


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else FOLDER_PATH
    main(folder)