#!/usr/bin/env python3
"""
OptiMesd Panel Stitcher — GUI
══════════════════════════════════════════════════════════════════
A desktop GUI wrapped around the original OptiMesd processing
pipeline (white-border removal → bright-strip extraction →
column-border crop → top-cut → stretch → stitch).

Workflow
────────
1. Choose an input folder that contains exactly 3 images
   (.jpg / .jpeg / .png). They are processed in alphabetical
   filename order (left → right of the final stitched panel).
2. Click "Process Images".
3. Review the "Normal" and "Inverted" stitched results in the
   preview panel below, alongside a live processing log.
4. Use the "Save As…" buttons to choose exactly where each
   result should be written to disk.

Run with:  python optimesd_gui.py
Requires:  opencv-python, numpy, scipy, pillow   (tkinter ships
           with most desktop Python installs; on some Linux
           distros install it separately, e.g. `sudo apt install
           python3-tk`)
"""

import os
import sys
import time
import queue
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2
from PIL import Image, ImageTk
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ══════════════════════════════════════════════════════════════════
#  DEFAULT CONFIG  (editable from the GUI's "Settings" panel)
# ══════════════════════════════════════════════════════════════════

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")

DEFAULTS = dict(
    # Stage-0 (white border removal)
    WHITE_BORDER_THRESH=200,
    WHITE_BORDER_COL_RATIO=0.85,
    WHITE_BORDER_ROW_RATIO=0.85,
    WHITE_BORDER_MIN_CONTENT=10,
    # Stage-1
    BRIGHT_THRESH=100,
    SMOOTH_WINDOW=7,
    VALLEY_SMOOTH_WINDOW=25,
    VALLEY_SEARCH=60,
    EXPAND_LEFT=0,
    EXPAND_RIGHT=50,
    # Stage-2
    GAUSS_KERNEL=31,
    GRADIENT_SIGMA_MULT=2,
    PEAK_DISTANCE=20,
    VALLEY_WINDOW=2,
    EXTRA_LEFT=0,
    EXTRA_RIGHT=0,
    # Stage-3 (stretch) -- editable in GUI
    SCALE_X=1.4,
    SCALE_Y=1.0,
    # Stage-4 (top cut per image position) -- editable in GUI
    TOP_CUT_1=24,
    TOP_CUT_2=12,
    TOP_CUT_3=25,
    # Stitching -- editable in GUI
    STITCH_MARGIN=2,
)


# ══════════════════════════════════════════════════════════════════
#  STAGE-0 : white border removal  (no I/O)
# ══════════════════════════════════════════════════════════════════

def remove_white_border(rgb: np.ndarray, cfg: dict, log) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    H, W = gray.shape

    white_row = (gray > cfg["WHITE_BORDER_THRESH"]).mean(axis=1) >= cfg["WHITE_BORDER_ROW_RATIO"]
    white_col = (gray > cfg["WHITE_BORDER_THRESH"]).mean(axis=0) >= cfg["WHITE_BORDER_COL_RATIO"]

    top, bottom, left, right = 0, H - 1, 0, W - 1
    while top < H and white_row[top]:
        top += 1
    while bottom > top and white_row[bottom]:
        bottom -= 1
    while left < W and white_col[left]:
        left += 1
    while right > left and white_col[right]:
        right -= 1

    if (bottom - top + 1) < cfg["WHITE_BORDER_MIN_CONTENT"]:
        log("  ⚠  remove_white_border: content height too small after crop – skipping.")
        return rgb
    if (right - left + 1) < cfg["WHITE_BORDER_MIN_CONTENT"]:
        log("  ⚠  remove_white_border: content width too small after crop – skipping.")
        return rgb

    cropped = rgb[top:bottom + 1, left:right + 1]
    removed_rows = top + (H - 1 - bottom)
    removed_cols = left + (W - 1 - right)
    if removed_rows > 0 or removed_cols > 0:
        log(f"  ✂  White border removed → {cropped.shape[1]}×{cropped.shape[0]}px")
    return cropped


# ══════════════════════════════════════════════════════════════════
#  STAGE-1 : bright-strip extraction  (no I/O)
# ══════════════════════════════════════════════════════════════════

def _find_bright_runs(col_means, cfg):
    smoothed = uniform_filter1d(col_means, size=cfg["SMOOTH_WINDOW"])
    bright = smoothed > cfg["BRIGHT_THRESH"]

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
            f"No bright strip found (threshold={cfg['BRIGHT_THRESH']}). "
            "Try lowering the Bright Threshold in Settings."
        )

    top2 = sorted(runs, key=lambda r: r[1] - r[0], reverse=True)[:2]
    top2.sort()
    (x0, x1) = top2[0]
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


def extract_strip(rgb, gray, cfg):
    col_means = gray.mean(axis=0).astype(np.float32)
    x0, x1, x0_2, x1_2 = _find_bright_runs(col_means, cfg)

    valley_smoothed = uniform_filter1d(col_means, size=cfg["VALLEY_SMOOTH_WINDOW"])
    left_edge = _nearest_valley(valley_smoothed, x0, "left", cfg["VALLEY_SEARCH"])
    right_edge = _nearest_valley(valley_smoothed, x1_2, "right", cfg["VALLEY_SEARCH"])
    left_edge = int(np.clip(left_edge - cfg["EXPAND_LEFT"], 0, rgb.shape[1] - 1))
    right_edge = int(np.clip(right_edge + cfg["EXPAND_RIGHT"], 0, rgb.shape[1] - 1))

    return rgb[:, left_edge:right_edge + 1]


# ══════════════════════════════════════════════════════════════════
#  STAGE-2 : column-border detection & crop  (no I/O)
# ══════════════════════════════════════════════════════════════════

def detect_and_crop(strip_rgb, cfg):
    gray = cv2.cvtColor(strip_rgb, cv2.COLOR_RGB2GRAY)
    profile = gray.mean(axis=0).astype(np.float32)

    k = cfg["GAUSS_KERNEL"] | 1  # GaussianBlur needs an odd kernel size
    profile_smooth = cv2.GaussianBlur(profile.reshape(1, -1), (1, k), 0).flatten()

    gradient = np.abs(np.gradient(profile_smooth))
    threshold = gradient.mean() + cfg["GRADIENT_SIGMA_MULT"] * gradient.std() - 1

    change_points, _ = find_peaks(gradient, height=threshold, distance=cfg["PEAK_DISTANCE"])

    valleys = []
    for peak in change_points:
        lo = max(0, peak - cfg["VALLEY_WINDOW"])
        hi = min(len(profile_smooth), peak + cfg["VALLEY_WINDOW"])
        valley = lo + np.argmin(profile_smooth[lo:hi])
        valleys.append(valley)

    if len(valleys) < 2:
        return strip_rgb

    x_left = max(0, min(valleys) + cfg["EXTRA_LEFT"])
    x_right = min(strip_rgb.shape[1] - 1, max(valleys) - cfg["EXTRA_RIGHT"])
    return strip_rgb[:, x_left:x_right]


# ══════════════════════════════════════════════════════════════════
#  STAGE-3 : stretch  (no I/O)
# ══════════════════════════════════════════════════════════════════

def stretch_panel(cropped_rgb, cfg):
    if cfg["SCALE_X"] == 1 and cfg["SCALE_Y"] == 1:
        return cropped_rgb
    h, w = cropped_rgb.shape[:2]
    new_w, new_h = max(1, int(w * cfg["SCALE_X"])), max(1, int(h * cfg["SCALE_Y"]))
    return cv2.resize(cropped_rgb, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)


# ══════════════════════════════════════════════════════════════════
#  PROCESS ONE IMAGE  (used by thread pool)
# ══════════════════════════════════════════════════════════════════

def process_segment(path, top_cut, cfg, log):
    stem = os.path.splitext(os.path.basename(path))[0]
    t0 = time.perf_counter()

    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    log(f"\n  [{stem}]  original size: {rgb.shape[1]}×{rgb.shape[0]}px")

    rgb = remove_white_border(rgb, cfg, log)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    strip = extract_strip(rgb, gray, cfg)
    cropped = detect_and_crop(strip, cfg)

    if top_cut > 0:
        cropped = cropped[top_cut:, :, :]

    stretched = stretch_panel(cropped, cfg)

    elapsed = time.perf_counter() - t0
    log(f"  ✔  {stem}  →  {stretched.shape[1]}×{stretched.shape[0]}px  [{elapsed:.3f}s]")
    return stem, stretched


# ══════════════════════════════════════════════════════════════════
#  STITCHING
# ══════════════════════════════════════════════════════════════════

def stitch_panels(panels, margin):
    max_h = max(p.shape[0] for p in panels)
    parts = []
    gap = np.zeros((max_h, margin, 3), dtype=np.uint8)

    for i, p in enumerate(panels):
        if p.shape[0] < max_h:
            pad = np.zeros((max_h - p.shape[0], p.shape[1], 3), dtype=np.uint8)
            p = np.vstack([p, pad])
        parts.append(p)
        if i < len(panels) - 1:
            parts.append(gap)

    return np.hstack(parts)


def mirror_panel(panel):
    return np.fliplr(panel)


# ══════════════════════════════════════════════════════════════════
#  IMAGE DISCOVERY
# ══════════════════════════════════════════════════════════════════

def discover_images(folder):
    """Returns a sorted (alphabetical, case-insensitive) list of image
    paths in `folder` whose extension is in VALID_EXTENSIONS."""
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")
    files = [
        f for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS
    ]
    files.sort(key=str.lower)
    return [os.path.join(folder, f) for f in files]


# ══════════════════════════════════════════════════════════════════
#  FULL PIPELINE  (callable independent of the GUI)
# ══════════════════════════════════════════════════════════════════

def run_pipeline(folder, cfg, log):
    """Runs the full pipeline on the 3 images found in `folder`.

    Returns: (ordered_stems, normal_rgb, inverted_rgb)
    """
    image_paths = discover_images(folder)

    if len(image_paths) != 3:
        raise ValueError(
            f"Expected exactly 3 images ({', '.join(VALID_EXTENSIONS)}) "
            f"in the selected folder, found {len(image_paths)}."
        )

    log(f"📁 Input folder : {folder}")
    for i, p in enumerate(image_paths, start=1):
        log(f"   {i}. {os.path.basename(p)}")

    top_cuts = [cfg["TOP_CUT_1"], cfg["TOP_CUT_2"], cfg["TOP_CUT_3"]]
    tasks = list(zip(image_paths, top_cuts))

    log(f"\n▶  Processing {len(tasks)} image(s) in parallel …")
    results = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        futures = {
            pool.submit(process_segment, path, cut, cfg, log): idx
            for idx, (path, cut) in enumerate(tasks)
        }
        for fut in futures:
            idx = futures[fut]
            results[idx] = fut.result()

    ordered_stems = [r[0] for r in results]
    panels = [r[1] for r in results]

    # ── Normal stitch ──────────────────────────────────────────────
    normal = stitch_panels(panels, cfg["STITCH_MARGIN"])
    log(f"\n▶  Normal stitched   → {normal.shape[1]}×{normal.shape[0]}px")

    # ── Inverted stitch ────────────────────────────────────────────
    mirrored = [mirror_panel(p) for p in panels]
    if len(mirrored) >= 3:
        mirrored[0], mirrored[-1] = mirrored[-1], mirrored[0]
    inverted = stitch_panels(mirrored, cfg["STITCH_MARGIN"])
    log(f"▶  Inverted stitched → {inverted.shape[1]}×{inverted.shape[0]}px")

    return ordered_stems, normal, inverted


# ══════════════════════════════════════════════════════════════════
#  GUI
# ══════════════════════════════════════════════════════════════════

class CombinedPreviewPanel(ttk.Frame):
    """
    A single panel holding BOTH the 'Normal' and 'Inverted' stitched
    results, side by side in one shared canvas — Normal on the left,
    Inverted directly to its right, both rendered at the same column
    width like a 2-column grid — with one pair of scrollbars shared
    by both. No separate sub-components/sub-panels are used for the
    two results.
    """


    LABEL_HEIGHT = 26         # space reserved above each column's title text
    COL_GAP = 24              # horizontal gap between the Normal / Inverted columns
    VIEW_HEIGHT = 520         # visible viewport height before vertical scrolling kicks in
    MIN_COL_WIDTH = 200       # never shrink a column smaller than this

    def __init__(self, parent, on_save_normal, on_save_inverted, on_save_both):
        super().__init__(parent, padding=6)
        self.normal_rgb = None
        self.inverted_rgb = None
        self._photo_normal = None   # keep references so they aren't GC'd
        self._photo_inverted = None

        # -- Header: title + the three save actions, all in one row --
        header = ttk.Frame(self)
        header.pack(fill="x")
        ttk.Label(header, text="Results", font=("Segoe UI", 10, "bold")).pack(side="left")
        self.info_label = ttk.Label(header, text="", foreground="#888")
        self.info_label.pack(side="left", padx=8)

        ttk.Button(header, text="💾 Save Both To Folder…", command=on_save_both).pack(side="right", padx=(4, 0))
        ttk.Button(header, text="💾 Save Inverted", command=on_save_inverted).pack(side="right", padx=4)
        ttk.Button(header, text="💾 Save Normal", command=on_save_normal).pack(side="right", padx=4)

        # -- Body: ONE canvas, ONE pair of scrollbars, two GRID columns --
        # Normal sits on the left, Inverted sits directly to its right,
        # both rendered at the SAME column width — a true 2-column grid,
        # not two separate stacked components.
        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, pady=(4, 0))

        self.canvas = tk.Canvas(body, bg="#2b2b2b", height=self.VIEW_HEIGHT,
                                 highlightthickness=1, highlightbackground="#444")
        vbar = ttk.Scrollbar(body, orient="vertical", command=self.canvas.yview)
        hbar = ttk.Scrollbar(body, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        vbar.grid(row=0, column=1, sticky="ns")
        hbar.grid(row=1, column=0, sticky="ew")
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=1)

        # Mouse-wheel scrolling while hovering the canvas (Win/Mac + Linux)
        self.canvas.bind("<Enter>", lambda e: self._bind_wheel())
        self.canvas.bind("<Leave>", lambda e: self._unbind_wheel())

        # Re-render at the new column width whenever the panel is resized,
        # so the grid stays responsive instead of leaving empty space.
        self.canvas.bind("<Configure>", self._on_resize)

        self._draw_placeholder()

    def _bind_wheel(self):
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)

    def _unbind_wheel(self):
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_mousewheel(self, event):
        if getattr(event, "num", None) == 4:
            self.canvas.yview_scroll(-3, "units")
        elif getattr(event, "num", None) == 5:
            self.canvas.yview_scroll(3, "units")
        else:
            self.canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

    def _on_resize(self, event):
        if self.normal_rgb is not None and self.inverted_rgb is not None:
            self._render()

    # -- internal drawing helpers ------------------------------------
    def _draw_placeholder(self):
        self.canvas.delete("all")
        self.canvas.create_text(
            10, self.VIEW_HEIGHT // 2, anchor="w",
            text="No output yet — process some images first.",
            fill="#888", font=("Segoe UI", 9)
        )
        self.canvas.configure(scrollregion=(0, 0, 0, self.VIEW_HEIGHT))
        self.info_label.configure(text="")

    def _col_width(self):
        canvas_w = self.canvas.winfo_width()
        if canvas_w < 10:
            canvas_w = 900  # not laid out yet — reasonable fallback
        col_w = (canvas_w - self.COL_GAP) // 2
        return max(self.MIN_COL_WIDTH, col_w)

    def _scaled_photo(self, rgb_array, target_w):
        h, w = rgb_array.shape[:2]
        scale = target_w / w
        new_h = max(1, int(h * scale))
        pil_img = Image.fromarray(rgb_array).resize((target_w, new_h), Image.LANCZOS)
        return ImageTk.PhotoImage(pil_img), new_h

    def set_images(self, normal_rgb, inverted_rgb):
        self.normal_rgb = normal_rgb
        self.inverted_rgb = inverted_rgb
        self._render()

    def _render(self):
        col_w = self._col_width()
        self._photo_normal, h_normal = self._scaled_photo(self.normal_rgb, col_w)
        self._photo_inverted, h_inverted = self._scaled_photo(self.inverted_rgb, col_w)
        nh, nw = self.normal_rgb.shape[:2]
        ih, iw = self.inverted_rgb.shape[:2]

        self.canvas.delete("all")

        # -- Column 1 : Normal (left) --------------------------------
        self.canvas.create_text(
            8, self.LABEL_HEIGHT // 2, anchor="w",
            text=f"Normal  —  {nw} × {nh} px (full resolution saved)",
            fill="#d4d4d4", font=("Segoe UI", 9, "bold")
        )
        self.canvas.create_image(0, self.LABEL_HEIGHT, anchor="nw", image=self._photo_normal)

        # -- Column 2 : Inverted (right, right next to Normal) -------
        x_inverted = col_w + self.COL_GAP
        self.canvas.create_text(
            x_inverted + 8, self.LABEL_HEIGHT // 2, anchor="w",
            text=f"Inverted  —  {iw} × {ih} px (full resolution saved)",
            fill="#d4d4d4", font=("Segoe UI", 9, "bold")
        )
        self.canvas.create_image(x_inverted, self.LABEL_HEIGHT, anchor="nw", image=self._photo_inverted)

        # -- Divider between the two grid columns ----------------------
        max_h = max(h_normal, h_inverted)
        x_div = col_w + self.COL_GAP // 2
        self.canvas.create_line(x_div, 0, x_div, self.LABEL_HEIGHT + max_h, fill="#444", dash=(3, 2))

        total_w = x_inverted + col_w
        total_h = self.LABEL_HEIGHT + max_h
        self.canvas.configure(scrollregion=(0, 0, total_w, total_h))
        self.info_label.configure(
            text="Normal (left) & Inverted (right) side by side — scroll for full resolution"
        )

    def clear(self):
        self.normal_rgb = None
        self.inverted_rgb = None
        self._photo_normal = None
        self._photo_inverted = None
        self._draw_placeholder()


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("OptiMesd — Panel Stitcher")
        self.geometry("1180x860")
        self.minsize(900, 650)

        try:
            self.style = ttk.Style(self)
            if "clam" in self.style.theme_names():
                self.style.theme_use("clam")
        except Exception:
            pass

        self.cfg = dict(DEFAULTS)
        self.msg_queue = queue.Queue()
        self.worker_thread = None
        self.normal_rgb = None
        self.inverted_rgb = None

        self._build_widgets()
        self.after(100, self._poll_queue)

    # ── UI construction ────────────────────────────────────────────
    def _build_widgets(self):
        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)

        # -- Input folder row --------------------------------------
        in_frame = ttk.LabelFrame(root, text="1. Input directory (must contain exactly 3 images: .jpg/.jpeg/.png)")
        in_frame.pack(fill="x", pady=(0, 8))

        self.folder_var = tk.StringVar()
        entry = ttk.Entry(in_frame, textvariable=self.folder_var)
        entry.pack(side="left", fill="x", expand=True, padx=(8, 4), pady=8)
        ttk.Button(in_frame, text="Browse", command=self._browse_folder).pack(side="left", padx=4, pady=8)
        self.process_btn = ttk.Button(in_frame, text=" Process Images", command=self._start_processing)
        self.process_btn.pack(side="left", padx=(4, 8), pady=8)

        # -- Settings (collapsible-ish, just a labeled frame) -------
        self._build_settings(root)

        # -- Progress bar --------------------------------------------
        self.progress = ttk.Progressbar(root, mode="indeterminate")
        self.progress.pack(fill="x", pady=(0, 8))

        # -- Main split: log (left) / previews (right) ---------------
        paned = ttk.PanedWindow(root, orient="vertical")
        paned.pack(fill="both", expand=True)

        log_frame = ttk.LabelFrame(paned, text="Processing log")
        self.log_text = tk.Text(log_frame, height=10, bg="#1e1e1e", fg="#d4d4d4",
                                 insertbackground="#d4d4d4", font=("Consolas", 9), wrap="word")
        log_scroll = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set, state="disabled")
        self.log_text.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)
        log_scroll.pack(side="right", fill="y", pady=8)
        paned.add(log_frame, weight=2)

        out_frame = ttk.LabelFrame(paned, text="2. Output preview")
        self.preview_panel = CombinedPreviewPanel(
            out_frame, self._save_normal, self._save_inverted, self._save_both
        )
        self.preview_panel.pack(fill="both", expand=True)

        paned.add(out_frame, weight=3)

        # -- Status bar ------------------------------------------------
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(root, textvariable=self.status_var, anchor="w",
                  foreground="#555").pack(fill="x", pady=(6, 0))

    def _build_settings(self, root):
        frame = ttk.LabelFrame(root, text="Settings (advanced — defaults work for most images)")
        frame.pack(fill="x", pady=(0, 8))

        self.setting_vars = {}

        def add_field(parent, key, label, col, width=6):
            ttk.Label(parent, text=label).grid(row=0, column=col * 2, sticky="e", padx=(10, 2), pady=6)
            var = tk.StringVar(value=str(self.cfg[key]))
            ttk.Entry(parent, textvariable=var, width=width).grid(
                row=0, column=col * 2 + 1, sticky="w", pady=6)
            self.setting_vars[key] = var

        add_field(frame, "TOP_CUT_1", "Top cut #1 (px):", 0)
        add_field(frame, "TOP_CUT_2", "Top cut #2 (px):", 1)
        add_field(frame, "TOP_CUT_3", "Top cut #3 (px):", 2)
        add_field(frame, "SCALE_X", "Scale X:", 3)
        add_field(frame, "SCALE_Y", "Scale Y:", 4)
        add_field(frame, "STITCH_MARGIN", "Stitch margin (px):", 5)

        ttk.Button(frame, text="Reset Defaults", command=self._reset_settings).grid(
            row=0, column=12, padx=10, pady=6)

    def _reset_settings(self):
        for key, var in self.setting_vars.items():
            var.set(str(DEFAULTS[key]))

    def _collect_settings(self):
        """Reads the editable settings fields into self.cfg, validating types."""
        cfg = dict(self.cfg)
        try:
            cfg["TOP_CUT_1"] = int(float(self.setting_vars["TOP_CUT_1"].get()))
            cfg["TOP_CUT_2"] = int(float(self.setting_vars["TOP_CUT_2"].get()))
            cfg["TOP_CUT_3"] = int(float(self.setting_vars["TOP_CUT_3"].get()))
            cfg["SCALE_X"] = float(self.setting_vars["SCALE_X"].get())
            cfg["SCALE_Y"] = float(self.setting_vars["SCALE_Y"].get())
            cfg["STITCH_MARGIN"] = int(float(self.setting_vars["STITCH_MARGIN"].get()))
        except ValueError:
            raise ValueError("Settings must be numeric. Please check the Settings panel.")
        return cfg

    # ── Folder browsing ─────────────────────────────────────────────
    def _browse_folder(self):
        folder = filedialog.askdirectory(title="Select input directory (3 images)")
        if folder:
            self.folder_var.set(folder)

    # ── Logging helper (thread-safe via queue) ─────────────────────
    def _log(self, message):
        self.msg_queue.put(("log", message))

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.msg_queue.get_nowait()
                if kind == "log":
                    self.log_text.configure(state="normal")
                    self.log_text.insert("end", payload + "\n")
                    self.log_text.see("end")
                    self.log_text.configure(state="disabled")
                elif kind == "done":
                    self._on_processing_done(*payload)
                elif kind == "error":
                    self._on_processing_error(payload)
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    # ── Processing (runs in a background thread) ───────────────────
    def _start_processing(self):
        folder = self.folder_var.get().strip()
        if not folder:
            messagebox.showwarning("No folder selected", "Please choose an input directory first.")
            return
        if self.worker_thread and self.worker_thread.is_alive():
            return

        try:
            cfg = self._collect_settings()
        except ValueError as e:
            messagebox.showerror("Invalid settings", str(e))
            return

        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")
        self.preview_panel.clear()
        self.normal_rgb = None
        self.inverted_rgb = None

        self.process_btn.configure(state="disabled")
        self.progress.start(12)
        self.status_var.set("Processing…")

        self.worker_thread = threading.Thread(
            target=self._worker, args=(folder, cfg), daemon=True
        )
        self.worker_thread.start()

    def _worker(self, folder, cfg):
        try:
            t0 = time.perf_counter()
            stems, normal, inverted = run_pipeline(folder, cfg, self._log)
            elapsed = time.perf_counter() - t0
            self._log(f"\n✅  Done in {elapsed:.3f}s.")
            self.msg_queue.put(("done", (stems, normal, inverted)))
        except Exception as e:
            self._log("\n❌  ERROR: " + str(e))
            self._log(traceback.format_exc())
            self.msg_queue.put(("error", str(e)))

    def _on_processing_done(self, stems, normal, inverted):
        self.progress.stop()
        self.process_btn.configure(state="normal")
        self.normal_rgb = normal
        self.inverted_rgb = inverted
        self.preview_panel.set_images(normal, inverted)
        self.status_var.set(f"Done — processed: {', '.join(stems)}")

    def _on_processing_error(self, error_message):
        self.progress.stop()
        self.process_btn.configure(state="normal")
        self.status_var.set("Failed — see log for details.")
        messagebox.showerror("Processing failed", error_message)

    # ── Saving ──────────────────────────────────────────────────────
    def _save_array(self, rgb_array, initial_name):
        path = filedialog.asksaveasfilename(
            title="Save image as…",
            defaultextension=".png",
            initialfile=initial_name,
            filetypes=[("PNG image", "*.png"), ("JPEG image", "*.jpg"), ("All files", "*.*")],
        )
        if not path:
            return
        bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        ok = cv2.imwrite(path, bgr)
        if ok:
            self.status_var.set(f"Saved → {path}")
        else:
            messagebox.showerror("Save failed", f"Could not write file:\n{path}")

    def _save_normal(self):
        if self.normal_rgb is None:
            messagebox.showinfo("Nothing to save", "Process some images first.")
            return
        self._save_array(self.normal_rgb, "stitched_normal.png")

    def _save_inverted(self):
        if self.inverted_rgb is None:
            messagebox.showinfo("Nothing to save", "Process some images first.")
            return
        self._save_array(self.inverted_rgb, "stitched_inverted.png")

    def _save_both(self):
        if self.normal_rgb is None or self.inverted_rgb is None:
            messagebox.showinfo("Nothing to save", "Process some images first.")
            return
        folder = filedialog.askdirectory(title="Choose a folder to save both results into")
        if not folder:
            return
        n_path = os.path.join(folder, "stitched_normal.png")
        i_path = os.path.join(folder, "stitched_inverted.png")
        ok1 = cv2.imwrite(n_path, cv2.cvtColor(self.normal_rgb, cv2.COLOR_RGB2BGR))
        ok2 = cv2.imwrite(i_path, cv2.cvtColor(self.inverted_rgb, cv2.COLOR_RGB2BGR))
        if ok1 and ok2:
            self.status_var.set(f"Saved both → {folder}")
        else:
            messagebox.showerror("Save failed", "Could not write one or both files.")


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()