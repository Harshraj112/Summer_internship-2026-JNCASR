#!/usr/bin/env python3
"""
pipeline_gui.py  –  GUI for the image-processing pipeline
==========================================================

Tab 1 : 16 individual stitched images (one per input image)
Tab 2 : Final composite – zoomable / pannable

Touchpad gestures (Tab 2):
  Two-finger scroll          → pan (vertical / horizontal)
  Ctrl + two-finger scroll   → zoom in/out  (Windows pinch = Ctrl+wheel)
  Click + drag               → pan
  Zoom buttons / +/-         → zoom

Usage:
    python pipeline_gui.py
Requires api_final.py in the same directory (or on PYTHONPATH).
"""

from __future__ import annotations

import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
from typing import Optional

from PIL import Image, ImageTk
import numpy as np

try:
    import api as api
except ImportError:
    api = None


# ══════════════════════════════════════════════════════════════════════════════
#  DESIGN TOKENS
# ══════════════════════════════════════════════════════════════════════════════

BG          = "#0E1117"
PANEL       = "#161B22"
BORDER      = "#30363D"
ACCENT      = "#58A6FF"
ACCENT_DARK = "#1F6FEB"
SUCCESS     = "#3FB950"
ERROR       = "#F85149"
TEXT        = "#E6EDF3"
TEXT_DIM    = "#8B949E"
CELL_BG     = "#1C2128"

FONT_TITLE  = ("Segoe UI", 13, "bold")
FONT_LABEL  = ("Segoe UI", 9)
FONT_BOLD   = ("Segoe UI", 9, "bold")
FONT_SMALL  = ("Segoe UI", 8)
FONT_STATUS = ("Segoe UI", 9)
FONT_TAB    = ("Segoe UI", 10, "bold")
FONT_HINT   = ("Segoe UI", 8)

# ── Platform detection ────────────────────────────────────────────────────────
_PLATFORM = sys.platform          # "win32" | "darwin" | "linux"
_IS_WIN   = _PLATFORM == "win32"
_IS_MAC   = _PLATFORM == "darwin"
_IS_LIN   = _PLATFORM.startswith("linux")


# ══════════════════════════════════════════════════════════════════════════════
#  CROSS-PLATFORM DELTA NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════

def _raw_delta(event) -> float:
    """Return a signed float: positive = scroll-up / zoom-in."""
    if event.num == 4:   return  1.0
    if event.num == 5:   return -1.0
    if _IS_MAC:
        return float(event.delta)
    return event.delta / 120.0


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER WIDGETS
# ══════════════════════════════════════════════════════════════════════════════

class FlatButton(tk.Button):
    def __init__(self, parent, text, command, primary=False, **kw):
        bg_c  = ACCENT if primary else PANEL
        fg_c  = "#0D1117" if primary else TEXT
        hov_c = ACCENT_DARK if primary else BORDER
        super().__init__(
            parent, text=text, command=command,
            bg=bg_c, fg=fg_c, activebackground=hov_c, activeforeground=fg_c,
            relief="flat", cursor="hand2", padx=14, pady=6,
            font=FONT_BOLD if primary else FONT_LABEL, **kw,
        )
        self._bg, self._hov = bg_c, hov_c
        self.bind("<Enter>", lambda _: self.config(bg=self._hov))
        self.bind("<Leave>", lambda _: self.config(bg=self._bg))


class Spinbox2(tk.Frame):
    def __init__(self, parent, label, lo, hi, default, **kw):
        super().__init__(parent, bg=PANEL, **kw)
        tk.Label(self, text=label, bg=PANEL, fg=TEXT_DIM, font=FONT_SMALL
                 ).pack(side="left", padx=(0, 5))
        self.var  = tk.IntVar(value=default)
        self.spin = tk.Spinbox(
            self, from_=lo, to=hi, textvariable=self.var, width=4,
            bg=BG, fg=TEXT, insertbackground=TEXT, buttonbackground=BORDER,
            relief="flat", highlightthickness=1, highlightcolor=ACCENT,
            highlightbackground=BORDER, font=FONT_LABEL,
            disabledbackground=BG, disabledforeground=TEXT_DIM,
        )
        self.spin.pack(side="left")

    @property
    def value(self): return int(self.var.get())
    def set_state(self, s): self.spin.config(state=s)


class ThumbCell(tk.Frame):
    W, H = 180, 140

    def __init__(self, parent, idx, **kw):
        super().__init__(parent, bg=CELL_BG, highlightthickness=1,
                         highlightbackground=BORDER, **kw)
        hdr = tk.Frame(self, bg=PANEL)
        hdr.pack(fill="x")
        tk.Label(hdr, text=f" {idx} ", bg=PANEL, fg=TEXT_DIM,
                 font=FONT_SMALL).pack(side="left", padx=3, pady=2)
        self.canvas = tk.Canvas(self, width=self.W, height=self.H,
                                bg=CELL_BG, highlightthickness=0)
        self.canvas.pack()
        self._photo = None
        self._placeholder()

    def _placeholder(self):
        self.canvas.delete("all")
        self.canvas.create_text(self.W // 2, self.H // 2, text="—",
                                fill=BORDER, font=("Segoe UI", 22))

    def show(self, pil_img: Image.Image):
        img = pil_img.copy()
        img.thumbnail((self.W, self.H), Image.LANCZOS)
        bg = Image.new("RGB", (self.W, self.H),
                       tuple(int(CELL_BG[i:i+2], 16) for i in (1, 3, 5)))
        bg.paste(img, ((self.W - img.width) // 2, (self.H - img.height) // 2))
        self._photo = ImageTk.PhotoImage(bg)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)
        self.config(highlightbackground=ACCENT)

    def reset(self):
        self._photo = None
        self.config(highlightbackground=BORDER)
        self._placeholder()

    def error(self):
        self.canvas.delete("all")
        self.canvas.create_text(self.W // 2, self.H // 2, text="✕",
                                fill=ERROR, font=("Segoe UI", 22))
        self.config(highlightbackground=ERROR)


# ══════════════════════════════════════════════════════════════════════════════
#  SMOOTH SCROLL HELPER
# ══════════════════════════════════════════════════════════════════════════════

class _Accumulator:
    """Collect fractional scroll steps; emit integer units when threshold met."""
    def __init__(self, scale: float = 1.0):
        self._acc   = 0.0
        self._scale = scale

    def feed(self, raw: float) -> int:
        self._acc += raw * self._scale
        units = int(self._acc)
        self._acc -= units
        return units

    def reset(self):
        self._acc = 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  ZOOMABLE CANVAS  (Tab 2)
#  — Smooth zoom: NEAREST resampling during gesture, LANCZOS after settle
#  — Mirror: horizontal flip toggled by toolbar button
# ══════════════════════════════════════════════════════════════════════════════

class ZoomCanvas(tk.Frame):
    """
    Smooth pan + zoom image viewer with mirror support.

    Gestures
    --------
    Two-finger scroll (vertical)    → pan up / down
    Two-finger scroll (horizontal)  → pan left / right  [Shift+scroll on Win]
    Ctrl + two-finger scroll        → zoom in / out     (Windows pinch)
    Click + drag                    → pan
    Zoom buttons                    → zoom
    """

    ZOOM_MIN  = 0.02
    ZOOM_MAX  = 12.0

    _PAN_SCALE    = 6.0    # canvas units per raw-delta-unit (pan)
    _ZOOM_PER_UNIT = 0.10  # zoom factor per raw-delta-unit (smooth)

    # ms to wait after last zoom event before doing the HQ LANCZOS redraw
    _HQ_DELAY_MS  = 120

    def __init__(self, parent, **kw):
        super().__init__(parent, bg=BG, **kw)

        self._pil_orig: Optional[Image.Image] = None   # original, never modified
        self._pil_src:  Optional[Image.Image] = None   # current (may be mirrored)
        self._mirrored  = False
        self._scale     = 1.0
        self._photo: Optional[ImageTk.PhotoImage] = None

        # Accumulators for smooth sub-unit scrolling
        self._pan_y_acc = _Accumulator(self._PAN_SCALE)
        self._pan_x_acc = _Accumulator(self._PAN_SCALE)

        # Drag state
        self._drag_last: Optional[tuple[int, int]] = None

        # HQ redraw debounce timer id
        self._hq_timer: Optional[str] = None

        self._build_widgets()
        self._bind_events()

    # ── Widget construction ──────────────────────────────────────────────────

    def _build_widgets(self):
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(self, bg=BG, cursor="fleur",
                                highlightthickness=0)
        vsb = ttk.Scrollbar(self, orient="vertical",   command=self.canvas.yview)
        hsb = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        # Toolbar
        bar = tk.Frame(self, bg=PANEL)
        bar.grid(row=2, column=0, columnspan=2, sticky="ew")

        for txt, cmd in [("＋", self._zoom_in), ("－", self._zoom_out),
                         ("⊡ Fit", self._zoom_fit), ("1:1", self._zoom_1)]:
            FlatButton(bar, txt, cmd).pack(side="left", padx=4, pady=4)

        tk.Frame(bar, bg=BORDER, width=1).pack(side="left", fill="y", padx=6, pady=4)

        # ── Mirror button (replaces old Invert) ──────────────────────────────
        self._mirror_btn = FlatButton(bar, "↔  Mirror", self._toggle_mirror)
        self._mirror_btn.pack(side="left", padx=4, pady=4)

        self._zoom_lbl = tk.Label(bar, text="—", bg=PANEL, fg=TEXT_DIM,
                                  font=FONT_SMALL)
        self._zoom_lbl.pack(side="right", padx=10)

        hint = ("Scroll = pan   •   Ctrl+scroll = zoom   •   Drag = pan"
                if _IS_WIN else
                "Two-finger swipe = pan   •   Ctrl+swipe = zoom   •   Drag = pan")
        tk.Label(bar, text=hint, bg=PANEL, fg=TEXT_DIM,
                 font=FONT_HINT).pack(side="right", padx=16)

    # ── Event binding ────────────────────────────────────────────────────────

    def _bind_events(self):
        c = self.canvas

        # Pan: plain scroll (vertical)
        c.bind("<MouseWheel>",         self._pan_y_win)
        c.bind("<Button-4>",           self._pan_y_lin)
        c.bind("<Button-5>",           self._pan_y_lin)

        # Pan: horizontal (Shift + scroll)
        c.bind("<Shift-MouseWheel>",   self._pan_x_win)
        c.bind("<Shift-Button-4>",     self._pan_x_lin)
        c.bind("<Shift-Button-5>",     self._pan_x_lin)

        # Zoom: Ctrl + scroll / Windows pinch
        c.bind("<Control-MouseWheel>", self._zoom_wheel_win)
        c.bind("<Control-Button-4>",   self._zoom_wheel_lin)
        c.bind("<Control-Button-5>",   self._zoom_wheel_lin)

        # macOS trackpad pinch
        try:
            c.bind("<Magnify>", self._magnify_mac)
        except Exception:
            pass

        # Drag to pan
        c.bind("<ButtonPress-1>",   self._on_drag_start)
        c.bind("<B1-Motion>",       self._on_drag_move)
        c.bind("<ButtonRelease-1>", self._on_drag_end)

        # Resize
        c.bind("<Configure>", lambda _: self._redraw())

    # ── Gesture handlers ─────────────────────────────────────────────────────

    def _pan_y_win(self, event):
        if event.state & 0x0004:          # Ctrl held → zoom
            self._do_zoom(_raw_delta(event), event.x, event.y)
            return
        units = self._pan_y_acc.feed(_raw_delta(event))
        if units:
            self.canvas.yview_scroll(-units, "units")

    def _pan_y_lin(self, event):
        units = self._pan_y_acc.feed(_raw_delta(event))
        if units:
            self.canvas.yview_scroll(-units, "units")

    def _pan_x_win(self, event):
        units = self._pan_x_acc.feed(_raw_delta(event))
        if units:
            self.canvas.xview_scroll(-units, "units")

    def _pan_x_lin(self, event):
        units = self._pan_x_acc.feed(_raw_delta(event))
        if units:
            self.canvas.xview_scroll(-units, "units")

    def _zoom_wheel_win(self, event):
        self._do_zoom(_raw_delta(event), event.x, event.y)

    def _zoom_wheel_lin(self, event):
        self._do_zoom(_raw_delta(event), event.x, event.y)

    def _magnify_mac(self, event):
        if hasattr(event, "delta") and event.delta:
            self._zoom_at(1.0 + float(event.delta), event.x, event.y)

    # ── Drag ─────────────────────────────────────────────────────────────────

    def _on_drag_start(self, event):
        self._drag_last = (event.x, event.y)
        self._pan_y_acc.reset()
        self._pan_x_acc.reset()

    def _on_drag_move(self, event):
        if self._drag_last is None:
            return
        dx = event.x - self._drag_last[0]
        dy = event.y - self._drag_last[1]
        self._drag_last = (event.x, event.y)
        sr = self.canvas.cget("scrollregion")
        if not sr:
            return
        try:
            _, _, sw, sh = (float(v) for v in str(sr).split())
        except Exception:
            return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if sw > cw:
            self.canvas.xview_moveto(self.canvas.xview()[0] - dx / sw)
        if sh > ch:
            self.canvas.yview_moveto(self.canvas.yview()[0] - dy / sh)

    def _on_drag_end(self, _event):
        self._drag_last = None

    # ── Zoom helpers ─────────────────────────────────────────────────────────

    def _do_zoom(self, raw_delta: float, cx: int, cy: int):
        """Zoom centred on canvas pixel (cx, cy); fast preview + deferred HQ."""
        factor = 1.0 + raw_delta * self._ZOOM_PER_UNIT
        self._zoom_at(factor, cx, cy)

    def _zoom_at(self, factor: float, cx: int, cy: int):
        if self._pil_src is None:
            return
        old = self._scale
        new = max(self.ZOOM_MIN, min(self.ZOOM_MAX, old * factor))
        if abs(new - old) < 1e-6:
            return

        # Image-space point under cursor
        img_x = self.canvas.canvasx(cx) / old
        img_y = self.canvas.canvasy(cy) / old

        self._scale = new
        # Fast NEAREST preview so the UI stays fluid
        self._redraw(fast=True)

        # Scroll to keep the same image point under cursor
        iw = self._pil_src.width  * new
        ih = self._pil_src.height * new
        if iw > 0:
            self.canvas.xview_moveto(max(0.0, (img_x * new - cx) / iw))
        if ih > 0:
            self.canvas.yview_moveto(max(0.0, (img_y * new - cy) / ih))

        # Schedule a high-quality redraw after the user pauses
        self._schedule_hq()

    def _schedule_hq(self):
        """Debounce: cancel pending HQ redraw and restart the timer."""
        if self._hq_timer is not None:
            self.after_cancel(self._hq_timer)
        self._hq_timer = self.after(self._HQ_DELAY_MS, self._hq_redraw)

    def _hq_redraw(self):
        """High-quality (LANCZOS) redraw after the zoom gesture settles."""
        self._hq_timer = None
        self._redraw(fast=False)

    def _zoom_in(self):
        self._zoom_at(1.25, self.canvas.winfo_width()  // 2,
                             self.canvas.winfo_height() // 2)

    def _zoom_out(self):
        self._zoom_at(0.80, self.canvas.winfo_width()  // 2,
                             self.canvas.winfo_height() // 2)

    def _zoom_1(self):
        if abs(self._scale - 1.0) < 1e-6:
            return
        self._scale = 1.0
        self._redraw()

    def _zoom_fit(self):
        if self._pil_src is None:
            return
        self.update_idletasks()
        cw = max(self.canvas.winfo_width(),  1)
        ch = max(self.canvas.winfo_height(), 1)
        sx = cw / max(self._pil_src.width,  1)
        sy = ch / max(self._pil_src.height, 1)
        self._scale = max(self.ZOOM_MIN, min(self.ZOOM_MAX, min(sx, sy)))
        self._redraw()
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)

    # ── Mirror (horizontal flip) ─────────────────────────────────────────────

    def _toggle_mirror(self):
        """Flip the image left-right (mirror effect) and toggle back."""
        if self._pil_orig is None:
            return
        self._mirrored = not self._mirrored
        if self._mirrored:
            # Horizontal flip — true mirror
            self._pil_src = self._pil_orig.transpose(Image.FLIP_LEFT_RIGHT)
            self._mirror_btn.config(
                text="↔  Restore",
                bg=ACCENT, fg="#0D1117",
            )
            self._mirror_btn._bg  = ACCENT
            self._mirror_btn._hov = ACCENT_DARK
        else:
            self._pil_src = self._pil_orig.copy()
            self._mirror_btn.config(
                text="↔  Mirror",
                bg=PANEL, fg=TEXT,
            )
            self._mirror_btn._bg  = PANEL
            self._mirror_btn._hov = BORDER
        self._redraw()

    # ── Render ───────────────────────────────────────────────────────────────

    def _redraw(self, fast: bool = False):
        """
        Redraw the canvas image.
        fast=True  → NEAREST (instant, used during gesture)
        fast=False → LANCZOS  (high quality, used when settled)
        """
        if self._pil_src is None:
            return
        w = max(1, int(self._pil_src.width  * self._scale))
        h = max(1, int(self._pil_src.height * self._scale))
        resample    = Image.NEAREST if fast else Image.LANCZOS
        resized     = self._pil_src.resize((w, h), resample)
        self._photo = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)
        self.canvas.configure(scrollregion=(0, 0, w, h))
        self._zoom_lbl.config(text=f"{self._scale * 100:.0f}%")

    # ── Public ───────────────────────────────────────────────────────────────

    def load(self, pil_img: Image.Image):
        self._pil_orig  = pil_img.convert("RGB")
        self._pil_src   = self._pil_orig.copy()
        self._mirrored  = False
        self._mirror_btn.config(text="↔  Mirror", bg=PANEL, fg=TEXT)
        self._mirror_btn._bg  = PANEL
        self._mirror_btn._hov = BORDER
        self._scale = 1.0
        self._pan_y_acc.reset()
        self._pan_x_acc.reset()
        self.update_idletasks()
        self._zoom_fit()

    def clear(self):
        self._pil_orig  = None
        self._pil_src   = None
        self._photo     = None
        self._mirrored  = False
        self.canvas.delete("all")
        self.canvas.configure(scrollregion=(0, 0, 0, 0))
        self._zoom_lbl.config(text="—")


# ══════════════════════════════════════════════════════════════════════════════
#  SCROLLABLE GRID  (Tab 1)
# ══════════════════════════════════════════════════════════════════════════════

class ScrollGrid(tk.Frame):
    """Scrollable 4-column thumbnail grid with proper touchpad support."""

    def __init__(self, parent, **kw):
        super().__init__(parent, bg=BG, **kw)

        self._scroll_acc = _Accumulator(scale=3.0)

        self.canvas = tk.Canvas(self, bg=BG, highlightthickness=0)
        vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.inner = tk.Frame(self.canvas, bg=BG)
        self._win  = self.canvas.create_window((0, 0), window=self.inner,
                                               anchor="nw", tags="inner")
        self.inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>",
                         lambda e: self.canvas.itemconfig("inner", width=e.width))

        self.canvas.bind("<Enter>", self._attach_wheel)
        self.canvas.bind("<Leave>", self._detach_wheel)
        self.inner.bind("<Enter>",  self._attach_wheel)
        self.inner.bind("<Leave>",  self._detach_wheel)
        self._wheel_bound = False

    def _on_inner_configure(self, _=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _attach_wheel(self, _=None):
        if self._wheel_bound:
            return
        self._wheel_bound = True
        self.bind_all("<MouseWheel>", self._on_wheel_win)
        self.bind_all("<Button-4>",   self._on_wheel_lin)
        self.bind_all("<Button-5>",   self._on_wheel_lin)

    def _detach_wheel(self, _=None):
        if not self._wheel_bound:
            return
        self._wheel_bound = False
        self.unbind_all("<MouseWheel>")
        self.unbind_all("<Button-4>")
        self.unbind_all("<Button-5>")

    def _on_wheel_win(self, event):
        units = self._scroll_acc.feed(_raw_delta(event))
        if units:
            self.canvas.yview_scroll(-units, "units")

    def _on_wheel_lin(self, event):
        units = self._scroll_acc.feed(_raw_delta(event))
        if units:
            self.canvas.yview_scroll(-units, "units")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

class PipelineGUI(tk.Tk):
    NUM_IMAGES = 16
    GRID_COLS  = 4

    def __init__(self):
        super().__init__()
        self.title("Image Pipeline Viewer")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.minsize(860, 700)

        self._folder: Optional[str] = None
        self._running = False
        self._cells:  list[ThumbCell] = []
        self._result: Optional[dict]  = None

        self._build_ui()
        self._apply_ttk_style()
        self._center()

    # ── UI ───────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Header
        hdr = tk.Frame(self, bg=PANEL, pady=10)
        hdr.pack(fill="x")
        tk.Label(hdr, text=" Image Pipeline Viewer",
                 bg=PANEL, fg=TEXT, font=FONT_TITLE).pack(side="left", padx=16)

        # Controls
        ctrl = tk.Frame(self, bg=BG, pady=10)
        ctrl.pack(fill="x", padx=16)

        frow = tk.Frame(ctrl, bg=BG)
        frow.pack(fill="x", pady=(0, 8))
        tk.Label(frow, text="Input folder:", bg=BG, fg=TEXT_DIM,
                 font=FONT_LABEL).pack(side="left")
        self._folder_lbl = tk.Label(frow, text="  No folder selected",
                                    bg=BG, fg=TEXT_DIM, font=FONT_LABEL, anchor="w")
        self._folder_lbl.pack(side="left", fill="x", expand=True, padx=6)
        FlatButton(frow, "Browse…", self._browse).pack(side="right")

        prow = tk.Frame(ctrl, bg=BG)
        prow.pack(fill="x")
        self._rows = Spinbox2(prow, "Rows (m):", 1, 4, 3)
        self._rows.pack(side="left", padx=(0, 12))
        self._cols = Spinbox2(prow, "Cols (n):", 1, 4, 3)
        self._cols.pack(side="left", padx=(0, 20))
        self._run_btn  = FlatButton(prow, " Run Pipeline",   self._run, primary=True)
        self._run_btn.pack(side="left")
        self._save_btn = FlatButton(prow, "  Save Composite", self._save)
        self._save_btn.pack(side="left", padx=(8, 0))
        self._save_btn.config(state="disabled")

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

        # Notebook
        nb = ttk.Notebook(self, style="Dark.TNotebook")
        nb.pack(fill="both", expand=True)

        # Tab 1
        tab1 = tk.Frame(nb, bg=BG)
        nb.add(tab1, text="  16 Stitched Images  ")
        self._scroll_grid = ScrollGrid(tab1)
        self._scroll_grid.pack(fill="both", expand=True)
        self._build_thumb_grid()

        # Tab 2
        tab2 = tk.Frame(nb, bg=BG)
        nb.add(tab2, text="  Final Composite (all 16)  ")

        self._composite_ph = tk.Label(
            tab2,
            text="Run the pipeline — the merged composite of all 16 images\n"
                 "will appear here (Output/<folder>/final_image.jpeg).\n\n"
                 "Scroll = pan  •  Ctrl+scroll = zoom  •  Drag = pan",
            bg=BG, fg=TEXT_DIM, font=("Segoe UI", 11), justify="center",
        )
        self._composite_ph.pack(expand=True)

        self._zoom_canvas = ZoomCanvas(tab2)

        # Status bar
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")
        sbar = tk.Frame(self, bg=PANEL, pady=6)
        sbar.pack(fill="x")
        self._status_var = tk.StringVar(value="Ready — select a folder and press Run.")
        self._status_lbl = tk.Label(sbar, textvariable=self._status_var,
                                    bg=PANEL, fg=TEXT_DIM, font=FONT_STATUS, anchor="w")
        self._status_lbl.pack(side="left", padx=14)
        self._progress = ttk.Progressbar(sbar, length=180, mode="indeterminate")
        self._progress.pack(side="right", padx=14)

        self._nb = nb

    def _build_thumb_grid(self):
        for w in self._scroll_grid.inner.winfo_children():
            w.destroy()
        self._cells.clear()
        for i in range(1, self.NUM_IMAGES + 1):
            r, c = divmod(i - 1, self.GRID_COLS)
            cell = ThumbCell(self._scroll_grid.inner, i)
            cell.grid(row=r, column=c, padx=5, pady=5, sticky="nsew")
            self._cells.append(cell)
        for c in range(self.GRID_COLS):
            self._scroll_grid.inner.columnconfigure(c, weight=1)

    def _apply_ttk_style(self):
        s = ttk.Style(self)
        try: s.theme_use("clam")
        except Exception: pass
        s.configure("Dark.TNotebook",        background=BG,    borderwidth=0)
        s.configure("Dark.TNotebook.Tab",    background=PANEL, foreground=TEXT_DIM,
                    padding=[14, 6], font=FONT_TAB)
        s.map("Dark.TNotebook.Tab",
              background=[("selected", BG)],
              foreground=[("selected", ACCENT)])
        s.configure("TProgressbar", troughcolor=PANEL, background=ACCENT,
                    bordercolor=BORDER, lightcolor=ACCENT, darkcolor=ACCENT)
        s.configure("TScrollbar",   troughcolor=PANEL, background=BORDER)

    def _center(self):
        self.update_idletasks()
        w, h = 960, 820
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

    def _set_status(self, msg, color=TEXT_DIM):
        self._status_var.set(msg)
        self._status_lbl.config(fg=color)

    # ── Browse ───────────────────────────────────────────────────────────────

    def _browse(self):
        path = filedialog.askdirectory(title="Select folder with 1.jpeg … 16.jpeg")
        if path:
            self._folder = path
            self._folder_lbl.config(text=f"  {path}", fg=TEXT)
            self._set_status(f"Folder: {Path(path).name}", color=TEXT)
            self._reset()

    def _reset(self):
        for cell in self._cells:
            cell.reset()
        self._zoom_canvas.clear()
        self._zoom_canvas.pack_forget()
        self._composite_ph.pack(expand=True)
        self._result = None
        self._save_btn.config(state="disabled")

    # ── Run ──────────────────────────────────────────────────────────────────

    def _run(self):
        if self._running:
            return
        if not self._folder:
            messagebox.showwarning("No folder", "Please select an input folder first.")
            return
        if api is None:
            messagebox.showerror("Import error",
                                 "api_final.py not found.\n"
                                 "Place it in the same directory as this script.")
            return

        rows, cols = self._rows.value, self._cols.value
        self._running = True
        self._reset()
        self._run_btn.config(state="disabled", text=" Running…")
        self._save_btn.config(state="disabled")
        self._rows.set_state("disabled")
        self._cols.set_state("disabled")
        self._progress.start(10)
        self._set_status(f"Processing {self.NUM_IMAGES} images  [{rows}×{cols}]…",
                         color=ACCENT)
        threading.Thread(target=self._worker, args=(self._folder, rows, cols),
                         daemon=True).start()

    def _worker(self, folder, rows, cols):
        try:
            result = api.process_folder(folder_path=folder, rows=rows, cols=cols)
            self.after(0, self._on_done, result)
        except Exception as exc:
            self.after(0, self._on_error, str(exc))

    # ── Done ─────────────────────────────────────────────────────────────────

    def _on_done(self, result: dict):
        self._result = result
        stitched_paths = result.get("stitched_paths", [])

        for i, cell in enumerate(self._cells):
            sp = stitched_paths[i] if i < len(stitched_paths) else None
            if sp and Path(sp).exists():
                try:
                    cell.show(Image.open(sp).convert("RGB"))
                except Exception:
                    cell.error()
            else:
                cell.error()

        # Composite
        final_path    = result.get("final_image_path", "")
        composite_arr = result.get("final_image")
        composite_pil: Optional[Image.Image] = None

        if final_path and Path(final_path).exists():
            try:
                composite_pil = Image.open(final_path).convert("RGB")
            except Exception:
                pass

        if composite_pil is None and composite_arr is not None \
                and hasattr(composite_arr, "size") and composite_arr.size > 1:
            try:
                import cv2
                if composite_arr.ndim == 2:
                    composite_pil = Image.fromarray(composite_arr)
                else:
                    composite_pil = Image.fromarray(
                        cv2.cvtColor(composite_arr, cv2.COLOR_BGR2RGB))
            except Exception:
                pass

        if composite_pil is not None:
            self._composite_ph.pack_forget()
            self._zoom_canvas.pack(fill="both", expand=True)
            self._zoom_canvas.load(composite_pil)
            self._nb.select(1)

        elapsed = result.get("elapsed_sec", 0)
        total   = result.get("success_count", 0)
        out_dir = str(Path(final_path).parent) if final_path else "Output/"
        self._set_status(
            f" Done — {total}/{self.NUM_IMAGES}  |  {elapsed:.1f}s  |  {out_dir}",
            color=SUCCESS)
        self._save_btn.config(state="normal")
        self._finish()

    def _on_error(self, msg):
        messagebox.showerror("Pipeline error", msg)
        self._set_status(f"Error: {msg}", color=ERROR)
        self._finish()

    def _finish(self):
        self._progress.stop()
        self._run_btn.config(state="normal", text="▶  Run Pipeline")
        self._rows.set_state("normal")
        self._cols.set_state("normal")
        self._running = False

    # ── Save ─────────────────────────────────────────────────────────────────

    def _save(self):
        if not self._result:
            return
        final_path = self._result.get("final_image_path", "")
        dest = filedialog.asksaveasfilename(
            title="Save composite as…", initialfile="final_image.jpeg",
            defaultextension=".jpeg",
            filetypes=[("JPEG", "*.jpeg *.jpg"), ("All files", "*.*")],
        )
        if not dest:
            return
        if final_path and Path(final_path).exists():
            import shutil
            shutil.copy2(final_path, dest)
            self._set_status(f"Saved → {dest}", color=SUCCESS)
        else:
            arr = self._result.get("final_image")
            if arr is not None and arr.size > 1:
                import cv2
                cv2.imwrite(dest, arr)
                self._set_status(f"Saved → {dest}", color=SUCCESS)
            else:
                messagebox.showwarning("Nothing to save", "No composite available.")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    app = PipelineGUI()
    app.mainloop()


if __name__ == "__main__":
    main()