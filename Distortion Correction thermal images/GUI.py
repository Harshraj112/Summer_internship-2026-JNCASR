
"""
Seek Thermal - Live GUI
========================

A Tkinter GUI wrapper around the original Seek Thermal OpenCV sample.

Buttons
-------
- Capture          : grabs the raw per-pixel temperature grid straight from
                      the camera -> saves it to CSV -> THEN builds and saves
                      a PNG image from that same CSV data (never from a
                      screenshot of the live preview). One click = one
                      CSV + one PNG, always in that order.
- Record / Stop     : repeats the exact same "pixels -> CSV -> image" pair
                      on a timer for as long as recording is on (saved into
                      a timestamped session folder under recordings/), and
                      additionally stitches a smooth video (video.mp4) built
                      frame-by-frame from the raw thermal data so you also
                      get a normal video file out of the session.
- Rotate            : cycles the orientation correction through all 4 sides
                      (0, 90, 180, 270 degrees) instead of a fixed constant.
                      Applies to the live preview, the saved CSV grids, the
                      saved images, and the recorded video.

New dependency vs. the original script: Pillow (PIL), used to push OpenCV
frames into a Tkinter label.

    pip install pillow

Everything else (opencv-python, numpy, the vendor `seekcamera` SDK/bindings)
is the same as the original sample.
"""

import csv
import os
import time
from datetime import datetime
from threading import Lock

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

from seekcamera import (
    SeekCameraIOType,
    SeekCameraColorPalette,
    SeekCameraManager,
    SeekCameraManagerEvent,
    SeekCameraFrameFormat,
    SeekCameraTemperatureUnit,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCREENSHOT_DIR = "screenshots"          # manual Capture -> PNGs land here
PIXEL_CSV_DIR = "pixel_captures"        # manual Capture -> CSVs land here
RECORDING_DIR = "recordings"            # Record sessions (video + their own
                                         # CSV/PNG subfolders) land here

PIXEL_CAPTURE_INTERVAL = 1.0            # background "keep a fresh grid" cadence
RECORD_CAPTURE_INTERVAL = 1.0           # how often a CSV+image pair is saved
                                         # while a Record session is running
PREVIEW_FPS_MS = 33                     # ~30 fps GUI refresh / video fps

CSV_IMAGE_COLORMAP = cv2.COLORMAP_INFERNO

# The 4 orientation states the Rotate button cycles through, in order.
ROTATIONS = [
    None,                            # 0 degrees
    cv2.ROTATE_90_CLOCKWISE,         # 90 degrees
    cv2.ROTATE_180,                  # 180 degrees
    cv2.ROTATE_90_COUNTERCLOCKWISE,  # 270 degrees (90 CCW)
]
ROTATION_LABELS = ["0°", "90°", "180°", "270°"]

# Start at 90 deg clockwise to match the original script's default mount fix.
DEFAULT_ROTATION_INDEX = 1


def apply_rotation(frame, rotation):
    """Rotate a frame/array by the given cv2 rotation constant (or None)."""
    if frame is None or rotation is None:
        return frame
    return cv2.rotate(frame, rotation)


# ---------------------------------------------------------------------------
# Camera-side state, shared between the camera's background callback thread
# and the Tkinter GUI thread.
# ---------------------------------------------------------------------------
class Renderer:
    def __init__(self):
        self.busy = False
        self.camera = None
        self.lock = Lock()

        # Latest frames straight from the camera, UN-rotated.
        self.color_frame = None      # camera_frame.color_argb8888
        self.thermal_frame = None    # camera_frame.thermography_float

        self.rotation_index = DEFAULT_ROTATION_INDEX

        # Most recently captured raw pixel grid (numpy array), ALREADY
        # rotated to match the current orientation, plus its timestamp.
        self.last_thermal_pixels = None
        self.last_pixel_capture_time = None

    @property
    def rotation(self):
        return ROTATIONS[self.rotation_index]

    @property
    def rotation_label(self):
        return ROTATION_LABELS[self.rotation_index]


def on_frame(_camera, camera_frame, renderer):
    """Async callback fired whenever a new frame is available."""
    with renderer.lock:
        renderer.color_frame = camera_frame.color_argb8888
        renderer.thermal_frame = camera_frame.thermography_float


def on_event(camera, event_type, event_status, renderer):
    """Async callback fired whenever a camera event occurs."""
    print("{}: {}".format(str(event_type), camera.chipid))

    if event_type == SeekCameraManagerEvent.CONNECT:
        if renderer.busy:
            return
        renderer.busy = True
        renderer.camera = camera
        camera.color_palette = SeekCameraColorPalette.TYRIAN
        camera.tempunit = SeekCameraTemperatureUnit.CELSIUS
        camera.register_frame_available_callback(on_frame, renderer)
        camera.capture_session_start(
            SeekCameraFrameFormat.COLOR_ARGB8888
            | SeekCameraFrameFormat.THERMOGRAPHY_FLOAT
        )

    elif event_type == SeekCameraManagerEvent.DISCONNECT:
        if renderer.camera == camera:
            camera.capture_session_stop()
            renderer.camera = None
            with renderer.lock:
                renderer.color_frame = None
                renderer.thermal_frame = None
            renderer.busy = False

    elif event_type == SeekCameraManagerEvent.ERROR:
        print("{}: {}".format(str(event_status), camera.chipid))

    elif event_type == SeekCameraManagerEvent.READY_TO_PAIR:
        return


# ---------------------------------------------------------------------------
# Pixels -> CSV -> image pipeline (same contract as the original script:
# the saved PNG is always derived from the CSV numbers, never a screenshot
# of the live preview).
# ---------------------------------------------------------------------------
def capture_pixels_from_camera(renderer):
    """Grab the current raw per-pixel temperature grid straight from the
    camera's thermography frame, rotate it to match the live preview, and
    stash it on the renderer. Returns the rotated grid (or None)."""
    with renderer.lock:
        thermal_frame = renderer.thermal_frame
    if thermal_frame is None:
        return None

    pixels = thermal_frame.data.copy()
    rotated = apply_rotation(pixels, renderer.rotation)
    renderer.last_thermal_pixels = rotated
    renderer.last_pixel_capture_time = datetime.now()
    return rotated


def save_pixel_csv(pixel_grid, capture_time, directory=PIXEL_CSV_DIR):
    """Save a 2D grid of raw per-pixel temperatures to a CSV file."""
    if pixel_grid is None:
        print("No raw pixel data captured yet; cannot save CSV.")
        return None

    os.makedirs(directory, exist_ok=True)
    timestamp = capture_time.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = os.path.join(directory, "pixels_{}.csv".format(timestamp))

    with open(filename, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        for row in pixel_grid:
            writer.writerow(row.tolist() if hasattr(row, "tolist") else row)

    print("Pixel data saved to {}".format(filename))
    return filename


def build_image_from_pixel_grid(pixel_grid, fixed_range=None):
    """Build a displayable 8-bit BGR image purely from a 2D grid of raw
    per-pixel temperatures -- the exact same data written out by
    save_pixel_csv.

    fixed_range: optional (min_temp, max_temp) tuple. Leave as None for
    single-shot captures (each image normalized to its own min/max, which
    is what you want for one-off stills). Pass a fixed tuple for a *video*
    sequence so brightness doesn't flicker frame-to-frame as the scene's
    min/max shifts slightly.
    """
    if pixel_grid is None:
        return None

    pixels = np.asarray(pixel_grid, dtype=np.float32)

    if fixed_range is not None:
        min_temp, max_temp = fixed_range
    else:
        min_temp = float(np.min(pixels))
        max_temp = float(np.max(pixels))
    spread = max_temp - min_temp

    if spread < 1e-6:
        normalized = np.zeros(pixels.shape, dtype=np.uint8)
    else:
        clipped = np.clip(pixels, min_temp, max_temp)
        normalized = ((clipped - min_temp) / spread * 255.0).astype(np.uint8)

    return cv2.applyColorMap(normalized, CSV_IMAGE_COLORMAP)


def save_image_from_pixels(pixel_grid, capture_time, directory=SCREENSHOT_DIR,
                            fixed_range=None):
    """Render an image from the raw pixel grid (see build_image_from_pixel_grid)
    and save it to disk."""
    image = build_image_from_pixel_grid(pixel_grid, fixed_range=fixed_range)
    if image is None:
        print("No raw pixel data captured yet; cannot build/save image.")
        return None

    os.makedirs(directory, exist_ok=True)
    timestamp = capture_time.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = os.path.join(directory, "seek_from_csv_{}.png".format(timestamp))

    cv2.imwrite(filename, image)
    print("Image generated from CSV pixel data and saved to {}".format(filename))
    return filename


def capture_csv_then_image(renderer, csv_dir=PIXEL_CSV_DIR, img_dir=SCREENSHOT_DIR):
    """Top-level capture routine used by both the Capture button and the
    Record loop:
        1. Pull the freshest raw temperature grid straight from the camera.
        2. Write that grid out to a CSV file.
        3. Build a brand-new image *from that same CSV data* and save it.

    Returns (csv_path, image_path), either of which may be None on failure.
    """
    pixel_grid = capture_pixels_from_camera(renderer)
    if pixel_grid is None:
        pixel_grid = renderer.last_thermal_pixels  # fall back to last known
    capture_time = renderer.last_pixel_capture_time or datetime.now()

    csv_path = save_pixel_csv(pixel_grid, capture_time, directory=csv_dir)
    if csv_path is None:
        return None, None

    img_path = save_image_from_pixels(pixel_grid, capture_time, directory=img_dir)
    return csv_path, img_path


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
class ThermalApp:
    COLORS = {
        "bg": "#14171c",
        "panel": "#1b2027",
        "fg": "#f5f5f5",
        "muted": "#9aa0a8",
        "accent": "#3b82f6",
        "blue": "#2563eb",
        "red": "#dc2626",
        "purple": "#7c3aed",
        "gray": "#4b5563",
    }
    MAX_DISPLAY_WIDTH = 640
    MAX_DISPLAY_HEIGHT = 480

    def __init__(self, root):
        self.root = root
        self.root.title("Seek Thermal - Live GUI")
        self.root.configure(bg=self.COLORS["bg"])
        self.root.resizable(False, False)

        self.renderer = Renderer()
        self.manager = SeekCameraManager(SeekCameraIOType.USB)
        # Enter the manager context manually (instead of the original's
        # blocking `with`) so it stays alive for the lifetime of the GUI.
        self.manager.__enter__()
        self.manager.register_event_callback(on_event, self.renderer)

        self.last_pixel_capture = time.time()

        # Recording state.
        self.recording = False
        self.record_writer = None
        self.record_session_dir = None
        self.record_count = 0
        self.record_range = None
        self.record_frame_size = None  # (h, w) the video writer expects
        self.last_record_capture = 0.0
        self.record_start_time = 0.0

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll()

    # -- UI construction ---------------------------------------------------
    def _build_ui(self):
        header = tk.Frame(self.root, bg=self.COLORS["bg"])
        header.pack(fill="x", padx=16, pady=(14, 6))

        tk.Label(
            header, text=" Seek Thermal Live",
            font=("Segoe UI", 16, "bold"),
            fg=self.COLORS["fg"], bg=self.COLORS["bg"],
        ).pack(side="left")

        self.rotation_var = tk.StringVar(
            value="Rotation: {}".format(self.renderer.rotation_label)
        )
        tk.Label(
            header, textvariable=self.rotation_var, font=("Segoe UI", 11),
            fg=self.COLORS["muted"], bg=self.COLORS["bg"],
        ).pack(side="right")

        video_wrap = tk.Frame(
            self.root, bg="#000000",
            highlightbackground=self.COLORS["accent"], highlightthickness=2,
        )
        video_wrap.pack(padx=16, pady=8)
        self.video_label = tk.Label(video_wrap, bg="#000000")
        self.video_label.pack()

        controls = tk.Frame(self.root, bg=self.COLORS["bg"])
        controls.pack(pady=(10, 4))

        self.capture_btn = self._make_button(
            controls, " Capture", self.COLORS["blue"], self._on_capture
        )
        self.capture_btn.grid(row=0, column=0, padx=8)

        self.record_btn = self._make_button(
            controls, " Record", self.COLORS["red"], self._on_toggle_record
        )
        self.record_btn.grid(row=0, column=1, padx=8)

        self.rotate_btn = self._make_button(
            controls, " Rotate", self.COLORS["purple"], self._on_rotate
        )
        self.rotate_btn.grid(row=0, column=2, padx=8)

        self.quit_btn = self._make_button(
            controls, " Quit", self.COLORS["gray"], self._on_close
        )
        self.quit_btn.grid(row=0, column=3, padx=8)

        self.status_var = tk.StringVar(value="Waiting for camera...")
        tk.Label(
            self.root, textvariable=self.status_var, font=("Consolas", 10),
            fg=self.COLORS["muted"], bg=self.COLORS["bg"], anchor="w",
        ).pack(fill="x", padx=18, pady=(6, 14))

    def _make_button(self, parent, text, color, command):
        btn = tk.Button(
            parent, text=text, command=command,
            font=("Segoe UI", 11, "bold"),
            bg=color, fg="white",
            activebackground=self._shade(color), activeforeground="white",
            relief="flat", bd=0, padx=18, pady=10, cursor="hand2",
        )
        btn.bind("<Enter>", lambda _e: btn.configure(bg=self._shade(color)))
        btn.bind("<Leave>", lambda _e: btn.configure(bg=color))
        return btn

    @staticmethod
    def _shade(hex_color, factor=0.85):
        hex_color = hex_color.lstrip("#")
        r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
        r, g, b = (max(0, int(c * factor)) for c in (r, g, b))
        return "#{:02x}{:02x}{:02x}".format(r, g, b)

    # -- main GUI loop -------------------------------------------------------
    def _poll(self):
        with self.renderer.lock:
            color_frame = self.renderer.color_frame
            thermal_frame = self.renderer.thermal_frame

        if color_frame is not None:
            preview = apply_rotation(color_frame.data, self.renderer.rotation)
            self._show_frame(preview)
        else:
            self._show_frame(self._placeholder_frame())

        now = time.time()

        # Keep a "last known good" pixel grid fresh in the background,
        # same cadence as the original script.
        if thermal_frame is not None and now - self.last_pixel_capture >= PIXEL_CAPTURE_INTERVAL:
            capture_pixels_from_camera(self.renderer)
            self.last_pixel_capture = now

        if self.recording:
            self._record_tick(thermal_frame, now)

        self.root.after(PREVIEW_FPS_MS, self._poll)

    def _show_frame(self, frame):
        if frame is None:
            return
        if frame.ndim == 3 and frame.shape[2] == 4:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        elif frame.ndim == 3:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        h, w = rgb.shape[:2]
        scale = min(self.MAX_DISPLAY_WIDTH / w, self.MAX_DISPLAY_HEIGHT / h)
        if scale > 0 and abs(scale - 1.0) > 1e-3:
            rgb = cv2.resize(
                rgb, (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_NEAREST,
            )

        photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        self.video_label.configure(image=photo)
        self.video_label.image = photo  # keep a reference, avoid GC

    def _placeholder_frame(self):
        img = np.full((360, 480, 3), 30, dtype=np.uint8)
        cv2.putText(
            img, "Waiting for camera...", (40, 180),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (130, 130, 130), 2, cv2.LINE_AA,
        )
        return img

    # -- Capture button ------------------------------------------------------
    def _on_capture(self):
        csv_path, img_path = capture_csv_then_image(self.renderer)
        if csv_path is None:
            self._set_status("No raw pixel data yet -- is the camera connected?")
            return
        self._set_status(
            "Captured: {} + {}".format(
                os.path.basename(csv_path), os.path.basename(img_path)
            )
        )

    # -- Record button --------------------------------------------------------
    def _on_toggle_record(self):
        if not self.recording:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self):
        pixel_grid = capture_pixels_from_camera(self.renderer) or self.renderer.last_thermal_pixels
        if pixel_grid is None:
            self._set_status("Can't start recording -- no pixel data from camera yet.")
            return

        # Fix the colormap's temperature range for this whole session so the
        # video doesn't flicker as each frame's own min/max wobbles slightly.
        pixels = np.asarray(pixel_grid, dtype=np.float32)
        margin = max(1.0, (float(np.max(pixels)) - float(np.min(pixels))) * 0.1)
        self.record_range = (
            float(np.min(pixels)) - margin,
            float(np.max(pixels)) + margin,
        )

        session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.record_session_dir = os.path.join(RECORDING_DIR, "rec_{}".format(session_ts))
        os.makedirs(self.record_session_dir, exist_ok=True)

        h, w = pixel_grid.shape[:2]
        self.record_frame_size = (h, w)
        video_path = os.path.join(self.record_session_dir, "video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = max(1, int(round(1000.0 / PREVIEW_FPS_MS)))
        self.record_writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

        self.recording = True
        self.record_count = 0
        self.last_record_capture = 0.0
        self.record_start_time = time.time()

        self.record_btn.configure(text="■  Stop Recording")
        self._set_status("Recording started -> {}".format(self.record_session_dir))

    def _record_tick(self, thermal_frame, now):
        if thermal_frame is None or self.record_writer is None:
            return

        # Smooth video: build a fresh frame straight from the raw thermal
        # data every tick (~30 fps), using the fixed range from session start.
        rotated = apply_rotation(thermal_frame.data, self.renderer.rotation)
        frame_img = build_image_from_pixel_grid(rotated, fixed_range=self.record_range)
        if frame_img is not None:
            target_h, target_w = self.record_frame_size
            if frame_img.shape[0] != target_h or frame_img.shape[1] != target_w:
                # Orientation may have changed mid-recording (Rotate button);
                # keep feeding the writer frames of the size it was opened with.
                frame_img = cv2.resize(frame_img, (target_w, target_h))
            self.record_writer.write(frame_img)

        # Periodic CSV+image pair, same pipeline as the Capture button,
        # saved into this session's own subfolders.
        if now - self.last_record_capture >= RECORD_CAPTURE_INTERVAL:
            csv_path, _ = capture_csv_then_image(
                self.renderer,
                csv_dir=os.path.join(self.record_session_dir, "pixel_captures"),
                img_dir=os.path.join(self.record_session_dir, "screenshots"),
            )
            self.last_record_capture = now
            if csv_path:
                self.record_count += 1

        elapsed = int(now - self.record_start_time)
        mm, ss = divmod(elapsed, 60)
        self._set_status(
            "Recording... {:02d}:{:02d}  ({} CSV+image pairs saved)".format(
                mm, ss, self.record_count
            )
        )

    def _stop_recording(self):
        if self.record_writer is not None:
            self.record_writer.release()
        self.recording = False
        self.record_btn.configure(text="⏺  Record")
        self._set_status(
            "Recording saved to {} ({} CSV+image pairs, plus video.mp4)".format(
                self.record_session_dir, self.record_count
            )
        )
        self.record_writer = None

    # -- Rotate button --------------------------------------------------------
    def _on_rotate(self):
        self.renderer.rotation_index = (self.renderer.rotation_index + 1) % len(ROTATIONS)
        self.rotation_var.set("Rotation: {}".format(self.renderer.rotation_label))
        self._set_status("Orientation set to {}".format(self.renderer.rotation_label))

    # -- misc -----------------------------------------------------------------
    def _set_status(self, message):
        self.status_var.set(message)

    def _on_close(self):
        try:
            if self.recording:
                self._stop_recording()
        except Exception:
            pass
        try:
            if self.renderer.camera is not None:
                self.renderer.camera.capture_session_stop()
        except Exception:
            pass
        try:
            self.manager.__exit__(None, None, None)
        except Exception:
            pass
        self.root.destroy()


def main():
    root = tk.Tk()
    ThermalApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()