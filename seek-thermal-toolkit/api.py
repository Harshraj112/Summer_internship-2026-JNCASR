"""
Seek Thermal - REST API
========================

A FastAPI wrapper around the original Seek Thermal OpenCV sample.
The Tkinter GUI has been removed entirely -- everything it used to do
(Capture, Record/Stop, Rotate CW/CCW, live preview) is now exposed over
HTTP instead of buttons on a window.

Endpoints
---------
GET  /                 : basic health/info
GET  /status           : camera + recording + rotation summary
POST /capture           : grabs the raw per-pixel temperature grid straight
                          from the camera -> saves it to CSV -> THEN builds
                          and saves a PNG image from that same CSV data
                          (never from a screenshot of the live preview).
                          One call = one CSV + one PNG, always in that order.
POST /record/start      : starts a recording session (timestamped folder
                          under recordings/). Repeats the exact same
                          "pixels -> CSV -> image" pair on a timer for as
                          long as recording is on, and additionally stitches
                          a smooth video (video.mp4) built frame-by-frame
                          from the raw thermal data.
POST /record/stop       : stops the current recording session.
GET  /record/status     : recording status (elapsed time, pairs saved...).
POST /rotate/cw         : cycles the orientation correction one step
                          clockwise through the 4 stops (0, 90, 180, 270).
POST /rotate/ccw        : cycles the orientation correction one step
                          counterclockwise.
GET  /rotation          : current rotation index/label.
GET  /preview.jpg        : a single current preview frame as a JPEG.
GET  /stream             : an MJPEG live stream (multipart/x-mixed-replace),
                          viewable directly in a browser <img> tag.

Rotation, like before, applies to the live preview, the saved CSV grids,
the saved images, and the recorded video. Default orientation is 0°
(no rotation).

Dependencies vs. the original Tkinter script:
    pip install fastapi uvicorn opencv-python numpy

(Pillow/Tkinter are no longer needed -- there is no GUI anymore.)
Everything else (the vendor `seekcamera` SDK/bindings) is the same as the
original sample.

Run with:
    python seek_thermal_api.py
or:
    uvicorn seek_thermal_api:app --host 0.0.0.0 --port 8000
"""

import csv
import os
import time
from datetime import datetime
from threading import Lock, Thread
from typing import Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse

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
RECORD_CAPTURE_INTERVAL = 1.0          # how often a CSV+image pair is saved
                                        # while a Record session is running
PREVIEW_FPS_MS = 33                     # ~30 fps background refresh / video fps

CSV_IMAGE_COLORMAP = cv2.COLORMAP_INFERNO

# The 4 orientation states the rotate endpoints cycle through, in order.
ROTATIONS = [
    None,                            # 0 degrees
    cv2.ROTATE_90_CLOCKWISE,         # 90 degrees
    cv2.ROTATE_180,                  # 180 degrees
    cv2.ROTATE_90_COUNTERCLOCKWISE,  # 270 degrees (90 CCW)
]
ROTATION_LABELS = ["0°", "90°", "180°", "270°"]

# FIX 1: Start at 0° (no rotation) instead of the old default of 90° CW.
DEFAULT_ROTATION_INDEX = 0


def apply_rotation(frame, rotation):
    """Rotate a frame/array by the given cv2 rotation constant (or None)."""
    if frame is None or rotation is None:
        return frame
    return cv2.rotate(frame, rotation)


def to_bgr_for_display(frame):
    """Coerce a raw camera color frame into a plain 3-channel BGR image
    suitable for cv2 ops (rotate, resize, imencode, VideoWriter)."""
    if frame is None:
        return None
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    if frame.ndim == 3:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)


def placeholder_frame():
    img = np.full((360, 480, 3), 30, dtype=np.uint8)
    cv2.putText(
        img, "Waiting for camera...", (40, 180),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (130, 130, 130), 2, cv2.LINE_AA,
    )
    return img


# ---------------------------------------------------------------------------
# Camera-side state, shared between the camera's background callback thread
# and the API's background polling thread.
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
    """Top-level capture routine used by both /capture and the Record loop:
        1. Pull the freshest raw temperature grid straight from the camera.
        2. Write that grid out to a CSV file.
        3. Build a brand-new image *from that same CSV data* and save it.

    Returns (csv_path, image_path), either of which may be None on failure.
    """
    pixel_grid = capture_pixels_from_camera(renderer)
    # FIX 2: numpy arrays cannot be used with 'or' because their truth value
    # is ambiguous.  Use an explicit None check instead.
    if pixel_grid is None:
        pixel_grid = renderer.last_thermal_pixels
    capture_time = renderer.last_pixel_capture_time or datetime.now()

    csv_path = save_pixel_csv(pixel_grid, capture_time, directory=csv_dir)
    if csv_path is None:
        return None, None

    img_path = save_image_from_pixels(pixel_grid, capture_time, directory=img_dir)
    return csv_path, img_path


# ---------------------------------------------------------------------------
# Service layer (replaces the old ThermalApp/Tkinter class). Owns the camera
# manager, runs a background polling thread, and exposes plain Python
# methods that the FastAPI routes below call into.
# ---------------------------------------------------------------------------
class ThermalService:
    def __init__(self):
        self.renderer = Renderer()
        self.manager = SeekCameraManager(SeekCameraIOType.USB)
        # Enter the manager context manually (instead of the original's
        # blocking `with`) so it stays alive for the lifetime of the API.
        self.manager.__enter__()
        self.manager.register_event_callback(on_event, self.renderer)

        self._last_pixel_capture = time.time()

        # Recording state.
        self.recording = False
        self.record_writer = None
        self.record_session_dir = None
        self.record_count = 0
        self.record_range = None
        self.record_frame_size = None  # (h, w) the video writer expects
        self.last_record_capture = 0.0
        self.record_start_time = 0.0
        self._record_lock = Lock()

        self.status_message = "Waiting for camera..."

        # Cached latest preview JPEG, refreshed by the background thread,
        # consumed by /preview.jpg and /stream.
        self._preview_lock = Lock()
        self._preview_jpeg: Optional[bytes] = None

        self._running = False
        self._thread: Optional[Thread] = None

    # -- lifecycle -----------------------------------------------------
    def start(self):
        self._running = True
        self._thread = Thread(target=self._background_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        try:
            if self.recording:
                self.stop_recording()
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

    # -- background loop (replaces the old Tkinter _poll) ---------------
    def _background_loop(self):
        while self._running:
            with self.renderer.lock:
                color_frame = self.renderer.color_frame
                thermal_frame = self.renderer.thermal_frame

            if color_frame is not None:
                preview_bgr = apply_rotation(
                    to_bgr_for_display(color_frame.data), self.renderer.rotation
                )
            else:
                preview_bgr = placeholder_frame()

            self._update_preview_cache(preview_bgr)

            now = time.time()

            # Keep a "last known good" pixel grid fresh in the background.
            if thermal_frame is not None and now - self._last_pixel_capture >= PIXEL_CAPTURE_INTERVAL:
                capture_pixels_from_camera(self.renderer)
                self._last_pixel_capture = now

            if self.recording:
                self._record_tick(thermal_frame, now)

            time.sleep(PREVIEW_FPS_MS / 1000.0)

    def _update_preview_cache(self, frame_bgr):
        if frame_bgr is None:
            return
        ok, encoded = cv2.imencode(".jpg", frame_bgr)
        if not ok:
            return
        with self._preview_lock:
            self._preview_jpeg = encoded.tobytes()

    # -- Capture -----------------------------------------------------------
    def capture(self):
        csv_path, img_path = capture_csv_then_image(self.renderer)
        if csv_path is None:
            self.status_message = "No raw pixel data yet -- is the camera connected?"
            return {"success": False, "message": self.status_message}

        self.status_message = "Captured: {} + {}".format(
            os.path.basename(csv_path), os.path.basename(img_path)
        )
        return {
            "success": True,
            "csv_path": csv_path,
            "image_path": img_path,
            "message": self.status_message,
        }

    # -- Record start/stop --------------------------------------------------
    def start_recording(self):
        with self._record_lock:
            if self.recording:
                return {"success": False, "message": "Already recording."}

            # FIX 2 (also here): use explicit None check instead of 'or' on a
            # numpy array, which raises "truth value of array is ambiguous".
            pixel_grid = capture_pixels_from_camera(self.renderer)
            if pixel_grid is None:
                pixel_grid = self.renderer.last_thermal_pixels
            if pixel_grid is None:
                self.status_message = "Can't start recording -- no pixel data from camera yet."
                return {"success": False, "message": self.status_message}

            # Fix the colormap's temperature range for this whole session so
            # the video doesn't flicker as each frame's own min/max wobbles.
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

            self.status_message = "Recording started -> {}".format(self.record_session_dir)
            return {
                "success": True,
                "session_dir": self.record_session_dir,
                "video_path": video_path,
                "message": self.status_message,
            }

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
                # Orientation may have changed mid-recording (rotate
                # endpoints); keep feeding the writer frames of the size it
                # was opened with.
                frame_img = cv2.resize(frame_img, (target_w, target_h))
            self.record_writer.write(frame_img)

        # Periodic CSV+image pair, same pipeline as /capture, saved into
        # this session's own subfolders.
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
        self.status_message = "Recording... {:02d}:{:02d}  ({} CSV+image pairs saved)".format(
            mm, ss, self.record_count
        )

    def stop_recording(self):
        with self._record_lock:
            if not self.recording:
                return {"success": False, "message": "Not currently recording."}

            if self.record_writer is not None:
                self.record_writer.release()
            self.recording = False
            session_dir = self.record_session_dir
            count = self.record_count

            self.status_message = "Recording saved to {} ({} CSV+image pairs, plus video.mp4)".format(
                session_dir, count
            )
            self.record_writer = None

            return {
                "success": True,
                "session_dir": session_dir,
                "pairs_saved": count,
                "message": self.status_message,
            }

    def get_record_status(self):
        elapsed = int(time.time() - self.record_start_time) if self.recording else 0
        mm, ss = divmod(elapsed, 60)
        return {
            "recording": self.recording,
            "session_dir": self.record_session_dir,
            "pairs_saved": self.record_count,
            "elapsed": "{:02d}:{:02d}".format(mm, ss) if self.recording else None,
        }

    # -- Rotate --------------------------------------------------------------
    def rotate_cw(self):
        """Advance one step clockwise (0 -> 90 -> 180 -> 270 -> 0 ...)."""
        self.renderer.rotation_index = (self.renderer.rotation_index + 1) % len(ROTATIONS)
        self.status_message = "Orientation set to {}".format(self.renderer.rotation_label)
        return {"rotation_index": self.renderer.rotation_index, "rotation_label": self.renderer.rotation_label}

    def rotate_ccw(self):
        """Step one back counterclockwise (0 -> 270 -> 180 -> 90 -> 0 ...)."""
        self.renderer.rotation_index = (self.renderer.rotation_index - 1) % len(ROTATIONS)
        self.status_message = "Orientation set to {}".format(self.renderer.rotation_label)
        return {"rotation_index": self.renderer.rotation_index, "rotation_label": self.renderer.rotation_label}

    # -- Preview / status -----------------------------------------------------
    def get_preview_jpeg(self) -> Optional[bytes]:
        with self._preview_lock:
            return self._preview_jpeg

    def mjpeg_generator(self):
        boundary = b"--frame"
        while True:
            jpeg = self.get_preview_jpeg()
            if jpeg is not None:
                yield (
                    boundary + b"\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                )
            time.sleep(PREVIEW_FPS_MS / 1000.0)

    def get_status(self):
        return {
            "camera_connected": self.renderer.camera is not None,
            "rotation_index": self.renderer.rotation_index,
            "rotation_label": self.renderer.rotation_label,
            "recording": self.recording,
            "message": self.status_message,
        }


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Seek Thermal API",
    description="HTTP API for capturing, recording, and rotating a Seek "
                "Thermal camera feed (no GUI).",
)

service: Optional[ThermalService] = None


@app.on_event("startup")
def _startup():
    global service
    service = ThermalService()
    service.start()


@app.on_event("shutdown")
def _shutdown():
    if service is not None:
        service.stop()


@app.get("/")
def root():
    return {"name": "Seek Thermal API", "status": "running"}


@app.get("/status")
def status():
    return service.get_status()


@app.post("/capture")
def capture():
    result = service.capture()
    if not result["success"]:
        raise HTTPException(status_code=409, detail=result["message"])
    return result


@app.post("/record/start")
def record_start():
    result = service.start_recording()
    if not result["success"]:
        raise HTTPException(status_code=409, detail=result["message"])
    return result


@app.post("/record/stop")
def record_stop():
    result = service.stop_recording()
    if not result["success"]:
        raise HTTPException(status_code=409, detail=result["message"])
    return result


@app.get("/record/status")
def record_status():
    return service.get_record_status()


@app.post("/rotate/cw")
def rotate_cw():
    return service.rotate_cw()


@app.post("/rotate/ccw")
def rotate_ccw():
    return service.rotate_ccw()


@app.get("/rotation")
def rotation():
    return {
        "rotation_index": service.renderer.rotation_index,
        "rotation_label": service.renderer.rotation_label,
    }


@app.get("/preview.jpg")
def preview_jpg():
    jpeg_bytes = service.get_preview_jpeg()
    if jpeg_bytes is None:
        raise HTTPException(status_code=503, detail="No frame available yet.")
    return Response(content=jpeg_bytes, media_type="image/jpeg")


@app.get("/stream")
def stream():
    return StreamingResponse(
        service.mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()