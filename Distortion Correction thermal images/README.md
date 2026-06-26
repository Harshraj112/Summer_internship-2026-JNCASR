# Seek Thermal Capture & Preprocessing Toolkit

Three complementary tools for working with a Seek Thermal camera feed: live
capture/recording, fisheye lens calibration, and turning the captured frames
into a cleaned-up image dataset.

```
GUI.py  ──capture──▶  screenshots/ (+ pixel_captures/, recordings/)
                              │
                              ▼
preprocessing_for_optimised_image.py  ──▶  Dataset_final/

K_and_D_value.ipynb  (standalone) — works out the fisheye K/D
                                     undistortion constants for this camera
```

## File-by-file

| File | What it is |
|------|------------|
| `GUI.py` | A Tkinter "live view" GUI (`Seek Thermal - Live GUI`) wrapped around the vendor Seek Thermal OpenCV sample. Shows the live color preview and gives three actions:<br>• **Capture** — grabs the raw per-pixel temperature grid straight from the camera, writes it to a CSV (`pixel_captures/`), then builds a PNG **from that same CSV data** (`screenshots/`) — never from a screenshot of the live preview.<br>• **Record / Stop** — repeats the same pixel→CSV→image pair on a timer into a timestamped session folder under `recordings/`, and additionally stitches a smooth `video.mp4` built frame-by-frame from the raw thermal data.<br>• **Rotate** — cycles orientation correction through 0°/90°/180°/270°, applied consistently to the live preview, saved CSVs, saved images, and recorded video. |
| `K_and_D_value.ipynb` | A short calibration/test notebook. Defines this camera's fisheye intrinsic matrix `K` and distortion coefficients `D`, computes the undistortion remap with `cv2.fisheye`, and undistorts a sample screenshot to verify the values look right (saves `Undistorted.png` and shows an original-vs-undistorted plot). Use this to (re)derive `K`/`D` if you swap cameras or change resolution — the constants found here are what you'd plug into any downstream pipeline that needs to undistort frames from this camera. |
| `preprocessing_for_optimised_image.py` | Batch-processes every image in `screenshots/` (i.e. the output of `GUI.py`'s Capture button) into a cleaned dataset in `Dataset_final/`: adaptive mean thresholding, CLAHE contrast enhancement, and Otsu binarization (run on the adaptive-threshold output). Saves `CLAHE_<name>.png` and `Otsu_<name>.png` per input image and prints each image's computed Otsu threshold. |

## Requirements

```bash
pip install opencv-python numpy pillow matplotlib
# Vendor SDK for GUI.py (not on PyPI — install per Seek Thermal's SDK docs):
#   seekcamera (Python bindings for the Seek Thermal USB SDK)
```

## Usage

```bash
# 1. Capture frames from the camera
python GUI.py
#    -> Capture button: one CSV + one PNG per click
#    -> Record button:  CSV+PNG pairs on a timer + a video.mp4, in recordings/<timestamp>/

# 2. (optional/one-off) Work out undistortion constants for this camera
#    open K_and_D_value.ipynb, point image_path at a sample screenshot, run it

# 3. Turn captured screenshots into a cleaned dataset
python preprocessing_for_optimised_image.py
```

`preprocessing_for_optimised_image.py` reads from `screenshots/` and writes to
`Dataset_final/` by default — edit `INPUT_FOLDER` / `OUTPUT_FOLDER` at the top
of the script to point at a different capture session (e.g. a `recordings/<timestamp>/screenshots/` folder instead of the top-level one).

## Key config knobs

- `GUI.py`: `PIXEL_CAPTURE_INTERVAL`, `RECORD_CAPTURE_INTERVAL`, `PREVIEW_FPS_MS` — capture/record/preview cadence; `CSV_IMAGE_COLORMAP` — colormap used to render CSV data into PNG/video; `DEFAULT_ROTATION_INDEX` — starting orientation.
- `K_and_D_value.ipynb`: `K`, `D` — camera intrinsics/distortion; `balance` — 0.0 crops more, 1.0 keeps maximum field of view.
- `preprocessing_for_optimised_image.py`: adaptive-threshold block size/constant (`11`, `2`), `CLAHE(clipLimit=7.0, tileGridSize=(16,16))` — tune these if the cleaned dataset looks over/under-enhanced for your captures.