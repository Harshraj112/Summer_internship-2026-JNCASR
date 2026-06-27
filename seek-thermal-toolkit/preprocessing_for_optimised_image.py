import os
import cv2

# ===========================
# Input / Output Folders
# ===========================
INPUT_FOLDER = "screenshots"
OUTPUT_FOLDER = "Dataset_final"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Supported image formats
EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# ===========================
# Process Every Image
# ===========================
for filename in os.listdir(INPUT_FOLDER):

    if not filename.lower().endswith(EXTENSIONS):
        continue

    image_path = os.path.join(INPUT_FOLDER, filename)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Skipping {filename} (Could not read)")
        continue

    # ==========================================
    # Step 1 : Adaptive Mean Threshold
    # ==========================================
    adaptive_mean = cv2.adaptiveThreshold(
        img,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        11,     # Block Size
        2       # Constant C
    )

    # ==========================================
    # Step 2 : CLAHE Enhancement
    # ==========================================
    clahe = cv2.createCLAHE(
        clipLimit=7.0,
        tileGridSize=(16, 16)
    )

    clahe_image = clahe.apply(img)

    # ==========================================
    # Step 3 : Otsu Binarization
    # (Applied on Adaptive Mean Image)
    # ==========================================
    threshold, otsu = cv2.threshold(
        adaptive_mean,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # ==========================================
    # Save Results
    # ==========================================
    name = os.path.splitext(filename)[0]

    cv2.imwrite(
        os.path.join(OUTPUT_FOLDER, f"CLAHE_{name}.png"),
        clahe_image
    )

    cv2.imwrite(
        os.path.join(OUTPUT_FOLDER, f"Otsu_{name}.png"),
        otsu
    )

    print(f"Processed: {filename} | Otsu Threshold = {threshold:.2f}")

print("\nAll images processed successfully.")