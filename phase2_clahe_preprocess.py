import os
from pathlib import Path

import cv2
import numpy as np

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────
SEGMENTED_DIR = r"D:\RL\segmented"
CLAHE_DIR     = r"D:\RL\clahe_preprocessed"

CLASSES = [
    "Stage_0 (Unripe)",
    "Stage_1(Early Ripe)",
    "Stage_2 (Partially Ripe)",
    "Stage_3 (Ripe)",
]

# ─────────────────────────────────────────
# CREATE OUTPUT FOLDERS
# ─────────────────────────────────────────
for cls in CLASSES:
    os.makedirs(os.path.join(CLAHE_DIR, cls), exist_ok=True)

# ─────────────────────────────────────────
# CLAHE SETUP
# ─────────────────────────────────────────
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def apply_clahe(image):
    """
    Apply CLAHE on L channel in LAB color space.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    return result

def extra_preprocess(image):
    """
    Additional light preprocessing after CLAHE:
    - Light Gaussian blur
    - Optional sharpening
    Output remains uint8 BGR, 224x224 (Swin input size).
    """
    # Ensure correct size
    image = cv2.resize(image, (224, 224))

    # Light blur
    blurred = cv2.GaussianBlur(image, (3, 3), 0)

    # Sharpen (unsharp mask style)
    kernel = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ], dtype=np.float32)
    sharpened = cv2.filter2D(blurred, -1, kernel)

    return sharpened

# ─────────────────────────────────────────
# MAIN PROCESS
# ─────────────────────────────────────────
def process_clahe():
    total_processed = 0
    total_skipped   = 0

    print("\n" + "=" * 60)
    print("STEP 2 — CLAHE + Preprocessing on segmented images")
    print("=" * 60)

    for cls in CLASSES:
        src_dir = os.path.join(SEGMENTED_DIR, cls)
        dst_dir = os.path.join(CLAHE_DIR, cls)

        images = [
            f for f in os.listdir(src_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"))
        ]

        print(f"\n{'-'*60}")
        print(f"Class: {cls}  ({len(images)} images)")
        print(f"{'-'*60}")

        for idx, fname in enumerate(images, 1):
            img_path = os.path.join(src_dir, fname)
            img      = cv2.imread(img_path)

            if img is None:
                print(f"  [SKIP - unreadable] {fname}")
                total_skipped += 1
                continue

            # Step 1: CLAHE
            clahe_img = apply_clahe(img)

            # Step 2: extra preprocessing
            final_img = extra_preprocess(clahe_img)

            # Save
            stem    = Path(fname).stem
            outname = f"{stem}_clahe.jpg"
            outpath = os.path.join(dst_dir, outname)
            cv2.imwrite(outpath, final_img)

            total_processed += 1
            print(f"  [{idx}/{len(images)}] ✅ {fname} → {outname}")

    print("\n" + "=" * 60)
    print("✅ Phase 2 (CLAHE + preprocessing) complete.")
    print(f"   Processed : {total_processed} images")
    print(f"   Skipped   : {total_skipped} images")
    print(f"   Saved to  : {CLAHE_DIR}")
    print("=" * 60)


def verify_output():
    print("\n📁 CLAHE-preprocessed dataset structure:")
    print("=" * 60)
    total = 0
    for cls in CLASSES:
        cls_dir = os.path.join(CLAHE_DIR, cls)
        count = len([
            f for f in os.listdir(cls_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        print(f"  📂 {cls}/  →  {count} images")
        total += count
    print("=" * 60)
    print(f"  ✅ Total CLAHE-preprocessed images: {total}")


if __name__ == "__main__":
    process_clahe()
    verify_output()
