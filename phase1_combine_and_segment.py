import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ─────────────────────────────────────────
# PATHS (CHANGE ONLY IF YOUR STRUCTURE CHANGES)
# ─────────────────────────────────────────
ROOT_DIR = r"D:\RL\Ripening Detection of Mango"

# Raw combined images (train+test for each class)
COMBINED_DIR  = r"D:\RL\raw_combined"

# Segmented, cropped mango images (output of YOLO detect+segment)
SEGMENTED_DIR = r"D:\RL\segmented"

# YOLO models
DETECT_MODEL_PATH  = r"D:\RL\detect.pt"
SEGMENT_MODEL_PATH = r"D:\RL\segment.pt"

# Class folder names (exact as they appear in ROOT_DIR)
CLASSES = [
    "Stage_0 (Unripe)",
    "Stage_1(Early Ripe)",
    "Stage_2 (Partially Ripe)",
    "Stage_3 (Ripe)",
]

# ─────────────────────────────────────────
# STEP 1 — COMBINE TRAINING + TEST PER CLASS
# ─────────────────────────────────────────
def combine_train_test():
    print("\n" + "=" * 60)
    print("STEP 1 — Combining Training and Test into raw_combined/")
    print("=" * 60)

    # Create output structure
    for cls in CLASSES:
        dst_dir = os.path.join(COMBINED_DIR, cls)
        os.makedirs(dst_dir, exist_ok=True)

        # Source dirs for this class
        class_root = os.path.join(ROOT_DIR, cls)
        train_dir  = os.path.join(class_root, "Training")
        test_dir   = os.path.join(class_root, "Test")

        count = 0

        # Copy training images
        if os.path.isdir(train_dir):
            for fname in os.listdir(train_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")):
                    src = os.path.join(train_dir, fname)
                    dst = os.path.join(dst_dir, f"train_{fname}")
                    shutil.copy2(src, dst)
                    count += 1

        # Copy test images
        if os.path.isdir(test_dir):
            for fname in os.listdir(test_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")):
                    src = os.path.join(test_dir, fname)
                    dst = os.path.join(dst_dir, f"test_{fname}")
                    shutil.copy2(src, dst)
                    count += 1

        print(f"  📂 {cls}  →  {count} images combined into {dst_dir}")

    print("\n✅ Combining train+test complete.\n")


# ─────────────────────────────────────────
# STEP 2 — YOLOv8 DETECT + SEGMENT
# ─────────────────────────────────────────

def load_models():
    print("=" * 60)
    print("STEP 2 — Loading YOLOv8 detect and segment models")
    print("=" * 60)
    detect_model  = YOLO(DETECT_MODEL_PATH)
    segment_model = YOLO(SEGMENT_MODEL_PATH)
    print("✅ Models loaded.\n")
    return detect_model, segment_model


def basic_preprocess(image):
    """Resize to 640x640 and light blur for YOLO inference."""
    image = cv2.resize(image, (640, 640))
    image = cv2.GaussianBlur(image, (3, 3), 0)
    return image


def detect_mango(detect_model, image, img_name):
    """
    Run YOLO detect model on full image.
    Returns best bbox (x1, y1, x2, y2) or None if no detection.
    """
    results = detect_model(image, conf=0.25, verbose=False)
    boxes   = results[0].boxes

    if boxes is None or len(boxes) == 0:
        print(f"  [SKIP - no detection] {img_name}")
        return None

    confidences = boxes.conf.cpu().numpy()
    best_idx    = int(np.argmax(confidences))
    x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)

    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    return (x1, y1, x2, y2)


def segment_mango(segment_model, image, bbox, img_name):
    """
    Run YOLO segment model on cropped bbox.
    Returns masked mango crop or None.
    """
    x1, y1, x2, y2 = bbox
    cropped = image[y1:y2, x1:x2]

    if cropped.size == 0:
        print(f"  [SKIP - empty crop] {img_name}")
        return None

    results = segment_model(cropped, conf=0.25, verbose=False)
    masks   = results[0].masks

    if masks is None or len(masks) == 0:
        # Fallback: just use bbox crop
        print(f"  [WARN] No mask, using bbox crop for {img_name}")
        return cropped

    mask = masks.data[0].cpu().numpy()
    mask = cv2.resize(mask, (cropped.shape[1], cropped.shape[0]))
    mask = (mask > 0.5).astype(np.uint8) * 255

    masked = cv2.bitwise_and(cropped, cropped, mask=mask)
    return masked


def process_segmentation(detect_model, segment_model):
    print("=" * 60)
    print("STEP 2 — Running YOLO detect + segment on combined images")
    print("=" * 60)

    # Create segmented output folders
    for cls in CLASSES:
        os.makedirs(os.path.join(SEGMENTED_DIR, cls), exist_ok=True)

    total_processed = 0
    total_skipped   = 0

    for cls in CLASSES:
        src_dir = os.path.join(COMBINED_DIR, cls)
        dst_dir = os.path.join(SEGMENTED_DIR, cls)

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

            # Preprocess
            img_pre = basic_preprocess(img)

            # Detect
            bbox = detect_mango(detect_model, img_pre, fname)
            if bbox is None:
                total_skipped += 1
                continue

            # Segment
            segmented = segment_mango(segment_model, img_pre, bbox, fname)
            if segmented is None:
                total_skipped += 1
                continue

            # Resize to 224x224 for Swin input
            segmented = cv2.resize(segmented, (224, 224))

            # Save
            stem    = Path(fname).stem
            outname = f"{stem}_seg.jpg"
            outpath = os.path.join(dst_dir, outname)
            cv2.imwrite(outpath, segmented)

            total_processed += 1
            print(f"  [{idx}/{len(images)}] ✅ {fname} → {outname}")

    print("\n" + "=" * 60)
    print("✅ YOLO detect + segment complete.")
    print(f"   Processed : {total_processed} images")
    print(f"   Skipped   : {total_skipped} images")
    print(f"   Saved to  : {SEGMENTED_DIR}")
    print("=" * 60)


def verify_segmented():
    print("\n📁 Segmented dataset structure:")
    print("=" * 60)
    total = 0
    for cls in CLASSES:
        cls_dir = os.path.join(SEGMENTED_DIR, cls)
        if not os.path.isdir(cls_dir):
            print(f"  📂 {cls}/  →  0 images (folder missing)")
            continue
        count = len([
            f for f in os.listdir(cls_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        print(f"  📂 {cls}/  →  {count} images")
        total += count
    print("=" * 60)
    print(f"  ✅ Total segmented images: {total}")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    # STEP 1: combine original Training + Test into raw_combined/
    combine_train_test()

    # STEP 2: load YOLO models and run detection + segmentation
    detect_model, segment_model = load_models()
    process_segmentation(detect_model, segment_model)

    # Summary
    verify_segmented()
