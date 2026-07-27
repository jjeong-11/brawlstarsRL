"""Harvests digit templates from recorded footage for digitReader.py.

Runs the (slow, tesseract-based) digit-line reader over video frames.
Whenever tesseract reads a clean >=2-digit number AND the crop segments
into exactly that many character blobs, each blob is a labeled digit
sample. Samples are accumulated per digit and averaged into one template
each, saved to digit_templates.npz.

Usage:
    python3 harvest_digit_templates.py testvideos/test_game1.mp4 [more videos...]
    python3 harvest_digit_templates.py --start 30 --end 60 testvideos/test_game3.mp4

Re-running ADDS to existing samples (stored in digit_samples.npz) before
re-averaging, so harvesting can be done video by video.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pytesseract # type: ignore

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path
from perception.digitReader import segment_digit_blobs, TEMPLATE_W, TEMPLATE_H, TEMPLATE_PATH

# digit sample cache lives next to digitReader.py in perception/
SAMPLES_PATH = Path(__file__).resolve().parent.parent / "perception" / "digit_samples.npz"


def _binarize(crop_bgr):
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 215, 255, cv2.THRESH_BINARY)
    return binary


def _tesseract_read(binary):
    resized = cv2.resize(binary, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
    padded = cv2.copyMakeBorder(resized, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=0)
    text = pytesseract.image_to_string(
        cv2.bitwise_not(padded), config="--psm 7 -c tessedit_char_whitelist=0123456789"
    ).strip()
    return text if text.isdigit() and len(text) >= 2 else None


def harvest_frame(frame, samples):
    """Finds digit-line candidates in a frame and harvests labeled blobs."""
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 90, 255]))
    bridged = cv2.morphologyEx(
        white, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    )
    contours, _ = cv2.findContours(bridged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    added = 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < 18 or not (7 <= h <= 45) or w < h * 1.2:
            continue
        pad_x, pad_y = int(w * 0.15) + 2, int(h * 0.3) + 2
        crop = frame[max(0, y - pad_y):min(height, y + h + pad_y),
                     max(0, x - pad_x):min(width, x + w + pad_x)]
        if crop.size == 0:
            continue

        binary = _binarize(crop)
        text = _tesseract_read(binary)
        if text is None:
            continue

        boxes = segment_digit_blobs(binary)
        if len(boxes) != len(text):
            continue  # segmentation and OCR disagree -> unreliable label

        for ch, (bx, by, bw, bh) in zip(text, boxes):
            blob = binary[by:by + bh, bx:bx + bw]
            blob = cv2.resize(blob, (TEMPLATE_W, TEMPLATE_H), interpolation=cv2.INTER_AREA)
            samples.setdefault(ch, []).append(blob.astype(np.float32))
            added += 1
    return added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--every", type=float, default=2.0)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=1e9)
    args = ap.parse_args()

    # Load previously collected samples so harvesting is incremental.
    samples = {}
    if SAMPLES_PATH.exists():
        prev = np.load(SAMPLES_PATH)
        for key in prev.files:
            digit = key.split("_")[0]
            samples.setdefault(digit, []).append(prev[key])

    total = 0
    for vid in args.videos:
        cap = cv2.VideoCapture(vid)
        sec = args.start
        while sec <= args.end:
            cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
            ok, frame = cap.read()
            if not ok:
                break
            total += harvest_frame(frame, samples)
            sec += args.every
        cap.release()
        print(f"{vid}: running total {total} labeled blobs")

    # Save raw samples (flattened keys digit_i) and averaged templates.
    flat = {}
    for digit, blobs in samples.items():
        for i, blob in enumerate(blobs):
            flat[f"{digit}_{i}"] = blob
    np.savez_compressed(SAMPLES_PATH, **flat)

    templates = {}
    for digit, blobs in samples.items():
        stack = np.stack(blobs)
        templates[digit] = stack.mean(axis=0)
    np.savez_compressed(TEMPLATE_PATH, **templates)

    counts = {d: len(b) for d, b in sorted(samples.items())}
    print(f"samples per digit: {counts}")
    missing = [str(d) for d in range(10) if str(d) not in samples]
    if missing:
        print(f"WARNING: no samples yet for digits: {missing} -- harvest more footage")
    print(f"templates saved to {TEMPLATE_PATH.name}")


if __name__ == "__main__":
    main()
