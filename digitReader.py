"""Fast digit reading via template matching.

Why this exists: pytesseract shells out to the tesseract binary on every
call (~100-300ms each), and the perception stack makes several calls per
tick -- it was the reason the live loop couldn't get past ~3fps. But the
game renders numbers in ONE fixed font, so a general OCR engine is
overkill: matching each character blob against a small library of digit
templates (harvested from real footage, see harvest_digit_templates.py)
reads the same numbers in ~1ms.

The matcher returns None when it isn't confident (unknown blob shapes,
missing templates), and callers fall back to tesseract for those cases --
so worst case is the old behavior, not a wrong read.
"""

from pathlib import Path

import cv2
import numpy as np

TEMPLATE_PATH = Path(__file__).resolve().parent / "digit_templates.npz"
TEMPLATE_W, TEMPLATE_H = 24, 32
MIN_SCORE = 0.55  # minimum normalized correlation to accept a digit

_TEMPLATES = None  # lazy {digit_str: [float32 exemplar, ...]}


def _load_templates():
    """Loads exemplar templates, grouped by digit.

    Keys in the npz are either plain digits ("5") or exemplar-indexed
    ("5_0", "5_1", ...). Multiple exemplars per digit are kept and matched
    with max-score, because a single averaged template blurred the
    distinguishing strokes of under-sampled digits (a 5/6 confusion showed
    up in validation with mean templates).
    """
    global _TEMPLATES
    if _TEMPLATES is None:
        _TEMPLATES = {}
        if TEMPLATE_PATH.exists():
            data = np.load(TEMPLATE_PATH)
            for key in data.files:
                digit = key.split("_")[0]
                _TEMPLATES.setdefault(digit, []).append(data[key].astype(np.float32))
    return _TEMPLATES


def segment_digit_blobs(binary: np.ndarray):
    """Splits a binarized digit line (white text on black) into per-character

    boxes, left to right. Filters specks and joins nothing -- the game
    font renders digits as separate blobs at every size seen in footage.
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours]
    if not boxes:
        return []
    max_h = max(b[3] for b in boxes)
    # Keep blobs comparable to the tallest one; drops outline specks and
    # the odd sparkle that lands inside the crop.
    boxes = [b for b in boxes if b[3] >= max_h * 0.55 and b[2] >= 2]
    boxes.sort(key=lambda b: b[0])
    return boxes


def has_templates():
    """True once digit templates have been harvested and can be used."""
    return bool(_load_templates())


def read_digits(binary: np.ndarray, min_digits: int = 2):
    """Reads an integer from a binarized digit-line crop.

    Returns int, or None when unsure. `min_digits` guards against reading
    lone noise blobs as numbers (HP callers keep the >=2 rule; the cube
    counter legitimately shows single digits and passes 1).
    """
    templates = _load_templates()
    if not templates:
        return None

    boxes = segment_digit_blobs(binary)
    if len(boxes) < min_digits or len(boxes) > 6:
        return None

    digits = []
    for (x, y, w, h) in boxes:
        blob = binary[y:y + h, x:x + w]
        blob = cv2.resize(blob, (TEMPLATE_W, TEMPLATE_H), interpolation=cv2.INTER_AREA)
        blob_f = blob.astype(np.float32)

        best_digit, best_score = None, -1.0
        for digit, exemplars in templates.items():
            for tmpl in exemplars:
                score = cv2.matchTemplate(blob_f, tmpl, cv2.TM_CCOEFF_NORMED)[0][0]
                if score > best_score:
                    best_score = score
                    best_digit = digit
        if best_digit is None or best_score < MIN_SCORE:
            return None
        digits.append(best_digit)

    return int("".join(digits))
