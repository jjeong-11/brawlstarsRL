"""Classifies whether a crop of the region above a detected health bar

contains real username/text (a genuine enemy or HUD label) versus map
decoration (bushes, crates, coin piles, chest reward tags, etc).

This replaces the hand-tuned `_has_text_above` heuristic in getEnemies.py.
Multiple single-signal heuristics were tried first (white-pixel fraction,
Canny edge density, connected-component "glyph" counting, brightness
fraction) and none generalized across map themes on their own -- usernames
render in different colors per map/brawler, and decorative sprites (crates,
gems) can have comparable edge density or brightness to real text. A small
logistic regression combining several weak signals does much better than
any single hand threshold.

No sklearn is available in this project's environment, so the model is a
plain numpy logistic regression trained via gradient descent -- there's
nothing exotic here, just weights and a sigmoid, saved to a JSON file next
to this module.
"""

import json
from pathlib import Path

import cv2
import numpy as np

WEIGHTS_PATH = Path(__file__).resolve().parent / "username_classifier_weights.json"

FEATURE_NAMES = [
    "white_frac",       # fraction of pixels that are near-white (low sat, high val)
    "bright_frac_200",  # fraction of pixels with V > 200 (any bright color, not just white)
    "bright_frac_150",  # fraction of pixels with V > 150 (softer brightness signal)
    "edge_density",     # fraction of pixels that are Canny edges
    "mean_saturation",  # average saturation (text tends to differ from muted map bg)
    "std_value",        # local brightness variance (texture/contrast)
    "glyph_blob_frac",  # fraction of adaptive-threshold connected components that
                         # are glyph-sized (not too big, not too small)
    "mean_value",        # average brightness
]


def extract_features(image: np.ndarray, bar_box) -> np.ndarray:
    """Extracts the feature vector for the region above a detected bar.

    `bar_box` is the (x, y, w, h) of the health-bar-shaped contour that was
    found; this function looks at the band above it (skipping the zone
    where the bar's own HP digits are known to overflow past its top edge)
    and characterizes whatever's there.
    """
    x, y, w, h = bar_box
    height, width = image.shape[:2]

    roi_xmin = max(0, x - int(w * 0.3))
    roi_xmax = min(width, x + w + int(w * 0.3))
    roi_ymin = max(0, y - int(h * 3.5))
    roi_ymax = max(0, y - int(h * 0.7))

    if roi_ymax <= roi_ymin or roi_xmax <= roi_xmin:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float64)

    roi = image[roi_ymin:roi_ymax, roi_xmin:roi_xmax]
    if roi.size == 0:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float64)

    return _features_from_crop(roi)


def _features_from_crop(roi: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    white_mask = (s_ch < 60) & (v_ch > 190)
    white_frac = float(white_mask.mean())

    bright_frac_200 = float((v_ch > 200).mean())
    bright_frac_150 = float((v_ch > 150).mean())

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(cv2.countNonZero(edges)) / edges.size

    mean_saturation = float(s_ch.mean()) / 255.0
    std_value = float(v_ch.std()) / 255.0
    mean_value = float(v_ch.mean()) / 255.0

    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -5
    )
    n, _, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    rh, rw = gray.shape
    glyph_count = 0
    total_components = max(1, n - 1)
    for i in range(1, n):
        bw, bh, area = stats[i, 2], stats[i, 3], stats[i, 4]
        if area < 8:
            continue
        if bw > rw * 0.4 or bh > rh * 0.7:
            continue
        glyph_count += 1
    glyph_blob_frac = glyph_count / total_components

    return np.array([
        white_frac, bright_frac_200, bright_frac_150, edge_density,
        mean_saturation, std_value, glyph_blob_frac, mean_value,
    ], dtype=np.float64)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def train(X: np.ndarray, y: np.ndarray, epochs=3000, lr=0.5, l2=0.05):
    """Trains a plain logistic regression via batch gradient descent.

    Uses class-balanced sample weights since real training data will
    usually have far more negative (decoration) examples than positive
    (real text) ones.
    """
    n, d = X.shape
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    Xn = (X - mean) / std

    n_pos = max(1, y.sum())
    n_neg = max(1, n - y.sum())
    w_pos = n / (2.0 * n_pos)
    w_neg = n / (2.0 * n_neg)
    sample_w = np.where(y == 1, w_pos, w_neg)

    weights = np.zeros(d, dtype=np.float64)
    bias = 0.0

    for _ in range(epochs):
        z = Xn @ weights + bias
        p = _sigmoid(z)
        error = (p - y) * sample_w
        grad_w = (Xn.T @ error) / n + l2 * weights
        grad_b = error.mean()
        weights -= lr * grad_w
        bias -= lr * grad_b

    return {
        "weights": weights.tolist(),
        "bias": float(bias),
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "feature_names": FEATURE_NAMES,
    }


def save_model(model: dict, path: Path = WEIGHTS_PATH):
    with open(path, "w") as f:
        json.dump(model, f, indent=2)


def _load_model(path: Path = WEIGHTS_PATH):
    with open(path) as f:
        return json.load(f)


_MODEL_CACHE = None


def predict_proba(features: np.ndarray, path: Path = WEIGHTS_PATH) -> float:
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        _MODEL_CACHE = _load_model(path)
    m = _MODEL_CACHE
    mean = np.array(m["feature_mean"])
    std = np.array(m["feature_std"])
    weights = np.array(m["weights"])
    bias = m["bias"]
    fn = (features - mean) / std
    z = fn @ weights + bias
    return float(_sigmoid(z))


def is_likely_username(image: np.ndarray, bar_box, threshold: float = 0.5) -> bool:
    """Drop-in replacement for the old `_has_text_above` heuristic."""
    if not WEIGHTS_PATH.exists():
        # No trained model available -- fail open rather than silently
        # rejecting every candidate (matches the old heuristic's role as a
        # filter, not a hard requirement).
        return True
    features = extract_features(image, bar_box)
    return predict_proba(features) >= threshold
