"""Trains the username/text classifier used by getEnemies.py to tell real

enemy usernames apart from map decoration.

Usage:
    python3 train_username_classifier.py

Reads labeled crops from training_data/username_classifier/ (a crop_*.png
per example plus labels.json mapping filename -> 0/1), extracts features
with username_classifier.extract_features's underlying crop-based helper,
trains a logistic regression, runs a quick k-fold sanity check, and writes
username_classifier_weights.json.

To add more training data later: drop new crops into that folder (they can
come from getEnemies.find_enemy_health_bars() candidates on new screenshots
-- see the crop-collection snippet in the module docstring below), add
their labels to labels.json, and re-run this script.
"""

import json
from pathlib import Path

import numpy as np
import cv2

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path
from perception.username_classifier import _features_from_crop, train, save_model, FEATURE_NAMES

DATA_DIR = Path(__file__).resolve().parent.parent / "training_data" / "username_classifier"
LABELS_PATH = DATA_DIR / "labels.json"


def load_dataset():
    """Loads crops from positive/ (label 1) and negative/ (label 0).

    See make_training_data.py for how to generate and sort the crops.
    (The original auto-generated labels.json workflow is retired -- those
    crops were auto-labeled and unreliable.)
    """
    import glob as _glob

    X, y, ids = [], [], []
    for label, sub in ((1, "positive"), (0, "negative")):
        for path in sorted(_glob.glob(str(DATA_DIR / sub / "*.png"))):
            if path.endswith("_ctx.png"):
                continue  # context images are for human eyes, not training
            img = cv2.imread(path)
            if img is None:
                print(f"WARNING: could not load {path}, skipping")
                continue
            X.append(_features_from_crop(img))
            y.append(label)
            ids.append(path)
    if not X:
        raise SystemExit(
            "No training crops found. Generate them with make_training_data.py, "
            "sort into positive/ and negative/, then re-run."
        )
    return np.array(X), np.array(y, dtype=np.float64), ids


def kfold_sanity_check(X, y, k=5):
    n = len(y)
    idx = np.arange(n)
    rng = np.random.default_rng(0)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)

    correct = 0
    total = 0
    tp = fp = fn = tn = 0
    for i in range(k):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        if len(np.unique(y[train_idx])) < 2:
            continue  # can't train on a single class
        model = train(X[train_idx], y[train_idx])
        mean = np.array(model["feature_mean"])
        std = np.array(model["feature_std"])
        w = np.array(model["weights"])
        b = model["bias"]
        z = ((X[test_idx] - mean) / std) @ w + b
        p = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        pred = (p >= 0.5).astype(int)
        actual = y[test_idx].astype(int)
        correct += (pred == actual).sum()
        total += len(actual)
        tp += ((pred == 1) & (actual == 1)).sum()
        fp += ((pred == 1) & (actual == 0)).sum()
        fn += ((pred == 0) & (actual == 1)).sum()
        tn += ((pred == 0) & (actual == 0)).sum()

    print(f"k-fold accuracy: {correct}/{total} = {correct/total:.2%}")
    print(f"  TP={tp} FP={fp} FN={fn} TN={tn}")
    if tp + fn > 0:
        print(f"  recall (catch real text): {tp/(tp+fn):.2%}")
    if tp + fp > 0:
        print(f"  precision (when flagged, was it real): {tp/(tp+fp):.2%}")


def main():
    X, y, ids = load_dataset()
    print(f"Loaded {len(y)} examples ({int(y.sum())} positive, {int(len(y)-y.sum())} negative)")
    print(f"Features: {FEATURE_NAMES}")

    print("\n--- k-fold sanity check (5-fold) ---")
    kfold_sanity_check(X, y, k=5)

    print("\n--- training final model on full dataset ---")
    model = train(X, y)
    save_model(model)
    print(f"Saved weights to username_classifier_weights.json")
    print(f"weights: {dict(zip(FEATURE_NAMES, model['weights']))}")
    print(f"bias: {model['bias']}")


if __name__ == "__main__":
    main()
