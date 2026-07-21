"""Generates unlabeled training crops for the username classifier.

WORKFLOW (what you do):
  1. Drop in-match screenshots into training_data/screenshots/
     (or point --video at a recorded match to auto-extract frames).
     Best sources: frames with visible ENEMIES and visible BOXES/CHESTS,
     across as many different maps and brawlers as possible.
  2. Run:  python3 make_training_data.py
  3. Look at each image in training_data/username_classifier/unsorted/.
     Every crop shows the region ABOVE one detected red health bar, and
     has a companion *_ctx.png showing where on the screenshot it came
     from (yellow box = the bar, magenta box = the crop region).
  4. MOVE each crop (just the crop, not the _ctx) into:
        training_data/username_classifier/positive/
            -> a floating USERNAME is visible in the crop
               (i.e. the bar belongs to an enemy brawler)
        training_data/username_classifier/negative/
            -> NO username: box/chest bars, terrain, effects, damage
               numbers, anything else
  5. Re-train:  python3 train_username_classifier.py

Aim for at least ~50 crops per class; more maps = better generalization.
"""

from pathlib import Path
import argparse
import glob

import cv2

from getEnemies import find_enemy_health_bars, _in_scoreboard_zone

BASE = Path(__file__).resolve().parent / "training_data"
SCREENSHOT_DIR = BASE / "screenshots"
UNSORTED_DIR = BASE / "username_classifier" / "unsorted"


def crop_region(image, bar_box):
    """Same geometry the classifier scores at inference time

    (username_classifier.extract_features) -- training crops must match.
    """
    x, y, w, h = bar_box
    height, width = image.shape[:2]
    x0 = max(0, x - int(w * 0.3))
    x1 = min(width, x + w + int(w * 0.3))
    y0 = max(0, y - int(h * 3.5))
    y1 = max(0, y - int(h * 0.7))
    if y1 <= y0 or x1 <= x0:
        return None, None
    return image[y0:y1, x0:x1], (x0, y0, x1, y1)


def process_image(image, tag, out_dir):
    count = 0
    height, width = image.shape[:2]
    for i, bar in enumerate(find_enemy_health_bars(image)):
        if _in_scoreboard_zone(bar, width, height):
            continue
        crop, region = crop_region(image, bar)
        if crop is None or crop.size == 0:
            continue
        name = f"{tag}_b{i}_{bar[0]}_{bar[1]}"
        cv2.imwrite(str(out_dir / f"{name}.png"), crop)

        ctx = image.copy()
        bx, by, bw, bh = bar
        cv2.rectangle(ctx, (bx, by), (bx + bw, by + bh), (0, 255, 255), 3)
        cv2.rectangle(ctx, (region[0], region[1]), (region[2], region[3]), (255, 0, 255), 3)
        scale = 900 / width
        ctx = cv2.resize(ctx, None, fx=scale, fy=scale)
        cv2.imwrite(str(out_dir / f"{name}_ctx.png"), ctx)
        count += 1
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", help="also extract frames from this video")
    ap.add_argument("--every", type=float, default=3.0,
                    help="seconds between extracted video frames (default 3)")
    args = ap.parse_args()

    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    UNSORTED_DIR.mkdir(parents=True, exist_ok=True)
    (BASE / "username_classifier" / "positive").mkdir(parents=True, exist_ok=True)
    (BASE / "username_classifier" / "negative").mkdir(parents=True, exist_ok=True)

    total = 0
    shots = sorted(glob.glob(str(SCREENSHOT_DIR / "*.png"))) + \
            sorted(glob.glob(str(SCREENSHOT_DIR / "*.jpg")))
    for path in shots:
        img = cv2.imread(path)
        if img is None:
            continue
        total += process_image(img, Path(path).stem, UNSORTED_DIR)

    if args.video:
        cap = cv2.VideoCapture(args.video)
        sec = 0.0
        while True:
            cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
            ok, frame = cap.read()
            if not ok:
                break
            total += process_image(frame, f"{Path(args.video).stem}_{int(sec)}s", UNSORTED_DIR)
            sec += args.every
        cap.release()

    print(f"Wrote {total} crops (plus context images) to {UNSORTED_DIR}")
    print("Now sort each crop into positive/ (username visible) or negative/ (no username),")
    print("then run: python3 train_username_classifier.py")


if __name__ == "__main__":
    main()
