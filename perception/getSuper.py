"""getSuper.py -- reads the player's SUPER charge from the HUD button.

The super button (bottom-right control cluster, the skull button) shows charge as
a gold/yellow ring that fills as you deal damage, then becomes a solid glowing
gold disc when it's ready to fire. We measure the gold-area fraction inside a
fixed button ROI and map it to a 0..1 charge level.

Why this matters for RL: the super only charges by dealing damage, so a *rise* in
this value is the cheapest reliable "the agent dealt damage" signal -- it drives
the charge-super reward in rl/rewards.py.

CALIBRATION (measured on media/testvideos, 2424x1080 letterboxed landscape):
  * Uncharged button  -> ~0% gold in the ROI.
  * Charging arc       -> a few % up to ~18%.
  * Full / ready       -> solid gold disc, ~18-25%.
The ROI and thresholds are frame-fraction based, like getGameState.py. On a
different device/aspect ratio, re-check with:  python -m perception.getSuper <img>
which saves an overlay of the ROI so you can nudge SUPER_ROI.
"""

import cv2
import numpy as np

# Super-button ROI as (x0, x1, y0, y1) fractions of the frame.
SUPER_ROI = (0.685, 0.775, 0.665, 0.825)

# Gold/yellow ring+disc color band (OpenCV HSV: H 0-180).
_GOLD_LO = np.array([18, 80, 120])
_GOLD_HI = np.array([40, 255, 255])

# Gold-area fraction of the ROI at full charge (calibrated). charge saturates at 1.
FULL_GOLD_FRAC = 0.18
# charge >= this reads as "ready to fire".
READY_CHARGE = 0.9


def find_super_info(image, roi_fracs=SUPER_ROI):
    """Return the super-charge reading for one frame.

    {
      "charge":       float 0..1  (fraction charged; 1.0 == ready),
      "ready":        bool        (>= READY_CHARGE),
      "gold_frac":    float       (raw gold-area fraction, for debugging),
      "bounding_box": (x, y, w, h) or None,
    }

    Assumes the HUD is on screen (call it only while in a match). On menu /
    loading frames the gold logo would read as "charged"; liveLoop gates this
    stage on the in_match state for exactly that reason.
    """
    h, w = image.shape[:2]
    x0, x1 = int(w * roi_fracs[0]), int(w * roi_fracs[1])
    y0, y1 = int(h * roi_fracs[2]), int(h * roi_fracs[3])
    roi = image[y0:y1, x0:x1]
    if roi.size == 0:
        return {"charge": 0.0, "ready": False, "gold_frac": 0.0, "bounding_box": None}

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, _GOLD_LO, _GOLD_HI)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    gold_frac = cv2.countNonZero(mask) / mask.size
    charge = float(min(1.0, gold_frac / FULL_GOLD_FRAC))
    return {
        "charge": charge,
        "ready": charge >= READY_CHARGE,
        "gold_frac": gold_frac,
        "bounding_box": (x0, y0, x1 - x0, y1 - y0),
    }


if __name__ == "__main__":  # quick calibration helper
    import sys
    if len(sys.argv) < 2:
        print("usage: python -m perception.getSuper <image_path>")
        raise SystemExit(1)
    img = cv2.imread(sys.argv[1])
    info = find_super_info(img)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in info.items()})
    x, y, w, h = info["bounding_box"]
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 3)
    cv2.putText(img, f"super {info['charge']*100:.0f}%", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    out = "super_roi_overlay.png"
    cv2.imwrite(out, img)
    print("overlay saved to", out)
