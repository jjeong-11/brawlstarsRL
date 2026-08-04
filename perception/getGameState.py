import cv2
import numpy as np
import pytesseract # type: ignore


# --- CALIBRATION (measured from test_videos/test_game1+2 frames) -----------
# Results screen ("You are #N!"): light-blue bg covers ~60% of the frame;
# nothing else measured above 0.17 (intro splash was the runner-up).
LIGHTBLUE_LOWER = np.array([95, 60, 180])
LIGHTBLUE_UPPER = np.array([115, 200, 255])
RESULTS_LIGHTBLUE_MIN = 0.40

# Loading screen: big gold Showdown logo CENTERED (center-region gold frac
# measured 0.38 on both videos; countdown frames peak at 0.19 because
# their gold lives in the top lineup banner instead).
GOLD_LOWER = np.array([15, 120, 120])
GOLD_UPPER = np.array([35, 255, 255])
LOADING_CENTER_GOLD_MIN = 0.28

# Mid-match death/spectate screen ("Defeated" + Exit button), measured on
# media/fixtures/defeated.png (2424x1080): title-region white frac 0.213 vs 0.030 in-match;
# exit-button-region white frac 0.119 vs 0.000 on every other screen tested.
# Both cues must fire, so a stray white effect can't fake a death.
WHITE_TEXT_LOWER = np.array([0, 0, 190])
WHITE_TEXT_UPPER = np.array([180, 90, 255])
DEFEATED_TITLE_ROI = (0.17, 0.30, 0.38, 0.62)   # (y0, y1, x0, x1) fractions
DEFEATED_TITLE_MIN = 0.10
DEFEATED_EXIT_ROI = (0.88, 0.97, 0.46, 0.60)
DEFEATED_EXIT_MIN = 0.04
# ---------------------------------------------------------------------------


def read_brawlers_left(image: np.ndarray):
    """Reads the "Brawlers left: N" counter shown top-left during Showdown.

    Returns the integer count, or None if the counter isn't visible /
    readable (which itself is a useful signal: no counter usually means
    we're not in an active Showdown match).
    """
    height, width = image.shape[:2]

    # Fixed HUD position, expressed as screen fractions. Generous margins
    # because the exact pixel position shifts slightly with aspect ratio.
    roi = image[
        int(height * 0.01):int(height * 0.09),
        int(width * 0.08):int(width * 0.28),
    ]
    if roi.size == 0:
        return None

    # The text is bright white with a dark purple outline.
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(
        hsv, np.array([0, 0, 190]), np.array([180, 80, 255])
    )

    if cv2.countNonZero(white_mask) < 100:
        return None

    # Upscale + pad, then OCR the whole line and take the trailing number.
    resized = cv2.resize(white_mask, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    padded = cv2.copyMakeBorder(resized, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=0)
    ocr_input = cv2.bitwise_not(padded)

    text = pytesseract.image_to_string(
        ocr_input, config="--psm 7"
    ).strip().lower()

    # Expect something like "brawlers left: 7". Be forgiving about OCR
    # noise in the words; just require a trailing integer.
    digits = ""
    for ch in reversed(text):
        if ch.isdigit():
            digits = ch + digits
        elif digits:
            break
    if not digits:
        return None

    value = int(digits)
    # Showdown counts run 1-10.
    if not (1 <= value <= 10):
        return None
    return value


def read_final_rank(image: np.ndarray):
    """Reads N from the "You are #N!" text on the results screen.

    Returns the placement (1-10) or None. Only meaningful on frames that
    already classified as the results screen.
    """
    height, width = image.shape[:2]
    roi = image[int(height * 0.02):int(height * 0.15),
                int(width * 0.04):int(width * 0.42)]
    if roi.size == 0:
        return None

    # Cream text on blue: low saturation, very bright.
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    text_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([60, 120, 255]))
    if cv2.countNonZero(text_mask) < 200:
        return None

    resized = cv2.resize(text_mask, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    padded = cv2.copyMakeBorder(resized, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=0)
    ocr_input = cv2.bitwise_not(padded)

    text = pytesseract.image_to_string(ocr_input, config="--psm 7").strip()

    # Look for '#N' anywhere in the OCR output ("You are #N!").
    for i, ch in enumerate(text):
        if ch == "#":
            digits = ""
            for c2 in text[i + 1:]:
                if c2.isdigit():
                    digits += c2
                else:
                    break
            if digits and 1 <= int(digits) <= 10:
                return int(digits)

    # Newer end menu says "Rank: N" instead — fall back to the first
    # integer in the line.
    digits = ""
    for ch in text:
        if ch.isdigit():
            digits += ch
        elif digits:
            break
    if digits and 1 <= int(digits) <= 10:
        return int(digits)
    return None


def _white_frac(hsv: np.ndarray, roi_frac) -> float:
    h, w = hsv.shape[:2]
    y0, y1, x0, x1 = roi_frac
    roi = hsv[int(h * y0):int(h * y1), int(w * x0):int(w * x1)]
    if roi.size == 0:
        return 0.0
    mask = cv2.inRange(roi, WHITE_TEXT_LOWER, WHITE_TEXT_UPPER)
    return cv2.countNonZero(mask) / mask.size


def get_game_state(image: np.ndarray, hsv: np.ndarray = None):
    """Coarse game-state classification for episode control.

    States:
      "match_end"  -- results screen; includes "rank" (N from "You are #N!",
                      may be None if the OCR misses) -> terminal reward + reset
      "defeated"   -- mid-match death/spectate screen ("Defeated" + Exit
                      button)                        -> death reward + tap Exit
      "loading"    -- Showdown loading screen (gold logo)      -> wait
      "in_match"   -- Brawlers-left counter readable; includes "brawlers_left"
      "unknown"    -- everything else: brawler intro splash, spawn countdown,
                      "BRAWL!" banner, menus...

    Check order matters: the results screen is checked before the counter
    because leftover HUD elements can linger during transitions, and
    "defeated" is checked before "in_match" because the Brawlers-left counter
    is still visible while spectating.
    """
    height, width = image.shape[:2]
    if hsv is None:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 1) Results screen: dominant light-blue background.
    lightblue = cv2.inRange(hsv, LIGHTBLUE_LOWER, LIGHTBLUE_UPPER)
    if cv2.countNonZero(lightblue) / lightblue.size >= RESULTS_LIGHTBLUE_MIN:
        return {
            "state": "match_end",
            "brawlers_left": None,
            "rank": read_final_rank(image),
        }

    # 2) Death/spectate screen: "Defeated" title + Exit button, both white.
    if (_white_frac(hsv, DEFEATED_TITLE_ROI) >= DEFEATED_TITLE_MIN
            and _white_frac(hsv, DEFEATED_EXIT_ROI) >= DEFEATED_EXIT_MIN):
        return {"state": "defeated", "brawlers_left": None, "rank": None}

    # 3) Loading screen: big gold logo in the center region.
    center = hsv[int(height * 0.25):int(height * 0.75),
                 int(width * 0.35):int(width * 0.65)]
    gold = cv2.inRange(center, GOLD_LOWER, GOLD_UPPER)
    if cv2.countNonZero(gold) / gold.size >= LOADING_CENTER_GOLD_MIN:
        return {"state": "loading", "brawlers_left": None, "rank": None}

    # 4) Active match: the Brawlers-left counter reads.
    brawlers_left = read_brawlers_left(image)
    if brawlers_left is not None:
        return {"state": "in_match", "brawlers_left": brawlers_left, "rank": None}

    return {"state": "unknown", "brawlers_left": None, "rank": None}
