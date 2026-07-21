import cv2
import numpy as np

from getEnemies import find_enemy_health_bars


# --- CALIBRATION CONSTANTS -------------------------------------------------
# Power cubes on the ground are saturated bright-green boxes with a yellow
# lightning bolt and a warm yellow glow. The GREEN band below is tighter /
# more saturated than the pale gas band in getGas.py, and the YELLOW-inside
# requirement is what separates cubes from every other green on the map
# (bushes and gas have no yellow core; coin piles are yellow but not green).
# NOTE: estimated visually from one screenshot with ground cubes -- like
# the gas constants, UNVERIFIED against measured cube pixels. Recalibrate
# once a screenshot with ground cubes is saved into testphotos/.
# (Hue cap is 78, not 80: crate paint-splotch decorations measure H 81-85
# and were sneaking in at 80.)
CUBE_GREEN_LOWER = np.array([40, 130, 110])
CUBE_GREEN_UPPER = np.array([78, 255, 255])

CUBE_YELLOW_LOWER = np.array([18, 130, 150])
CUBE_YELLOW_UPPER = np.array([35, 255, 255])

# Single-cube sprite size bounds in native pixels (loosely scaled to this
# project's ~2244x1000 captures).
CUBE_MIN_DIM = 20
CUBE_MAX_DIM = 150
# Minimum fraction of yellow pixels inside a candidate's (slightly padded)
# bounding box for the lightning bolt to count as present.
CUBE_YELLOW_FRAC = 0.02
# ---------------------------------------------------------------------------


def find_ground_cubes(image: np.ndarray, exclude_positions=None):
    """Finds power cubes lying on the ground.

    `exclude_positions` is an optional list of (x, y, radius) tuples --
    the player anchor and detected enemy/box positions. Brawler sprites
    are themselves green-and-yellow (ring/bar + weapon flashes), which
    fooled an earlier version into reporting the PLAYER as a cube, so any
    candidate overlapping an excluded position is dropped.

    Returns a list of {"center": (x,y), "bounding_box": (x,y,w,h),
    "clustered": bool} -- `clustered` is True when the blob is large
    enough that it's probably several cubes piled together (dropped by a
    defeated brawler); contour merging means a tight pile comes back as
    ONE detection, not one per cube.
    """
    height, width = image.shape[:2]
    exclude_positions = list(exclude_positions or [])

    # Any red HP-bar candidate implies a brawler or box sprite directly
    # below it -- and brawler sprites are green-and-yellow enough to fake
    # a cube even when the enemy VALIDATION failed (e.g. its bar was cut
    # off at the screen edge and didn't OCR). Cubes on the ground never
    # have a red bar floating over them, so the zone under every raw bar
    # candidate is excluded regardless of whether it validated as an
    # enemy/box. (Side effect: a real cube lying directly under an enemy
    # is suppressed -- acceptable, it's about to be picked up anyway.)
    for (bx, by, bw, bh) in find_enemy_health_bars(image):
        exclude_positions.append(
            (bx + bw // 2, by + bh + int(0.65 * bw), max(20, int(0.75 * bw)))
        )
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    green_mask = cv2.inRange(hsv, CUBE_GREEN_LOWER, CUBE_GREEN_UPPER)
    yellow_mask = cv2.inRange(hsv, CUBE_YELLOW_LOWER, CUBE_YELLOW_UPPER)

    # Merge each cube's green faces (the bolt cuts through the middle of
    # the sprite) without gluing distant blobs together.
    green_mask = cv2.morphologyEx(
        green_mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    green_mask = cv2.morphologyEx(
        green_mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )

    contours, _ = cv2.findContours(
        green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    cubes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Size window: big enough for one cube, small enough to exclude
        # bushes/zone tiles. A pile of cubes can exceed the single-cube
        # cap, so the upper bound is generous and flagged as a cluster.
        if w < CUBE_MIN_DIM or h < CUBE_MIN_DIM:
            continue
        if w > CUBE_MAX_DIM * 4 or h > CUBE_MAX_DIM * 4:
            continue

        # HUD exclusion: super-charge button (top-right) is also a green
        # sprite with a bright icon inside, and the bottom-right
        # attack/super/gadget button cluster is yellow-orange rendered
        # over whatever terrain is beneath it (yellow skull + green grass
        # = fake cube). The bottom-right zone reaches to fx 0.68 because
        # the outer attack ring extends well left of the buttons
        # themselves (measured from a real false positive at fx=0.70).
        fx, fy = (x + w / 2) / width, (y + h / 2) / height
        if fx > 0.78 and fy < 0.30:
            continue
        if fx > 0.68 and fy > 0.50:
            continue

        # The lightning bolt: require yellow inside the padded bbox.
        pad = max(4, int(min(w, h) * 0.15))
        y0, y1 = max(0, y - pad), min(height, y + h + pad)
        x0, x1 = max(0, x - pad), min(width, x + w + pad)
        region = yellow_mask[y0:y1, x0:x1]
        yellow_frac = cv2.countNonZero(region) / max(1, region.size)
        if yellow_frac < CUBE_YELLOW_FRAC:
            continue

        # Skip candidates sitting on the player or a detected enemy/box.
        center_x, center_y = x + w // 2, y + h // 2
        on_entity = False
        for (ex, ey, er) in exclude_positions:
            if er <= 0:
                continue
            dist = ((center_x - ex) ** 2 + (center_y - ey) ** 2) ** 0.5
            if dist < er * 1.8:
                on_entity = True
                break
        if on_entity:
            continue

        cubes.append({
            "center": (x + w // 2, y + h // 2),
            "bounding_box": (x, y, w, h),
            "clustered": w > CUBE_MAX_DIM or h > CUBE_MAX_DIM,
        })

    return cubes
