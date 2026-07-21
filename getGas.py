import cv2
import numpy as np


# --- CALIBRATION CONSTANTS -------------------------------------------------
# The gas clouds are pale, bright, spring-green puffs. These bounds were
# chosen to be disjoint from most other greens measured in this project's
# captures:
#   - map bushes:        H 59-62 but DARK (V 72-116)  -> excluded by V floor
#   - teal bushes:       H ~95                        -> excluded by hue cap
#   - player ring:       V 85-136                     -> excluded by V floor
# KNOWN LIMITATION: bright light-green terrain (zone tiles in one map,
# grass tufts in another) measures H 45-63 / S 93-196 / V 184-233, which
# overlaps this band -- pure color cannot fully separate gas from that
# terrain. NOTE: estimated visually from one gas screenshot + measured
# non-gas samples; UNVERIFIED against real gas pixel values. Recalibrate
# with calibrate_from_sample() once real gas screenshots (ideally one with
# the player standing IN the gas) are in testphotos/.
GAS_LOWER = np.array([40, 70, 170])
GAS_UPPER = np.array([68, 200, 255])

# Power cubes are green with a yellow lightning bolt and a yellow glow.
# Gas-colored pixels near that much yellow belong to a cube, not gas.
YELLOW_LOWER = np.array([18, 140, 150])
YELLOW_UPPER = np.array([35, 255, 255])

# Fraction of the window around the player that must be gas-colored for
# the player to count as standing in gas. When actually inside the cloud,
# puffs surround the character densely, so this can be fairly high --
# which also guards against a stray light-green tuft near the player.
IN_GAS_FRACTION = 0.18
# ---------------------------------------------------------------------------


def _gas_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Binary mask of gas-colored pixels, with power-cube areas erased."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GAS_LOWER, GAS_UPPER)

    yellow = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
    yellow = cv2.dilate(
        yellow, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    )
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(yellow))

    # Drop lone specks so scattered bright-green noise doesn't add up.
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    return mask


def is_player_in_gas(image: np.ndarray, player_pos, window_scale: float = 2.5) -> bool:
    """True if the player appears to be standing inside the poison gas.

    Checks the density of gas-colored pixels in a window centered on the
    player: inside the cloud, puffs fill a large share of the player's
    surroundings; outside it (even right at the edge), they don't.

    `player_pos` is (x, y, radius) from getAnchor.find_player_position.
    `window_scale` sets the window's half-size in player radii.
    """
    height, width = image.shape[:2]
    px, py, pr = int(player_pos[0]), int(player_pos[1]), int(player_pos[2])
    if pr <= 0:
        return False

    half = int(pr * window_scale)
    x0, x1 = max(0, px - half), min(width, px + half)
    y0, y1 = max(0, py - half), min(height, py + half)
    window = image[y0:y1, x0:x1]
    if window.size == 0:
        return False

    mask = _gas_mask(window)
    frac = cv2.countNonZero(mask) / mask.size
    return frac >= IN_GAS_FRACTION


def calibrate_from_sample(image: np.ndarray, gas_points):
    """Helper for recalibrating GAS_LOWER/GAS_UPPER from a real screenshot.

    Pass a list of (x, y) pixel coordinates that are inside gas clouds;
    prints the HSV percentile spread so the constants above can be updated
    with measured values instead of visual estimates.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    samples = []
    for (x, y) in gas_points:
        patch = hsv[max(0, y - 8):y + 8, max(0, x - 8):x + 8].reshape(-1, 3)
        samples.append(patch)
    all_px = np.vstack(samples)
    for i, ch in enumerate("HSV"):
        p = np.percentile(all_px[:, i], [2, 50, 98]).astype(int)
        print(f"{ch}: p2={p[0]} median={p[1]} p98={p[2]}")
