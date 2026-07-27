import cv2
import numpy as np


# --- CALIBRATION CONSTANTS -------------------------------------------------
# MEASURED, not estimated. Derived by colour-clustering five gas screenshots
# spanning four map themes (neon-purple, night-teal, graveyard, Starr Rail),
# with the player both inside and outside the cloud. The gas cluster centre in
# each:
#       H 56 S 100 V 222     H 55 S 105 V 221     H 55 S 104 V 222
#       H 49 S  97 V 211     H 49 S  97 V 211
# i.e. H 49-56, S 97-105, V 211-222 across every theme. Gas is an engine-wide
# overlay drawn on top of the map rather than map art, so it does NOT vary by
# theme -- one band covers all maps, and a map-specific gas profile would be
# pointless.
#
# THE SATURATION CAP IS THE IMPORTANT NUMBER. Green foliage collides with gas
# in hue on several maps (one map's bushes sit at H 58, gas at H 64) and is
# separated only by saturation and value:
#       gas    S  97-105   V 211-222   pale, washed out, translucent
#       bush   S 166-255   V  47-233   deep, solid, opaque
# The previous cap of S<=200 let every one of those bushes through and made the
# detector fire on foliage. S<=155 keeps an ~11 point margin on both sides.
GAS_LOWER = np.array([42, 55, 165])
GAS_UPPER = np.array([66, 155, 255])

# Power cubes are green with a yellow lightning bolt and a yellow glow.
# Gas-colored pixels near that much yellow belong to a cube, not gas.
YELLOW_LOWER = np.array([18, 140, 150])
YELLOW_UPPER = np.array([35, 255, 255])

# Fraction of the window around the player that must be gas-coloured for the
# player to count as standing in gas.
#
# MEASURED on the five calibration screenshots (3 inside the cloud, 2 outside),
# at the default window_scale of 2.5 player radii:
#       inside   0.160  0.257  0.352
#       outside  0.000  0.003
# The two groups are separated by two orders of magnitude, so the exact
# threshold barely matters -- but the old value of 0.18 sat ABOVE the lowest
# genuine inside case (0.160) and missed it. 0.08 sits between the groups with
# a 20x margin over the highest outside reading and a 2x margin under the
# lowest inside one.
IN_GAS_FRACTION = 0.08
# ---------------------------------------------------------------------------


# Precomputed kernels (building a structuring element per call adds up in
# the live loop, where the gas stage runs every tick).
# The cube-glow guard band used to be a 41x41 ELLIPSE dilate, which is not
# separable and measured ~13ms per call at half resolution -- switching to
# the separable RECT kernel of the same size is ~12x faster and the guard
# band's exact corner shape has no effect on the fraction measurements.
_YELLOW_DILATE_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (41, 41))
_OPEN_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def _gas_mask(image_bgr: np.ndarray, hsv: np.ndarray = None) -> np.ndarray:
    """Binary mask of gas-colored pixels, with power-cube areas erased.

    `hsv` may be passed when the caller already has the HSV conversion of
    the same image, to avoid converting twice.
    """
    if hsv is None:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GAS_LOWER, GAS_UPPER)

    yellow = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
    yellow = cv2.dilate(yellow, _YELLOW_DILATE_KERNEL)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(yellow))

    # Drop lone specks so scattered bright-green noise doesn't add up.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _OPEN_KERNEL)
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


def _pool_gas_grid(mask: np.ndarray, rect, grid_size, downscale: float) -> np.ndarray:
    """Pool a gas mask into a (gh, gw) grid of 0..1 coverage over `rect`.

    `rect` is in FULL-resolution frame pixels (the play rect from
    getTerrain.play_rect); `mask` is at `downscale`. The grid is deliberately
    the same shape and alignment as the terrain occupancy grid so the planner
    can add them together cell for cell.
    """
    gw, gh = int(grid_size[0]), int(grid_size[1])
    x, y, w, h = rect
    sh, sw = mask.shape[:2]
    x0 = max(0, min(sw - 1, int(x * downscale)))
    y0 = max(0, min(sh - 1, int(y * downscale)))
    x1 = max(x0 + 1, min(sw, int((x + w) * downscale)))
    y1 = max(y0 + 1, min(sh, int((y + h) * downscale)))
    roi = mask[y0:y1, x0:x1]
    if roi.size == 0:
        return np.zeros((gh, gw), np.float32)
    small = cv2.resize(roi, (gw, gh), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32) / 255.0


def gas_info(image: np.ndarray, player_pos=None, downscale: float = 0.5,
             hsv: np.ndarray = None, grid_rect=None, grid_size=(48, 27)):
    """Full gas picture for the RL state: where the gas is, which way is safe.

    Returns::

        {
          "in_gas":  bool,                      # player standing in the cloud
          "sides":   (left, right, top, bottom),# gas coverage in each half-plane
                                                # around the player (0..1)
          "safe_vector": (dx, dy),              # unit vector pointing AWAY from
                                                # the gas centroid (screen coords,
                                                # +y is DOWN). (0,0) = no gas seen.
          "frac":    float,                     # gas fraction of the whole frame
          "grid":    (gh, gw) float | None,     # per-cell coverage, only when
                                                # `grid_rect` is given
        }

    Pass `grid_rect` (the play rect from getTerrain.play_rect) to also get a
    spatial gas grid aligned to the terrain occupancy grid. This is what lets
    the path planner ROUTE AROUND gas rather than only reacting once `in_gas`
    trips -- which needs the player to be ~18% surrounded, i.e. already well
    inside the cloud and taking damage. The grid is pooled from the mask this
    function already builds, so it costs almost nothing on top.

    `player_pos` is (x, y, r) from getAnchor; frame center is used when None
    (e.g. anchor lost). Runs on a downscaled frame — measured ~0.07 gas frac on
    a real spectate frame vs ~0.002/0.000 on gasless frames, so the numbers are
    meaningful well before the cloud reaches the player.
    """
    height, width = image.shape[:2]
    if hsv is not None:
        # Reuse the caller's full-res HSV: NEAREST downscale so hue values
        # never get interpolated across the red wraparound.
        small_hsv = cv2.resize(hsv, None, fx=downscale, fy=downscale,
                               interpolation=cv2.INTER_NEAREST)
        mask = _gas_mask(None, hsv=small_hsv)
    else:
        small = cv2.resize(image, None, fx=downscale, fy=downscale,
                           interpolation=cv2.INTER_LINEAR)
        mask = _gas_mask(small)
    sh, sw = mask.shape[:2]

    if player_pos is not None and player_pos[2] > 0:
        px, py = int(player_pos[0] * downscale), int(player_pos[1] * downscale)
        # In-gas density check on the SAME downscaled mask (the old code
        # called is_player_in_gas(), which rebuilt a full-res gas mask --
        # including a second big dilate -- just for the player window).
        half = max(1, int(player_pos[2] * 2.5 * downscale))
        wx0, wx1 = max(0, px - half), min(sw, px + half)
        wy0, wy1 = max(0, py - half), min(sh, py + half)
        window = mask[wy0:wy1, wx0:wx1]
        in_gas = (window.size > 0
                  and cv2.countNonZero(window) / window.size >= IN_GAS_FRACTION)
    else:
        px, py = sw // 2, sh // 2
        in_gas = False
    px = min(max(px, 1), sw - 1)
    py = min(max(py, 1), sh - 1)

    def cov(region):
        return float(np.count_nonzero(region)) / max(1, region.size)

    sides = (
        cov(mask[:, :px]),        # left of player
        cov(mask[:, px:]),        # right of player
        cov(mask[:py, :]),        # above player
        cov(mask[py:, :]),        # below player
    )

    total = cv2.countNonZero(mask)
    if total > 0:
        m = cv2.moments(mask, binaryImage=True)
        gx, gy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        dx, dy = px - gx, py - gy        # away from the gas centroid
        norm = (dx * dx + dy * dy) ** 0.5
        safe_vector = (dx / norm, dy / norm) if norm > 1e-6 else (0.0, 0.0)
    else:
        safe_vector = (0.0, 0.0)

    return {
        "in_gas": in_gas,
        "sides": sides,
        "safe_vector": safe_vector,
        "frac": total / float(mask.size),
        "grid": (_pool_gas_grid(mask, grid_rect, grid_size, downscale)
                 if grid_rect is not None else None),
    }


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
