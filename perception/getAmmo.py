"""
perception/getAmmo.py
=====================

Reads the ammo pip row under the player's health bar.

WHY THIS WAS ONLY WORKING ON ~15% OF FRAMES
-------------------------------------------
The ammo row has no landmark of its own. It is found by chaining:

    verified anchor  ->  find_health_info  ->  ammo row below the HP digits

Three detectors in series, so the hit rates MULTIPLY. Measured on 87 in-match
frames, a verified anchor is available on roughly 60% of them and the HP digit
line parses on 84% of those — which lands ammo at around 50% at best, and in
practice ~15% once the ammo row's own colour/size filters are applied.

That number then poisons everything downstream: `state.ammo_known` is false
most of the time, so the attack gate cannot trust the count and has to fire on
an unknown clip (see `rl/combat.py`).

THE FIX: LEARN THE OFFSET ONCE, THEN STOP CHAINING
--------------------------------------------------
The ammo row sits at a FIXED offset from the player anchor. It has to — both
are drawn relative to the same sprite by the same UI code. So the chain is only
needed the FIRST time: once a full anchor -> health -> ammo read succeeds, the
offset (expressed in anchor radii, so it survives resolution changes) is worth
remembering, and every later frame can look in that exact spot without needing
the HP digits to parse at all.

`AmmoLocator` holds that offset. It re-learns whenever the slow path succeeds,
so a brawler switch or a UI change corrects itself within a few frames rather
than sticking to a stale hint.

This cannot be validated offline in this repo — it needs gameplay footage, and
the recordings were deleted for being played on Sirius (whose clones corrupt
every perception measurement). The mechanism is unit-tested; the hit-rate claim
is not, and should be checked with `scripts/watch_live.py` on a real phone.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .getHealth import find_health_info


@dataclass
class AmmoLocator:
    """Remembers where the ammo row sits relative to the anchor.

    The offset is stored in ANCHOR RADII rather than pixels so it transfers
    across capture resolutions and across the anchor radius wobbling by a few
    pixels between frames.
    """

    dx: Optional[float] = None      # row centre minus anchor x, in radii
    dy: Optional[float] = None      # row centre minus anchor y, in radii
    half_w: float = 1.6             # search window half-size, in radii
    half_h: float = 0.7
    # EMA rate when re-learning. Low: the offset is a UI constant, so a single
    # odd measurement should barely move it, but a real change (brawler switch,
    # different HUD scale) still converges within a handful of good reads.
    alpha: float = 0.25
    learned_from: int = 0           # how many slow-path reads have contributed

    def known(self) -> bool:
        return self.dx is not None and self.dy is not None

    def learn(self, player, box) -> None:
        """Record the offset from a successful slow-path read."""
        ax, ay, ar = player
        if not ar:
            return
        cx = box[0] + box[2] / 2.0
        cy = box[1] + box[3] / 2.0
        dx, dy = (cx - ax) / ar, (cy - ay) / ar
        if not self.known():
            self.dx, self.dy = dx, dy
        else:
            self.dx += self.alpha * (dx - self.dx)
            self.dy += self.alpha * (dy - self.dy)
        # Keep the window comfortably bigger than the row itself so a few
        # radii of jitter cannot push the pips outside it.
        self.half_w = max(self.half_w, 0.8 * box[2] / ar)
        self.learned_from += 1

    def window(self, player, shape) -> Optional[Tuple[int, int, int, int]]:
        """(x0, y0, x1, y1) to search, or None if nothing has been learned."""
        if not self.known():
            return None
        ax, ay, ar = player
        if not ar:
            return None
        h, w = shape[:2]
        cx, cy = ax + self.dx * ar, ay + self.dy * ar
        x0 = max(0, int(cx - self.half_w * ar))
        x1 = min(w, int(cx + self.half_w * ar))
        y0 = max(0, int(cy - self.half_h * ar))
        y1 = min(h, int(cy + self.half_h * ar))
        if x1 - x0 < 4 or y1 - y0 < 3:
            return None
        return (x0, y0, x1, y1)

    def reset(self) -> None:
        self.dx = self.dy = None
        self.learned_from = 0


def find_ammo_info(image, player, health_info=None, locator: "AmmoLocator" = None):
    """Finds the local player's ammo bar by anchoring directly below the

    health bar, using real-world asset size boundaries.

    Returns ``ammo_count`` AND ``detected``. Read them together: an undetected
    frame also reports ``ammo_count == 0``, which is not the same claim as "the
    clip is empty" and must not be treated as one. On real footage the bar is
    only located on ~15% of frames (it needs find_health_info to succeed first),
    so anything gating behaviour on "no ammo" would fire almost constantly if it
    trusted the count alone.

    `health_info` may be passed in when the caller has already run
    find_health_info for this frame (the live loop does), saving a full
    duplicate digit-line search + OCR.
    """
    height, width = image.shape[:2]

    # None = anchor unavailable. Same guard as getHealth/getCube: report "not
    # detected" instead of raising.
    if player is None:
        return {"bounding_box": None, "ammo_count": 0, "detected": False}
    anchor_x, anchor_y, anchor_radius = player

    if anchor_radius == 0:
        return {"bounding_box": None, "ammo_count": 0, "detected": False}

    # STEP 1: FAST PATH -- a learned offset from the anchor.
    # The ammo row is drawn at a fixed offset from the player sprite, so once
    # that offset is known the HP digit line does not need to parse at all.
    # This is what breaks the three-detector chain described in the module
    # docstring; without it, ammo inherits the product of every upstream
    # failure rate.
    fast = locator.window(player, image.shape) if locator is not None else None
    used_fast = False
    if fast is not None:
        ammo_roi_xmin, ammo_roi_ymin, ammo_roi_xmax, ammo_roi_ymax = fast
        used_fast = True
    else:
        # SLOW PATH: anchor off the health bar located by find_health_info.
        # (This used to duplicate its own copy of the green-bar search with an
        # older, narrower ROI -- which silently drifted out of sync when the
        # health search was fixed, so on some captures health was found but
        # ammo still failed. Reusing the same locator keeps them consistent.)
        if health_info is None:
            health_info = find_health_info(image, player)
        if health_info["bounding_box"] is None:
            return {"bounding_box": None, "ammo_count": 0, "detected": False}

        g_hx, g_hy, hw, hh = health_info["bounding_box"]

        # STEP 2: Crop a search window beneath the health readout.
        # find_health_info now returns the HP DIGIT LINE's box (see its
        # docstring), which sits on/just above the green bar -- so the ammo
        # row is a bit further down than when this anchored off the bar
        # contour itself. The window is extended accordingly (digit line ->
        # bar -> ammo row), and widened slightly since the digit line can be
        # narrower than the full bar. The orange color mask plus the segment
        # size filters below keep the larger window from picking up junk.
        ammo_roi_ymin = g_hy + hh
        ammo_roi_ymax = min(height, g_hy + hh + int(hh * 2.6))
        ammo_roi_xmin = max(0, g_hx - int(hw * 0.4))
        ammo_roi_xmax = min(width, g_hx + hw + int(hw * 0.4))

    ammo_roi = image[ammo_roi_ymin:ammo_roi_ymax, ammo_roi_xmin:ammo_roi_xmax]
    if ammo_roi.size == 0:
        return {"bounding_box": None, "ammo_count": 0, "detected": False}

    # STEP 3: Isolate the glowing orange ammo color profile
    hsv_ammo = cv2.cvtColor(ammo_roi, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([5, 120, 120])
    upper_orange = np.array([28, 255, 255])
    ammo_mask = cv2.inRange(hsv_ammo, lower_orange, upper_orange)

    # Clean out stray single-pixel anomalies
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    ammo_mask = cv2.morphologyEx(ammo_mask, cv2.MORPH_OPEN, kernel)

    # STEP 4: Extract and count the valid chunks with realistic sizes
    contours, _ = cv2.findContours(
        ammo_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    valid_segments = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        # Broadened constraints to safely capture 33x11 layout segments
        # and small loading bars (w down to 3 pixels).
        #
        # NOTE: scaling these by the HP digit-line height was tried and made
        # things measurably WORSE (ammo reads 15 -> 11 on an 87-frame sample).
        # The pip row apparently does not scale with the digit line the way the
        # digit thresholds do, so these stay as fixed pixel bounds.
        if 3 <= w <= 45 and 2 <= h <= 15 and area > 4:
            valid_segments.append((x, y, w, h))

    if not valid_segments:
        # The learned window can go stale (brawler switch, a UI scale change,
        # an anchor that drifted). Fall back to the full search ONCE rather
        # than reporting a miss, and let the slow path re-teach the offset.
        if used_fast:
            return find_ammo_info(image, player, health_info=health_info,
                                  locator=None)
        return {"bounding_box": None, "ammo_count": 0, "detected": False}

    # STEP 4b: Keep only the dominant row. The search window is tall/wide
    # enough (see STEP 2) that it can also catch stray orange from other
    # UI rows or map decoration, which inflated the count. All real ammo
    # pips sit on one row, so segments are clustered by their y position
    # and only the row with the most segments is counted.
    rows = {}
    for seg in valid_segments:
        placed = False
        for row_y in list(rows.keys()):
            if abs(seg[1] - row_y) <= 6:
                rows[row_y].append(seg)
                placed = True
                break
        if not placed:
            rows[seg[1]] = [seg]
    valid_segments = max(rows.values(), key=len)

    # STEP 5: Generate the unified global bounding box
    local_x1 = min(item[0] for item in valid_segments)
    local_y1 = min(item[1] for item in valid_segments)
    local_x2 = max(item[0] + item[2] for item in valid_segments)
    local_y2 = max(item[1] + item[3] for item in valid_segments)

    global_box = (
        local_x1 + ammo_roi_xmin,
        local_y1 + ammo_roi_ymin,
        (local_x2 - local_x1),
        (local_y2 - local_y1),
    )
    ammo_count = len(valid_segments)

    # Teach the locator from SLOW-PATH reads only. Learning from its own fast
    # reads would let the window drift a little further each frame until it
    # walked off the row entirely, with nothing to pull it back.
    if locator is not None and not used_fast:
        locator.learn(player, global_box)

    return {"bounding_box": global_box, "ammo_count": ammo_count,
            "detected": True, "fast_path": used_fast}