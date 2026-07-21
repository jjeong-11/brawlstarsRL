import cv2
import numpy as np

from getHealth import _ocr_crop


def find_player_position(image: np.ndarray):
    """Finds the local player's position.

    PRIMARY STRATEGY (digit-first): the one screen element unique to the
    local player is a white, OCR-parseable HP number sitting ON or JUST
    ABOVE a GREEN health bar. Enemies have the same white digits but over
    a RED bar; chests' reward tags are red too; map clutter (bushes,
    grass, leaves, crates) has no readable number at all; and the enemy's
    green cube-badge digits are green, not white. So: find white
    digit-lines anywhere on screen, OCR them, and keep the one with green
    bar pixels directly beneath it. (Earlier strategies -- ring-shape
    matching and green-bar-shape matching -- each got fooled by some map
    arrangement; comments in _find_player_by_ring tell that story.)

    The ring-based approach remains as a fallback for frames where the HP
    readout is obscured or unreadable.

    Returns (x, y, radius) like before; position/radius are derived from
    the located bar (ring center ~1.05 bar-widths below the bar's bottom,
    ring radius ~0.55 bar-widths -- calibrated on captures where both bar
    and ring were measured).
    """
    height, width = image.shape[:2]

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # STEP 1: White digit-line candidates across the whole screen.
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 90, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, bridge)

    contours, _ = cv2.findContours(
        white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    best = None  # (green_frac, digit_box)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Digit-line shaped, and not part of the fixed HUD chrome.
        if w < 18 or not (7 <= h <= 45) or w < h * 1.2:
            continue
        if _in_hud_chrome_zone(x + w / 2, y + h / 2, width, height):
            continue

        # CENTER PRIOR: the camera always keeps the local player near the
        # middle of the screen, so their HP readout can only ever be in
        # the central region. Without this, a white damage popup floating
        # over green bushes at the screen edge (seen in real footage)
        # out-scored the real readout on green support and hijacked the
        # anchor.
        cx_f = (x + w / 2) / width
        cy_f = (y + h / 2) / height
        if not (0.25 <= cx_f <= 0.75 and 0.15 <= cy_f <= 0.80):
            continue

        # STEP 2: Must OCR as a number (>= 2 digits). Letters, icons and
        # decorations fail here.
        pad_x = int(w * 0.15) + 2
        pad_y = int(h * 0.3) + 2
        crop = image[
            max(0, y - pad_y):min(height, y + h + pad_y),
            max(0, x - pad_x):min(width, x + w + pad_x),
        ]
        if _ocr_crop(crop) is None:
            continue

        # STEP 3a: RED VETO. Enemy digits (and chest reward tags) sit ON a
        # red bar, so a tight band right at/under the digit row being
        # red-heavy disqualifies the candidate. The band is deliberately
        # tight: an enemy standing on grass has plenty of green BELOW its
        # red bar, which fooled an earlier wider-band version of this
        # check -- but the rows immediately behind/under its digits are
        # unmistakably red.
        band_xmin = max(0, x - int(w * 0.2))
        band_xmax = min(width, x + w + int(w * 0.2))
        veto_ymin = min(height, y + int(h * 0.2))
        # Deep band on purpose: in the digits-ABOVE-bar layout (seen on
        # real footage), the enemy's red bar sits a full line-height below
        # its digits -- a shallow band probed only the gap between them
        # and let an adjacent enemy's readout hijack the anchor.
        veto_ymax = min(height, y + h + int(h * 2.2))
        veto_hsv = hsv[veto_ymin:veto_ymax, band_xmin:band_xmax]
        if veto_hsv.size == 0:
            continue
        veto_red = cv2.bitwise_or(
            cv2.inRange(veto_hsv, np.array([0, 120, 90]), np.array([10, 255, 255])),
            cv2.inRange(veto_hsv, np.array([170, 120, 90]), np.array([180, 255, 255])),
        )
        if cv2.countNonZero(veto_red) / veto_red.size >= 0.08:
            continue

        # STEP 3b: GREEN SUPPORT. The player's digits sit on/just above
        # their green bar, so the band below the digits normally contains
        # green. BUT: when the player has just been hit, the whole bar
        # flashes orange/white for a few frames and the green measurably
        # drops to zero (seen in real footage) -- so green support makes a
        # candidate PREFERRED (tier 1) rather than required (tier 2).
        band_ymin = min(height, y + int(h * 0.3))
        band_ymax = min(height, y + h + int(h * 2.2))
        band_green = green_mask[band_ymin:band_ymax, band_xmin:band_xmax]
        if band_green.size == 0:
            continue
        green_frac = cv2.countNonZero(band_green) / band_green.size

        # Selection: closest to screen center within each tier. (An
        # earlier version took the highest green fraction instead, but
        # effect noise near the player's green ring glow could out-score
        # the real readout -- while the camera guarantee that the player
        # is the most-centered readable number on screen is much stronger.)
        dist_sq = ((x + w / 2) - width / 2) ** 2 + ((y + h / 2) - height / 2) ** 2
        tier = 1 if green_frac >= 0.10 else 2
        if (best is None or (tier, dist_sq) < (best[0], best[1])):
            best = (tier, dist_sq, (x, y, w, h))

    if best is not None:
        dx, dy, dw, dh = best[2]

        # STEP 4: Derive the anchor purely from the digit line's geometry.
        # (An earlier version tried to refine this by locating the green
        # bar contour below the digits, but adjacent bushes kept getting
        # merged into that contour and dragging the position sideways --
        # the digit line itself is the cleaner reference. The offsets are
        # calibrated from captures where digits, bar, and ring were all
        # measured; the anchor doesn't need to be pixel-perfect, it only
        # centers the search windows of the downstream HUD readers.)
        player_x = dx + dw // 2
        player_y = min(height - 1, int(dy + dh + 1.15 * dw))
        player_radius = max(30, int(0.85 * dw))

        return player_x, player_y, player_radius

    # FALLBACK: no readable HP-over-green anywhere (e.g. the readout is
    # momentarily obscured by an effect).
    return _find_player_by_ring(image, hsv, green_mask)


def _in_hud_chrome_zone(cx, cy, width, height):
    """Fixed HUD chrome zones that can never contain the player's own bar

    or ring: scoreboard (top-left), chat bubble + super-charge counter
    (top-right), movement joystick (bottom-left), attack buttons
    (bottom-right), exit bar (bottom-center). Eyeballed from this
    project's captures at a fixed aspect ratio -- would need re-deriving
    for other capture setups.
    """
    fx, fy = cx / width, cy / height
    if fx < 0.24 and fy < 0.22:
        return True
    if fx > 0.78 and fy < 0.30:
        return True
    if fx < 0.22 and fy > 0.62:
        return True
    if fx > 0.74 and fy > 0.52:
        return True
    if 0.38 < fx < 0.58 and fy > 0.82:
        return True
    return False


def _find_player_by_ring(image: np.ndarray, hsv, mask):
    """Legacy ring-based player finder, now only a fallback.

    Finds the player's green floor ring by shape, then verifies it has
    near-white pixels (username/HP text) floating above it. History: this
    approach went through several rounds of being fooled by green map
    clutter -- bushes pass the color mask, dashed-grass texture is
    bar-shaped, whitish leaf decorations satisfied the white-pixels check
    -- which is why the bar-first strategy above replaced it as primary.
    """
    height, width = image.shape[:2]
    screen_area = height * width

    # Clean up small noise particles
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    ring_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        ring_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)

        # Filter out tiny pixel specs and massive full-sized map zones
        if area < (screen_area * 0.0001) or area > (screen_area * 0.015):
            continue

        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        x, y, w, h = cv2.boundingRect(contour)

        candidates.append({
            "contour": contour, "cx": cx, "cy": cy, "area": area, "w": w, "h": h,
        })

    if not candidates:
        return width // 2, height // 2, 0

    # Ring-shaped: fairly consistent on-screen height, width varies with
    # how the character's legs split it into arc fragments.
    def is_plausible_ring_fragment(c):
        w, h = c["w"], c["h"]
        if not (height * 0.045 <= h <= height * 0.17):
            return False
        if not (width * 0.01 <= w <= width * 0.075):
            return False
        aspect = h / max(1, w)
        return 0.5 <= aspect <= 3.5

    ring_candidates = [
        c for c in candidates
        if is_plausible_ring_fragment(c)
        and not _in_hud_chrome_zone(c["cx"], c["cy"], width, height)
    ]
    search_pool = ring_candidates if ring_candidates else candidates

    # Sort all ring-shaped elements from bottom to top
    search_pool = sorted(search_pool, key=lambda c: c["cy"], reverse=True)

    def has_text_above(cx, cy, ring_w):
        roi_xmin = max(0, cx - int(ring_w * 1.2))
        roi_xmax = min(width, cx + int(ring_w * 1.2))
        roi_ymin = max(0, cy - int(height * 0.22))
        roi_ymax = max(0, cy - int(height * 0.02))
        if roi_ymax <= roi_ymin or roi_xmax <= roi_xmin:
            return False
        roi = hsv[roi_ymin:roi_ymax, roi_xmin:roi_xmax]
        if roi.size == 0:
            return False
        lower_white = np.array([0, 0, 190])
        upper_white = np.array([180, 60, 255])
        white_mask = cv2.inRange(roi, lower_white, upper_white)
        return cv2.countNonZero(white_mask) >= 20

    best_ring_pieces = []
    for i, candidate in enumerate(search_pool):
        cx, cy = candidate["cx"], candidate["cy"]
        if has_text_above(cx, cy, candidate["w"]):
            vertical_tolerance = int(height * 0.03)
            best_ring_pieces = [
                c
                for c in search_pool[i:]
                if abs(c["cy"] - cy) < vertical_tolerance
                and abs(c["cx"] - cx) < (width * 0.06)
            ]
            break

    if not best_ring_pieces:
        best_ring_pieces = [search_pool[0]]

    all_points = np.vstack([p["contour"] for p in best_ring_pieces])
    (circle_x, circle_y), radius = cv2.minEnclosingCircle(all_points)

    return int(circle_x), int(circle_y), int(radius)
