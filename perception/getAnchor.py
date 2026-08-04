from dataclasses import dataclass

import cv2
import numpy as np

from .getHealth import _ocr_crop

# THE CAMERA PRIOR, in one place, because both detection paths depend on it.
#
# The camera keeps the local player near the middle of the screen. That is the
# whole justification for the digit-first path's gate below -- but until now the
# RING FALLBACK did not honour it and searched the entire frame, so the two
# paths disagreed about where the player could physically be.
#
# Measured on a 45-minute live session (467 traces, 354 anchors): 16.9% of
# accepted anchors sat outside the x-range the digit path considers possible at
# all. Those can only have come from the ungated ring fallback, and they are the
# reason the world map fuses into noise and the map localiser never matches.
#
# The ring sits BELOW the digit line it is paired with, so its vertical window
# is the digit window extended downward; horizontally the two are identical.
CENTRE_X = (0.25, 0.75)
CENTRE_Y_DIGITS = (0.15, 0.80)
CENTRE_Y_RING = (0.15, 0.90)


def _in_centre_prior(cx, cy, width, height, y_range=CENTRE_Y_DIGITS) -> bool:
    """Could the local player's readout/ring be at this point at all?"""
    fx, fy = cx / width, cy / height
    return (CENTRE_X[0] <= fx <= CENTRE_X[1]) and (y_range[0] <= fy <= y_range[1])


def _longest_bar_run(mask, min_coverage: float = 0.55) -> int:
    """Longest run of consecutive rows that are at least `min_coverage` filled.

    Kept as a diagnostic. It measures "is there a solid horizontal bar here",
    which is the right question for telling an HP BAR from a same-coloured
    CHARACTER -- but see the reverted-experiment note in the red veto before
    building a decision on it, because grass answers it too.
    """
    if mask is None or mask.size == 0:
        return 0
    coverage = mask.astype(bool).mean(axis=1)
    best = run = 0
    for c in coverage:
        run = run + 1 if c >= min_coverage else 0
        if run > best:
            best = run
    return best


@dataclass(frozen=True)
class AnchorResult:
    """One anchor detection, with the provenance needed to debug it.

    `source` is the part that was missing. A wrong anchor and a missing anchor
    are completely different bugs -- one poisons every downstream reading, the
    other merely stalls the planner -- and from the logs alone they were
    indistinguishable, because both arrive as a well-formed (x, y, radius).
    """

    x: int
    y: int
    radius: int
    verified: bool
    source: str        # "digits" | "ring" | "ring_unverified" | "none"


def find_player_position(image: np.ndarray, hsv: np.ndarray = None, prior=None):
    """Finds the local player's position. Returns (x, y, radius).

    Thin wrapper over :func:`find_player_position_ex`, kept because callers all
    over the project expect a 3-tuple. Anything positioning a HUD SEARCH WINDOW
    should use the _ex form and check `verified` instead -- see its docstring.
    """
    x, y, r, _ = find_player_position_ex(image, hsv, prior=prior)
    return x, y, r


def find_player_position_ex(image: np.ndarray, hsv: np.ndarray = None, prior=None):
    """Finds the local player's position. Returns (x, y, radius, verified).

    Thin wrapper over :func:`find_player_position_full`; use that one when you
    want to know WHICH strategy produced the anchor.
    """
    r = find_player_position_full(image, hsv, prior=prior)
    return r.x, r.y, r.radius, r.verified


def find_player_position_full(image: np.ndarray, hsv: np.ndarray = None,
                              prior=None) -> AnchorResult:
    """Finds the local player's position. Returns an :class:`AnchorResult`.

    `verified` means the anchor was confirmed by a readable HP number, i.e. it
    came from the digit-first strategy below, or from a ring that had a
    parseable number floating above it.

    THE DISTINCTION IS LOAD-BEARING. Measured over 87 in-match frames:

        verified anchor    -> the HP readout parses on 84% of frames
        unverified anchor  -> 13%

    Unverified anchors are still worth returning: a roughly-right position is
    useful for navigation, and the ring fallback is usually in the right
    neighbourhood even when it has locked onto a crate. But they are NOT good
    enough to position the HP/ammo/cube search windows with, because those are
    all offsets from this point and a wrong origin silently reads the wrong
    part of the screen. `liveLoop` therefore steers with any anchor and only
    runs the HUD readers on verified ones.

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

    Position/radius are derived from
    the located bar (ring center ~1.05 bar-widths below the bar's bottom,
    ring radius ~0.55 bar-widths -- calibrated on captures where both bar
    and ring were measured).

    `hsv` may be passed when the caller already converted this frame
    (LivePerception computes one shared HSV per tick).

    `prior` is the previous frame's anchor. It changes which candidate WINS,
    not which candidates are considered.

    The tie-break used to be "closest to screen centre", justified by the
    camera keeping the player near the middle. That is true but weak: an enemy
    standing between the player and the centre of the screen satisfies it
    better than the player does, and enemy readouts are the exact thing this
    detector keeps getting fooled by. The player's own PREVIOUS position is a
    far tighter prior -- he moves a handful of pixels between frames while
    other brawlers move freely and appear and disappear -- so when it is
    available it replaces the centre prior for selection. The centre prior
    remains as the gate on where a candidate may be at all, and as the
    tie-break on the first frame.
    """
    height, width = image.shape[:2]

    if hsv is None:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # SEARCH REGION: the center prior below rejects everything outside
    # x 0.25-0.75 / y 0.15-0.80 anyway, so the white-mask + morphology +
    # contour scan only needs to cover that central window (~1/3 of the
    # pixels; a small margin keeps boundary-straddling digit lines whole).
    sx0 = max(0, int(width * 0.25) - 40)
    sx1 = min(width, int(width * 0.75) + 40)
    sy0 = max(0, int(height * 0.15) - 40)
    sy1 = min(height, int(height * 0.80) + 40)
    center_hsv = hsv[sy0:sy1, sx0:sx1]

    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(center_hsv, lower_green, upper_green)

    # STEP 1: White digit-line candidates across the central region.
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 90, 255])
    white_mask = cv2.inRange(center_hsv, lower_white, upper_white)
    bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, bridge)

    contours, _ = cv2.findContours(
        white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    best = None  # (green_frac, digit_box)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x, y = x + sx0, y + sy0  # back to full-frame coordinates

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
        if not _in_centre_prior(x + w / 2, y + h / 2, width, height):
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
        # Threshold raised from 0.08 after measuring the funnel on real frames:
        # at 0.08 the veto rejected the LOCAL PLAYER's own readout on roughly
        # two thirds of frames (candidates found on 20/87 frames with the veto
        # vs 53/87 without it), which forced the whole detector down into the
        # ring fallback -- and that fallback yields a usable anchor only 9% of
        # the time. 0.15 recovers ~55% more primary-path frames while still
        # rejecting enemy readouts, whose bars are solidly red rather than
        # incidentally red-tinged.
        #
        # A SHAPE-BASED REPLACEMENT WAS TRIED AND REVERTED. The idea was to veto
        # on a RUN of near-fully-red rows (an HP bar is a solid rectangle; a red
        # brawler's coverage wobbles), with green in the same band as a
        # counter-signal. It fails for a concrete reason: GRASS. An enemy
        # standing on grass supplies a long run of near-fully-GREEN rows, which
        # cancels the veto, and on media/testphotos/brawl-interface5.png and
        # -7.png the detector then locked confidently onto Bot 8 / Bot 1 --
        # source="digits", verified=True, i.e. wrong in the one way downstream
        # cannot detect. Any future attempt needs a green test that grass cannot
        # satisfy; band coverage is not it.
        if cv2.countNonZero(veto_red) / veto_red.size >= 0.15:
            continue

        # STEP 3b: GREEN SUPPORT. The player's digits sit on/just above
        # their green bar, so the band below the digits normally contains
        # green. BUT: when the player has just been hit, the whole bar
        # flashes orange/white for a few frames and the green measurably
        # drops to zero (seen in real footage) -- so green support makes a
        # candidate PREFERRED (tier 1) rather than required (tier 2).
        band_ymin = min(height, y + int(h * 0.3))
        band_ymax = min(height, y + h + int(h * 2.2))
        # green_mask covers only the central search region; index locally.
        band_green = green_mask[
            max(0, band_ymin - sy0):max(0, band_ymax - sy0),
            max(0, band_xmin - sx0):max(0, band_xmax - sx0),
        ]
        if band_green.size == 0:
            continue
        green_frac = cv2.countNonZero(band_green) / band_green.size

        # Selection: closest to screen center within each tier. (An
        # earlier version took the highest green fraction instead, but
        # effect noise near the player's green ring glow could out-score
        # the real readout -- while the camera guarantee that the player
        # is the most-centered readable number on screen is much stronger.)
        ref_x, ref_y = (prior[0], prior[1]) if prior else (width / 2, height / 2)
        dist_sq = ((x + w / 2) - ref_x) ** 2 + ((y + h / 2) - ref_y) ** 2
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

        return AnchorResult(player_x, player_y, player_radius, True, "digits")

    # FALLBACK: no readable HP-over-green anywhere (e.g. the readout is
    # momentarily obscured by an effect). The ring finder searches the whole
    # frame, so build the full-frame green mask lazily here (the hot path
    # above only ever needed the central region).
    green_full = cv2.inRange(hsv, lower_green, upper_green)
    return _find_player_by_ring(image, hsv, green_full)


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
        return AnchorResult(width // 2, height // 2, 0, False, "none")

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

    # THE CENTRE PRIOR APPLIES HERE TOO. This is the fix for the 16.9% of live
    # anchors that landed outside the x-window the digit path treats as
    # physically impossible: this function used to search the whole frame, so
    # any green blob with a number floating above it -- an ENEMY's ring, most
    # often -- could win and be returned VERIFIED. Downstream cannot tell that
    # apart from a real detection, so the world map fused around an enemy and
    # the map localiser was handed noise to match against 71 templates.
    ring_candidates = [
        c for c in candidates
        if is_plausible_ring_fragment(c)
        and not _in_hud_chrome_zone(c["cx"], c["cy"], width, height)
        and _in_centre_prior(c["cx"], c["cy"], width, height, CENTRE_Y_RING)
    ]
    # NOTE: no `else candidates`. That fallback used the raw green-blob list --
    # unfiltered by shape, by HUD chrome or by position -- so on a frame with no
    # plausible ring the bottom-most BUSH became the anchor. Returning nothing
    # is strictly better: liveLoop coasts on the last good anchor, whereas a
    # confident wrong answer propagates into every HUD search window.
    search_pool = ring_candidates

    # Sort all ring-shaped elements from bottom to top
    search_pool = sorted(search_pool, key=lambda c: c["cy"], reverse=True)

    def has_text_above(cx, cy, ring_w):
        """Is there a READABLE HP NUMBER above this ring?

        This used to test for "at least 20 near-white pixels" in a large box,
        which almost anything satisfies -- sparkles, map decoration, a nearby
        brawler's readout. Measured consequence: the fallback produced an
        anchor whose HP could then be read on only 9% of frames, while the
        primary path managed 95%. In practice it was locking onto crates,
        whose dark-green tops pass the ring colour mask.

        Requiring an actual OCR-parseable number is both the correct test (the
        local player is precisely the ring with an HP readout above it) and the
        one that matters downstream, since every HUD reader searches relative
        to this anchor.
        """
        roi_xmin = max(0, cx - int(ring_w * 1.6))
        roi_xmax = min(width, cx + int(ring_w * 1.6))
        roi_ymin = max(0, cy - int(height * 0.22))
        roi_ymax = max(0, cy - int(height * 0.02))
        if roi_ymax <= roi_ymin or roi_xmax <= roi_xmin:
            return False
        roi = image[roi_ymin:roi_ymax, roi_xmin:roi_xmax]
        if roi.size == 0:
            return False
        from .getHealth import find_digit_line
        value, _ = find_digit_line(roi)
        return value is not None

    if not search_pool:
        # Nothing ring-shaped in the place the player can actually be.
        return AnchorResult(width // 2, height // 2, 0, False, "none")

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

    # Whether any ring actually had a readable HP number above it. When none
    # did we still return the best guess -- a rough position is genuinely
    # useful for navigation -- but the caller is told not to trust it enough to
    # position HUD search windows with.
    verified = bool(best_ring_pieces)
    if not best_ring_pieces:
        best_ring_pieces = [search_pool[0]]

    all_points = np.vstack([p["contour"] for p in best_ring_pieces])
    (circle_x, circle_y), radius = cv2.minEnclosingCircle(all_points)

    return AnchorResult(int(circle_x), int(circle_y), int(radius), verified,
                        "ring" if verified else "ring_unverified")
