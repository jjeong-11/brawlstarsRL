import cv2
import numpy as np
import pytesseract # type: ignore

import username_classifier
from getHealth import find_digit_line


def _read_bar_hp(image: np.ndarray, bar_box):
    """Reads the HP number belonging to a red bar candidate.

    The digits render in two different layouts depending on the frame:
    overlaid ON the bar (most of the still captures) or floating fully
    ABOVE it (seen throughout one recorded match -- where the old
    tight-crop OCR chopped every digit off and silently reported None for
    every real enemy on screen). A digit-line search over a window
    spanning both layouts handles either. Returns int or None.
    """
    x, y, w, h = bar_box
    height, width = image.shape[:2]

    # Wide horizontal margins on purpose: a DEPLETED bar's red segment is
    # narrower than the full bar, but the digits stay centered on the full
    # bar -- a tight window cropped "4368" down to "43" on real footage.
    x0 = max(0, x - int(w * 1.2))
    x1 = min(width, x + w + int(w * 1.2))
    y0 = max(0, y - int(h * 2.8))
    y1 = min(height, y + h + int(h * 0.7))
    roi = image[y0:y1, x0:x1]
    if roi.size == 0:
        return None

    value, _box = find_digit_line(roi)
    return value


def find_enemy_positions(image: np.ndarray, scale_factor: float = 1.0):
    """Finds enemy floor rings (red circles) anywhere on the screen, using the
    same strategy as find_player_position() in getAnchor.py: a ring is only
    trusted once it's confirmed to have a companion UI element (username /
    health bar) floating directly above it. Unlike the player, there can be
    several enemies on screen at once, so every ring that passes the check is
    returned instead of just the single best match.

    NOTE: this is no longer the primary detector used by
    find_enemies_with_stats(). The ring's squashed-ellipse shape turned out
    to false-positive on other red/orange ground clutter -- coin piles,
    gems, gadgets -- since a lot of map decoration is roughly oval and
    reddish. It's kept here for reference / possible future use as a
    secondary confirmation signal. See find_enemy_health_bars() below.
    """
    work_image = image
    if scale_factor != 1.0:
        h, w = image.shape[:2]
        work_image = cv2.resize(
            image, (int(w * scale_factor), int(h * scale_factor)),
            interpolation=cv2.INTER_LINEAR,
        )

    height, width = work_image.shape[:2]
    screen_area = height * width

    # Step 1: Scan the entire screen for enemy-red assets.
    # Red wraps around the HSV hue wheel, so two ranges are combined.
    # Saturation is kept high (>=135) on purpose: the muted maroon tree
    # stumps scattered around the map sit at S~110-131, while the enemy
    # ring/username/reticle red is always a vivid S~143-220. That gap is
    # what keeps stumps out of the mask without needing shape heuristics.
    hsv = cv2.cvtColor(work_image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 135, 90])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 135, 90])
    upper_red2 = np.array([180, 255, 255])
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, lower_red1, upper_red1),
        cv2.inRange(hsv, lower_red2, upper_red2),
    )

    # Clean up small noise particles
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
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

        candidates.append({"contour": contour, "cx": cx, "cy": cy, "area": area})

    if not candidates:
        return []

    # Step 2: Sort all red elements from bottom to top (highest Y to lowest Y)
    candidates.sort(key=lambda c: c["cy"], reverse=True)

    used_ids = set()
    enemies = []

    # Step 3: Find every red object that has a "companion" UI element above it.
    # Each one that qualifies marks the base of a distinct enemy's floor ring.
    for candidate in candidates:
        if id(candidate) in used_ids:
            continue

        cx, cy = candidate["cx"], candidate["cy"]

        has_ui_above = False
        for other in candidates:
            if other is candidate:
                continue
            # Check if 'other' is horizontally aligned and vertically higher up
            x_aligned = abs(cx - other["cx"]) < (width * 0.05)
            y_above = 0 < (cy - other["cy"]) < (height * 0.15)
            if x_aligned and y_above:
                has_ui_above = True
                break

        if not has_ui_above:
            continue

        # Gather any neighboring fragments (ring split by legs/scenery) at this
        # same height level so the full ring merges into one detection.
        vertical_tolerance = int(height * 0.03)
        ring_pieces = [
            c
            for c in candidates
            if id(c) not in used_ids
            and abs(c["cy"] - cy) < vertical_tolerance
            and abs(c["cx"] - cx) < (width * 0.06)
        ]

        for piece in ring_pieces:
            used_ids.add(id(piece))

        # Step 4: Merge the matching pieces to find the true absolute center
        all_points = np.vstack([p["contour"] for p in ring_pieces])
        (circle_x, circle_y), radius = cv2.minEnclosingCircle(all_points)

        # Scale back up to native image coordinates if we downscaled earlier
        inv_scale = 1.0 / scale_factor
        native_cx = int(circle_x * inv_scale)
        native_cy = int(circle_y * inv_scale)
        native_radius = int(radius * inv_scale)

        enemies.append({
            "center": (native_cx, native_cy),
            "radius": native_radius,
            "bounding_box": (
                native_cx - native_radius,
                native_cy - native_radius,
                native_radius * 2,
                native_radius * 2,
            ),
        })

    return _merge_nearby_enemies(enemies, width * inv_scale)


def _merge_nearby_enemies(enemies, native_width):
    """Collapses duplicate detections of the same enemy.

    A single ring can still get split into more than one group above (e.g.
    the username satisfies "has_ui_above" for several disconnected ring
    fragments independently, especially when an attack effect breaks the
    ring into pieces). Any two detections whose centers are close relative
    to their own size are almost certainly the same enemy, so they're
    merged into one, keeping the larger radius and the union bounding box.
    """
    n = len(enemies)
    if n <= 1:
        return enemies

    # Union-find so that A-B-C chains of close detections all end up in one
    # cluster, even if A and C aren't directly within merge range of each
    # other (a simple one-pass "compare everything to the first" approach
    # misses these transitive chains).
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            a, b = enemies[i], enemies[j]
            ax, ay = a["center"]
            bx, by = b["center"]
            dist = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
            merge_radius = max(a["radius"], b["radius"], native_width * 0.03) * 1.5
            if dist < merge_radius:
                union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(enemies[i])

    merged = []
    for group in groups.values():
        if len(group) == 1:
            merged.append(group[0])
            continue

        x0 = min(e["bounding_box"][0] for e in group)
        y0 = min(e["bounding_box"][1] for e in group)
        x1 = max(e["bounding_box"][0] + e["bounding_box"][2] for e in group)
        y1 = max(e["bounding_box"][1] + e["bounding_box"][3] for e in group)

        best = max(group, key=lambda e: e["radius"])
        merged.append({
            "center": best["center"],
            "radius": max(e["radius"] for e in group),
            "bounding_box": (x0, y0, x1 - x0, y1 - y0),
        })

    return merged


# Same red range find_enemy_positions() uses to isolate the ring/username/reticle
# red from the muted maroon tree-stump red. Reused here so the HUD readers below
# key off the same calibrated color as the detector that finds the enemy in the
# first place, instead of guessing a fresh range.
_ENEMY_RED_LOWER1 = np.array([0, 135, 90])
_ENEMY_RED_UPPER1 = np.array([10, 255, 255])
_ENEMY_RED_LOWER2 = np.array([170, 135, 90])
_ENEMY_RED_UPPER2 = np.array([180, 255, 255])


def _enemy_red_mask(hsv_roi):
    return cv2.bitwise_or(
        cv2.inRange(hsv_roi, _ENEMY_RED_LOWER1, _ENEMY_RED_UPPER1),
        cv2.inRange(hsv_roi, _ENEMY_RED_LOWER2, _ENEMY_RED_UPPER2),
    )


def find_enemy_health_bars(image: np.ndarray):
    """Scans the ENTIRE screen directly for enemy health-bar shaped contours.

    This is the primary enemy detector: instead of first guessing a floor
    ring and hoping a health bar sits above it, it looks for the bar itself.
    A health bar is a thin, wide capsule -- a much rarer shape than the
    ring's squashed ellipse (1.2-2.2), which kept matching round/oval
    clutter like coin piles.

    The white HP-number text overlaid on the bar often cuts it into two
    separate left/right contours (digit strokes break the red mask). A
    horizontal-only closing kernel bridges those gaps back together before
    contour extraction; it's horizontal-only (not square) specifically so
    it can't also bridge the vertical gap up to the username above.

    Returns a list of (x, y, w, h) bounding boxes in native image coords.
    """
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = _enemy_red_mask(hsv)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    bars = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        aspect_ratio = w / max(1, h)

        # NOTE: upper aspect bound is generous (real captures showed bars
        # up to ~10:1 once digit-gap bridging merges them into one piece)
        # -- may need further tuning once tested against more footage.
        if area < 100 or not (2.2 <= aspect_ratio <= 12.0):
            continue

        # Absolute pixel-size sanity bounds (loosely scaled to a 1080p-ish
        # capture) so we don't pick up huge banners or tiny noise that just
        # happens to fall inside the aspect-ratio window.
        # NOTE: unverified against a large sample of real footage yet --
        # may need tightening/loosening once tested against more screenshots.
        if not (18 <= w <= 260 and 4 <= h <= 60):
            continue

        # NOTE: an earlier version of this excluded the whole bottom-right
        # quadrant to dodge the attack/aim control cluster, but that zone
        # was large enough to also exclude a real enemy (kpic) standing in
        # that part of the map. The round attack button doesn't pass the
        # aspect-ratio/size checks above anyway (confirmed against real
        # captures), so the extra exclusion wasn't earning its keep and has
        # been dropped. If UI-button false positives show up in practice,
        # prefer a small fixed-pixel corner box over a fractional quadrant.

        bars.append((x, y, w, h))

    return bars


def _ocr_health_digits(image: np.ndarray, bar_box):
    """Given an already-located health-bar bounding box, crops and OCRs the

    digits inside it. This is the shared crop/threshold/OCR pipeline pulled
    out of find_enemy_health_info() so find_enemies_with_stats() can reuse
    it directly on a bar it already found, instead of re-running the same
    contour search a second time.
    """
    x, y, w, h = bar_box
    height, width = image.shape[:2]

    pad_x = int(w * 0.15)
    # The HP digits are taller than the bar itself and overflow above its
    # top edge (confirmed by inspecting real captures) -- a small 10% pad
    # was chopping the tops off every digit, which silently killed OCR.
    pad_y = int(h * 0.6)

    start_x = max(0, x - pad_x)
    start_y = max(0, y - pad_y)
    end_x = min(width, x + w + pad_x)
    end_y = min(height, y + h + pad_y)

    crop = image[start_y:end_y, start_x:end_x]
    if crop.size == 0:
        return None

    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized_gray = cv2.resize(
        gray_crop, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC
    )
    _, text_mask = cv2.threshold(resized_gray, 215, 255, cv2.THRESH_BINARY)

    final_pad = 20
    padded_mask = cv2.copyMakeBorder(
        text_mask,
        final_pad,
        final_pad,
        final_pad,
        final_pad,
        cv2.BORDER_CONSTANT,
        value=0,
    )
    final_ocr_input = cv2.bitwise_not(padded_mask)

    config = "--psm 7 -c tessedit_char_whitelist=0123456789"
    text = pytesseract.image_to_string(final_ocr_input, config=config).strip()
    # Require at least 2 digits: real HP values are never single-digit in
    # practice, but stray sparks/rim highlights inside red-bar-shaped
    # clutter occasionally OCR as a lone digit and produced phantom
    # "HP=4" boxes/enemies. (Same rule as the player-side OCR in getHealth.)
    if not text.isdigit() or len(text) < 2:
        return None
    return int(text)


def find_enemy_health_info(image, enemy):
    """Finds an enemy's health bar and extracts the current health value via OCR.

    Mirrors find_health_info() from getHealth.py, but enemy HP bars render in
    red rather than the player's green, so the color mask uses the same red
    range find_enemy_positions() already calibrated for the ring/username.
    """
    height, width = image.shape[:2]
    anchor_x, anchor_y, anchor_radius = enemy

    if anchor_radius == 0:
        return {"bounding_box": None, "current_health": None}

    # STEP 1: Crop the vertical window above the enemy's feet (same geometry as the player)
    roi_xmin = max(0, int(anchor_x - 1.2 * anchor_radius))
    roi_xmax = min(width, int(anchor_x + 1.2 * anchor_radius))
    roi_ymin = max(0, int(anchor_y - 2.4 * anchor_radius))
    roi_ymax = max(0, int(anchor_y - 1.3 * anchor_radius))

    roi = image[roi_ymin:roi_ymax, roi_xmin:roi_xmax]
    if roi.size == 0:
        return {"bounding_box": None, "current_health": None}

    # STEP 2: Find the red health bar contour
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    red_mask = _enemy_red_mask(hsv_roi)

    contours, _ = cv2.findContours(
        red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    health_box = None
    best_aspect_ratio = 0

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        aspect_ratio = w / max(1, h)

        if 2.2 <= aspect_ratio <= 6.0 and area > 100:
            if aspect_ratio > best_aspect_ratio:
                best_aspect_ratio = aspect_ratio
                health_box = (x, y, w, h)

    if health_box is None:
        return {"bounding_box": None, "current_health": None}

    hx, hy, hw, hh = health_box
    global_box = (hx + roi_xmin, hy + roi_ymin, hw, hh)

    # STEP 3: Crop with an expanded horizontal safety margin
    pad_x = int(hw * 0.15)
    pad_y = int(hh * 0.10)

    start_x = max(0, hx - pad_x)
    start_y = max(0, hy - pad_y)
    end_x = min(roi.shape[1], hx + hw + pad_x)
    end_y = min(roi.shape[0], hy + hh + pad_y)

    health_bar_crop = roi[start_y:end_y, start_x:end_x]

    # STEP 4: Enhanced Text Preprocessing (Grayscale -> Resize -> Threshold)
    gray_crop = cv2.cvtColor(health_bar_crop, cv2.COLOR_BGR2GRAY)
    resized_gray = cv2.resize(
        gray_crop, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC
    )
    _, text_mask = cv2.threshold(resized_gray, 215, 255, cv2.THRESH_BINARY)

    final_pad = 20
    padded_mask = cv2.copyMakeBorder(
        text_mask,
        final_pad,
        final_pad,
        final_pad,
        final_pad,
        cv2.BORDER_CONSTANT,
        value=0,
    )
    final_ocr_input = cv2.bitwise_not(padded_mask)

    # STEP 5: Run OCR using PSM 7 (Treat image as a single text line)
    config = "--psm 7 -c tessedit_char_whitelist=0123456789"
    text = pytesseract.image_to_string(final_ocr_input, config=config).strip()
    current_health = int(text) if text.isdigit() else None

    return {"bounding_box": global_box, "current_health": current_health}


# NOTE: this used to be a hand-tuned "is there white text above the bar"
# heuristic (_has_text_above). It worked on the map it was tuned against,
# but broke as soon as a second map used a differently-colored username
# (purple instead of white) -- and every other single-signal heuristic
# tried in its place (Canny edge density, connected-component glyph
# counting, brightness fraction) turned out to be foolable by some map
# decoration too (crate wood-grain texture, chest reward tags, gem
# sprites). None of them cleanly separated real usernames from clutter on
# their own. `username_classifier.is_likely_username()` replaces it with a
# small trained classifier that combines several of those signals -- see
# username_classifier.py and train_username_classifier.py. It's trained on
# a small, manually-labeled set of crops (training_data/username_classifier/)
# and should be retrained as more labeled examples are collected from new
# maps/captures; see that module's docstring for how.


def find_enemy_cube_info(image, enemy):
    """Finds an enemy's power-cube badge above their head and extracts the count via OCR.

    Mirrors find_cube_info() from getCube.py, but uses the calibrated enemy
    red range instead of the player's green.
    """
    height, width = image.shape[:2]
    anchor_x, anchor_y, anchor_radius = enemy

    if anchor_radius == 0:
        return {"bounding_box": None, "cube_count": None}

    # STEP 1: Define the vertical column ROI above the enemy
    roi_xmin = max(0, int(anchor_x - 1.3 * anchor_radius))
    roi_xmax = min(width, int(anchor_x + 1.3 * anchor_radius))
    roi_ymin = max(0, int(anchor_y - 3.8 * anchor_radius))
    roi_ymax = max(0, int(anchor_y - 1.1 * anchor_radius))

    roi = image[roi_ymin:roi_ymax, roi_xmin:roi_xmax]
    if roi.size == 0:
        return {"bounding_box": None, "cube_count": None}

    # STEP 2: Generate the red mask
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = _enemy_red_mask(hsv_roi)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    valid_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 4 or h < 4 or cv2.contourArea(contour) < 12:
            continue
        valid_contours.append((x, y, w, h, contour))

    if not valid_contours:
        return {"bounding_box": None, "cube_count": None}

    # STEP 3: Find the highest row (the cube asset row)
    min_local_y = min(item[1] for item in valid_contours)

    absolute_top_y = min_local_y + roi_ymin
    if (anchor_y - absolute_top_y) < (anchor_radius * 2.5):
        return {"bounding_box": None, "cube_count": None}

    row_threshold = int(anchor_radius * 0.4)
    row_contours = [
        item for item in valid_contours if abs(item[1] - min_local_y) < row_threshold
    ]

    local_x1 = min(item[0] for item in row_contours)
    local_y1 = min(item[1] for item in row_contours)
    local_x2 = max(item[0] + item[2] for item in row_contours)
    local_y2 = max(item[1] + item[3] for item in row_contours)

    row_w = local_x2 - local_x1
    row_h = local_y2 - local_y1

    global_x = local_x1 + roi_xmin
    global_y = local_y1 + roi_ymin
    best_box = (global_x, global_y, row_w, row_h)

    # STEP 4: Isolate the digits directly from the binary mask
    row_mask = mask[local_y1:local_y2, local_x1:local_x2]

    num_w = int(row_w * 0.65)
    num_x = row_w - num_w
    number_mask = row_mask[:, num_x:]

    if number_mask.size == 0:
        return {"bounding_box": best_box, "cube_count": None}

    # STEP 5: Read the digits -- template fast path (min_digits=1, badge
    # counts are often single digits), tesseract only if no templates.
    try:
        import digitReader
        if digitReader.has_templates():
            return {
                "bounding_box": best_box,
                "cube_count": digitReader.read_digits(number_mask, min_digits=1),
            }
    except ImportError:
        pass

    pad = 15
    padded = cv2.copyMakeBorder(
        number_mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0
    )
    resized = cv2.resize(padded, None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST)
    thresh = cv2.bitwise_not(resized)

    config = "--psm 7 -c tessedit_char_whitelist=0123456789"
    text = pytesseract.image_to_string(thresh, config=config).strip()
    cube_count = int(text) if text.isdigit() else None

    return {"bounding_box": best_box, "cube_count": cube_count}


def _in_scoreboard_zone(bar_box, width, height):
    """The top-left elimination scoreboard ("Bot 1", "Bot 2", ...) contains

    real text and can have red/orange-ish elements near it, but it's a
    fixed HUD panel, not an in-game character -- a bar-shaped candidate
    found there should never be reported as an enemy. Bounds are eyeballed
    from this project's captures at a fixed aspect ratio; would need
    re-deriving for a different capture setup.
    """
    x, y, w, h = bar_box
    fx, fy = x / width, y / height
    return fx < 0.30 and fy < 0.25


def _overlaps_player(bar_box, player_pos, margin=1.5):
    """Some brawlers render their own HP bar as a segmented red/yellow/green

    gradient rather than solid green -- the red segment can pass the enemy
    red mask and get picked up as a candidate "enemy" bar sitting right on
    top of the local player. If the player's own anchor is known, drop any
    candidate whose bar sits within a margin of it.
    """
    if player_pos is None:
        return False
    px, py, pr = player_pos
    if pr <= 0:
        return False
    x, y, w, h = bar_box
    bar_cx, bar_cy = x + w / 2, y + h / 2
    dist = ((bar_cx - px) ** 2 + (bar_cy - py) ** 2) ** 0.5
    return dist < (pr * margin)


def find_entities(image, player_pos=None):
    """Single-pass combined detector: scans the red bar candidates ONCE,

    OCRs each ONCE, classifies each ONCE, and splits the results into
    enemies (username above) and boxes (no username). Use this instead of
    calling find_enemies_with_stats() and find_boxes() back-to-back --
    they do the same full-screen scan and per-candidate OCR twice, which
    matters a lot in the live loop (entity OCR is the most expensive
    stage in the whole pipeline).

    Returns {"enemies": [...], "boxes": [...]} with the same per-item
    shapes as the individual functions.
    """
    height, width = image.shape[:2]
    bars = find_enemy_health_bars(image)

    accepted = []  # (entity, is_enemy)
    for bar_box in bars:
        if _in_scoreboard_zone(bar_box, width, height):
            continue
        if _overlaps_player(bar_box, player_pos):
            continue

        bx, by, bw, bh = bar_box

        # De-dup: a single depleted bar splits into several red fragments
        # (remaining-HP segment + rim pieces), each of which passes the
        # shape filter and would produce a duplicate entity. Any candidate
        # whose bar sits within one bar-width of an already-accepted one
        # is the same entity.
        cx = bx + bw // 2
        duplicate = False
        for prev, _ in accepted:
            pbx, pby, pbw, pbh = prev["health"]["bounding_box"]
            if abs(cx - (pbx + pbw // 2)) < max(bw, pbw) and abs(by - pby) < max(bh, pbh) * 2.5:
                duplicate = True
                break
        if duplicate:
            continue

        current_health = _read_bar_hp(image, bar_box)
        if current_health is None:
            continue

        pseudo_radius = max(1, int(bw / 2.0))
        anchor_x = bx + bw // 2
        anchor_y = by + bh + int(1.3 * pseudo_radius)
        entity = {
            "center": (anchor_x, anchor_y),
            "radius": pseudo_radius,
            "bounding_box": (
                anchor_x - pseudo_radius,
                anchor_y - pseudo_radius,
                pseudo_radius * 2,
                pseudo_radius * 2,
            ),
            "health": {"bounding_box": bar_box, "current_health": current_health},
        }

        is_enemy = username_classifier.is_likely_username(image, bar_box)
        if is_enemy:
            entity["cubes"] = find_enemy_cube_info(
                image, (anchor_x, anchor_y, pseudo_radius)
            )
        accepted.append((entity, is_enemy))

    return {
        "enemies": [e for e, is_en in accepted if is_en],
        "boxes": [e for e, is_en in accepted if not is_en],
    }


def find_boxes(image, player_pos=None):
    """Finds destructible boxes/chests: red HP bars with a readable health

    number but NO username above them. This is exactly the signature that
    find_enemies_with_stats() rejects -- an enemy always has a floating
    username over its bar, a box never does -- so the two detectors split
    the same bar candidates between them.

    Returns a list of {"center", "radius", "bounding_box", "health"} in the
    same shape as enemy detections (center/radius approximate the box
    sprite below the bar).
    """
    height, width = image.shape[:2]
    bars = find_enemy_health_bars(image)

    boxes = []
    for bar_box in bars:
        if _in_scoreboard_zone(bar_box, width, height):
            continue
        if _overlaps_player(bar_box, player_pos):
            continue

        current_health = _read_bar_hp(image, bar_box)
        if current_health is None:
            continue

        # A username above means it's an enemy, not a box.
        if username_classifier.is_likely_username(image, bar_box):
            continue

        bx, by, bw, bh = bar_box
        pseudo_radius = max(1, int(bw / 2.0))
        anchor_x = bx + bw // 2
        anchor_y = by + bh + int(1.3 * pseudo_radius)

        boxes.append({
            "center": (anchor_x, anchor_y),
            "radius": pseudo_radius,
            "bounding_box": (
                anchor_x - pseudo_radius,
                anchor_y - pseudo_radius,
                pseudo_radius * 2,
                pseudo_radius * 2,
            ),
            "health": {"bounding_box": bar_box, "current_health": current_health},
        })

    return boxes


def find_enemies_with_stats(image, scale_factor: float = 1.0, player_pos=None):
    """Scans the whole screen for enemy health bars (the primary, most

    reliable signal -- see find_enemy_health_bars()), then tries to also
    read each one's power-cube badge. A candidate is only kept if at least
    one of health or cubes could be read; since health is now found by its
    own shape directly rather than guessed from a ring, `current_health`
    should be non-None for essentially every kept candidate, and the cube
    badge remains a best-effort bonus signal on top of that.

    `player_pos`, if given as (x, y, radius) from getAnchor.find_player_position,
    is used to drop any candidate that's actually sitting on the local
    player rather than an enemy (see _overlaps_player).

    `scale_factor` is accepted for backward compatibility with existing
    callers but is no longer used: downscaling hurts both the bar's own
    shape match and the OCR crop quality, and bars/digits are small enough
    already that scanning at native resolution is worth the extra cost.
    """
    height, width = image.shape[:2]
    bars = find_enemy_health_bars(image)

    validated_enemies = []
    for bar_box in bars:
        if _in_scoreboard_zone(bar_box, width, height):
            continue
        if _overlaps_player(bar_box, player_pos):
            continue

        # Reject bar-shaped, bar-colored map clutter (crate tile rims, coin
        # piles, chest reward tags) up front: a real health bar always has
        # a floating username directly above it, terrain doesn't. See
        # username_classifier.py for why this is a trained classifier
        # rather than another hand-tuned color/shape check.
        if not username_classifier.is_likely_username(image, bar_box):
            continue

        bx, by, bw, bh = bar_box

        current_health = _read_bar_hp(image, bar_box)

        # Derive a rough character anchor below the bar so the cube-badge
        # search (which still works off the anchor+radius geometry used
        # elsewhere in this file) has somewhere to look above it.
        # NOTE: pseudo_radius is an approximation (bar width / 2) standing
        # in for the old ring radius -- unverified against real footage,
        # may need tuning.
        pseudo_radius = max(1, int(bw / 2.0))
        anchor_x = bx + bw // 2
        anchor_y = by + bh + int(1.3 * pseudo_radius)
        cube_info = find_enemy_cube_info(image, (anchor_x, anchor_y, pseudo_radius))

        # A readable HP number is now REQUIRED, not just one-of-two: with
        # the digit-gap bridging and taller OCR crop in place, every real
        # enemy bar in the test set OCRs successfully, while the remaining
        # false positives (e.g. a pink crate whose wood texture the cube
        # OCR misread as a digit) sneak through precisely on the weaker
        # cube-only path. Cubes stay as bonus info, not as validation.
        if current_health is None:
            continue

        validated_enemies.append({
            "center": (anchor_x, anchor_y),
            "radius": pseudo_radius,
            "bounding_box": (
                anchor_x - pseudo_radius,
                anchor_y - pseudo_radius,
                pseudo_radius * 2,
                pseudo_radius * 2,
            ),
            "health": {"bounding_box": bar_box, "current_health": current_health},
            "cubes": cube_info,
        })

    return validated_enemies