import cv2
import numpy as np

from getHealth import find_health_info


def find_ammo_info(image, player, health_info=None):
    """Finds the local player's ammo bar by anchoring directly below the

    health bar, using real-world asset size boundaries.

    `health_info` may be passed in when the caller has already run
    find_health_info for this frame (the live loop does), saving a full
    duplicate digit-line search + OCR.
    """
    height, width = image.shape[:2]
    anchor_x, anchor_y, anchor_radius = player

    if anchor_radius == 0:
        return {"bounding_box": None, "ammo_count": 0}

    # STEP 1: Anchor off the health bar located by find_health_info.
    # (This used to duplicate its own copy of the green-bar search with an
    # older, narrower ROI -- which silently drifted out of sync when the
    # health search was fixed, so on some captures health was found but
    # ammo still failed. Reusing the same locator keeps them consistent.)
    if health_info is None:
        health_info = find_health_info(image, player)
    if health_info["bounding_box"] is None:
        return {"bounding_box": None, "ammo_count": 0}

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
        return {"bounding_box": None, "ammo_count": 0}

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
        # and small loading bars (w down to 3 pixels)
        if 3 <= w <= 45 and 2 <= h <= 15 and area > 4:
            valid_segments.append((x, y, w, h))

    if not valid_segments:
        return {"bounding_box": None, "ammo_count": 0}

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

    return {"bounding_box": global_box, "ammo_count": ammo_count}