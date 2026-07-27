import cv2
import numpy as np
import pytesseract # type: ignore


def find_cube_info(image, player):
    """Finds the power cube counter above the player's head and extracts the

    count using OCR directly from the clean green mask channel.
    """
    height, width = image.shape[:2]

    # `player` may be None when the anchor detector could not locate (or could
    # not verify) the brawler. Every other HUD reader already tolerated that;
    # this one unpacked it blind and took down a live training run on tick one.
    if player is None:
        return {"bounding_box": None, "cube_count": None}
    anchor_x, anchor_y, anchor_radius = player

    if anchor_radius == 0:
        return {"bounding_box": None, "cube_count": None}

    # STEP 1: Define the vertical column ROI above the player
    roi_xmin = max(0, int(anchor_x - 1.3 * anchor_radius))
    roi_xmax = min(width, int(anchor_x + 1.3 * anchor_radius))
    # Top at 4.5r (was 3.8r): measured on a real frame the counter row sat
    # ~4.5 radii above the anchor and was getting clipped -> missing count.
    roi_ymin = max(0, int(anchor_y - 4.5 * anchor_radius))
    roi_ymax = max(0, int(anchor_y - 1.1 * anchor_radius))

    roi = image[roi_ymin:roi_ymax, roi_xmin:roi_xmax]
    if roi.size == 0:
        return {"bounding_box": None, "cube_count": None}

    # STEP 2: Generate the clean green mask
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv_roi, lower_green, upper_green)

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

    # If the highest element is too low, you have 0 cubes (badge row is hidden)
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

    # STEP 4: Isolate the digits directly from the binary mask.
    # The counter is a YELLOW lightning badge followed by GREEN digits, so the
    # badge never enters the green mask — the green row box already IS the
    # digits. The old "take the right ~65%" crop assumed the badge was green
    # and chopped the left third off the first digit (single digits misread,
    # "12" became "2"). Instead, cut at the badge's real position: any yellow
    # inside the row marks the badge, digits start right of it; no yellow
    # means the whole row is digits.
    yellow_row = cv2.inRange(
        hsv_roi[local_y1:local_y2, local_x1:local_x2],
        np.array([18, 120, 120]), np.array([35, 255, 255]),
    )
    yellow_cols = np.where(yellow_row.any(axis=0))[0]
    if len(yellow_cols) > 0:
        # Digits are the row contours that start entirely RIGHT of the badge
        # (the badge's green glow overlaps the bolt, so column-slicing the
        # mask would leave glow fragments that corrupt the digit read).
        badge_right_local = local_x1 + int(yellow_cols.max())
        digit_contours = [
            item for item in row_contours if item[0] > badge_right_local
        ]
    else:
        digit_contours = row_contours

    if not digit_contours:
        return {"bounding_box": best_box, "cube_count": None}

    dx1 = min(item[0] for item in digit_contours)
    dy1 = min(item[1] for item in digit_contours)
    dx2 = max(item[0] + item[2] for item in digit_contours)
    dy2 = max(item[1] + item[3] for item in digit_contours)
    number_mask = mask[dy1:dy2, dx1:dx2]

    if number_mask.size == 0 or cv2.countNonZero(number_mask) == 0:
        return {"bounding_box": best_box, "cube_count": None}

    # STEP 5: Read the digits. Fast path: template matching against the
    # game's fixed digit font (~1ms, digitReader.py) -- the counter can be
    # a single digit, hence min_digits=1. Tesseract only runs if no
    # templates have been harvested yet.
    try:
        from . import digitReader
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