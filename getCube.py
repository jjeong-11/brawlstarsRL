import cv2
import numpy as np
import pytesseract # type: ignore


def find_cube_info(image, player):
    """Finds the power cube counter above the player's head and extracts the

    count using OCR directly from the clean green mask channel.
    """
    height, width = image.shape[:2]
    anchor_x, anchor_y, anchor_radius = player

    if anchor_radius == 0:
        return {"bounding_box": None, "cube_count": None}

    # STEP 1: Define the vertical column ROI above the player
    roi_xmin = max(0, int(anchor_x - 1.3 * anchor_radius))
    roi_xmax = min(width, int(anchor_x + 1.3 * anchor_radius))
    roi_ymin = max(0, int(anchor_y - 3.8 * anchor_radius))
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

    # STEP 4: Isolate the digits directly from the binary mask
    # Crop the row out of the mask channel instead of the raw BGR image
    row_mask = mask[local_y1:local_y2, local_x1:local_x2]

    # The badge is on the left, digits are on the right (take the right ~65%)
    num_w = int(row_w * 0.65)
    num_x = row_w - num_w
    number_mask = row_mask[:, num_x:]

    if number_mask.size == 0:
        return {"bounding_box": best_box, "cube_count": None}

    # STEP 5: Read the digits. Fast path: template matching against the
    # game's fixed digit font (~1ms, digitReader.py) -- the counter can be
    # a single digit, hence min_digits=1. Tesseract only runs if no
    # templates have been harvested yet.
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