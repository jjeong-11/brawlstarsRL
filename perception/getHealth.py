import cv2
import numpy as np
import pytesseract # type: ignore


def _ocr_crop(crop):
    """Reads a number from a BGR crop.

    FAST PATH: template matching against the game's fixed digit font
    (~1ms, see digitReader.py) -- validation on real footage showed it's
    both ~80x faster than tesseract AND more accurate on this font
    (tesseract misread "6800" as "5800" and truncated "11600" to "600").
    FALLBACK: the original tesseract pipeline, for blobs the matcher
    isn't confident about (or before templates have been harvested).
    """
    if crop.size == 0:
        return None

    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    try:
        from . import digitReader
        if digitReader.has_templates():
            _, native_binary = cv2.threshold(gray_crop, 215, 255, cv2.THRESH_BINARY)
            # No tesseract fallback when the matcher is unsure: validation
            # showed the matcher beats tesseract on this font, and "unsure"
            # almost always means the crop is non-digit clutter -- paying
            # ~100ms of tesseract per clutter candidate was exactly what
            # kept the live loop under 3fps.
            return digitReader.read_digits(native_binary)
    except ImportError:
        pass

    # Upscale first using CUBIC to reconstruct smooth font lines from small pixels
    resized_gray = cv2.resize(
        gray_crop, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC
    )

    # Threshold the smooth high-res image to cleanly isolate the white text
    _, text_mask = cv2.threshold(resized_gray, 215, 255, cv2.THRESH_BINARY)

    # Add an outer border canvas layer so characters never touch image boundaries
    final_pad = 20
    padded_mask = cv2.copyMakeBorder(
        text_mask, final_pad, final_pad, final_pad, final_pad,
        cv2.BORDER_CONSTANT, value=0,
    )

    # Invert to standard black text on a clean white background
    final_ocr_input = cv2.bitwise_not(padded_mask)

    # Run OCR using PSM 7 (Treat image as a single text line)
    config = "--psm 7 -c tessedit_char_whitelist=0123456789"
    text = pytesseract.image_to_string(final_ocr_input, config=config).strip()
    # Require at least 2 digits: brawler HP is never a single digit in
    # practice, but stray white specks (sparkles, icon fragments) below
    # the real HP line occasionally OCR as a lone digit and, with the
    # bottom-up search order, would win over the real readout above.
    if not text.isdigit() or len(text) < 2:
        return None
    return int(text)


def find_health_info(image, player):
    """Finds the player's HP value and health-bar position.

    Strategy note: earlier versions located the green bar contour first and
    OCR'd inside it. That broke in two separate ways on real captures: the
    white digits split the green mask into fragments, and on grass-heavy
    maps the gap-bridging fix then merged the bar with dashed-grass ground
    texture into one giant blob. The digits themselves turned out to be the
    sturdier landmark -- the HP number is always rendered as WHITE text
    with a dark outline (unlike usernames, which change color per map), in
    a band above the player's feet, at the bottom of the username/HP text
    stack. So this scans that band for white digit-lines, bottom-most
    first, and the first line that OCRs as a number IS the HP readout.
    The returned bounding_box is that digit line's box (which sits on/just
    above the green bar), and downstream anchoring (the ammo search) works
    off it the same way it worked off the old bar box.
    """
    height, width = image.shape[:2]

    # None means the anchor detector could not locate/verify the player. Treat
    # it exactly like a zero radius rather than raising — callers gate on this
    # already, but an unguarded unpack here once crashed a live training run
    # (see getCube.find_cube_info for the same fix).
    if player is None:
        return {"bounding_box": None, "current_health": None}
    anchor_x, anchor_y, anchor_radius = player

    if anchor_radius == 0:
        return {"bounding_box": None, "current_health": None}

    # STEP 1: A generous window above the player's feet. Generous on
    # purpose: the text stack's height above the ring varies per brawler
    # (taller sprites push it higher -- one real capture had it ~3.3 radii
    # up, outside the old 1.3-2.4x window).
    # Widened from (2.0 x, 1.2-3.6 y) radii: on real captures the HP line was
    # sometimes clipped by the top edge of the old window, which leaves a
    # partial digit blob that fails OCR rather than a clean miss.
    roi_xmin = max(0, int(anchor_x - 2.3 * anchor_radius))
    roi_xmax = min(width, int(anchor_x + 2.3 * anchor_radius))
    roi_ymin = max(0, int(anchor_y - 4.2 * anchor_radius))
    # The lower edge stays at 1.2 radii deliberately. Candidates are tried
    # BOTTOM-UP, and the orange ammo pips sit just below the HP line — extend
    # this down to 1.0 and the pip row wins as the "bottom-most digit line",
    # which then anchors the ammo search onto itself. Measured: it cost ~20% of
    # ammo reads.
    roi_ymax = max(0, int(anchor_y - 1.2 * anchor_radius))

    roi = image[roi_ymin:roi_ymax, roi_xmin:roi_xmax]
    if roi.size == 0:
        return {"bounding_box": None, "current_health": None}

    value, local_box = find_digit_line(roi, scale=anchor_radius / _REF_RADIUS)
    if value is None:
        return {"bounding_box": None, "current_health": None}

    x, y, w, h = local_box
    global_box = (x + roi_xmin, y + roi_ymin, w, h)
    return {"bounding_box": global_box, "current_health": value}


# The digit-line size filters and the gap-bridging kernel below were tuned on
# captures where the player ring measured ~50px. But the anchor radius varies
# from ~25 to ~150px across real footage (brawler sprite size and the game's
# mild zoom), so fixed PIXEL thresholds silently stop matching at other scales:
# at r=25 real digits fall under the w>=18 floor, and at r=150 a 25px bridge is
# too narrow to join them into one blob. Everything is now expressed relative
# to this reference radius and scaled by the caller's anchor.
_REF_RADIUS = 50.0


def find_digit_line(roi, scale: float = 1.0):
    """Finds and OCRs the bottom-most white digit-line inside a BGR ROI.

    Shared by find_health_info (which builds its ROI above the player
    anchor) and getAnchor (which probes candidate bar regions when locating
    the player in the first place). Returns (value, (x, y, w, h)) in ROI
    coordinates, or (None, None).

    `scale` is anchor_radius / 50: pass it whenever the caller knows the
    player's on-screen size, so the size filters track the actual digit size
    instead of assuming a mid-range zoom.
    """
    if roi.size == 0:
        return None, None
    scale = max(0.35, min(3.0, float(scale)))

    # Mask the near-white digit pixels (high value, low saturation -- the
    # digit fill), then bridge the small gaps between neighboring digits
    # horizontally so each number becomes one line-shaped blob.
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 90, 255])
    white_mask = cv2.inRange(hsv_roi, lower_white, upper_white)

    bridge_w = max(7, int(round(25 * scale)))
    bridge_h = max(3, int(round(3 * scale)))
    bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (bridge_w, bridge_h))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, bridge)

    contours, _ = cv2.findContours(
        white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Digit-line shaped: wider than tall, tall enough to be text,
        # not a huge background wash. Thresholds scale with the player's
        # on-screen size (see _REF_RADIUS).
        w_min = max(8, int(round(18 * scale)))
        h_min = max(4, int(round(7 * scale)))
        h_max = max(h_min + 4, int(round(45 * scale)))
        if w < w_min or not (h_min <= h <= h_max) or w < h * 1.2:
            continue
        candidates.append((x, y, w, h))

    # Try candidates bottom-up. The HP number sits at the BOTTOM of the
    # text stack (username above it), so the first line from the bottom
    # that parses as digits is the HP readout. Things like the shield icon
    # or map decorations may produce line-shaped white blobs too -- they
    # simply fail OCR and get skipped.
    candidates.sort(key=lambda c: c[1], reverse=True)

    for (x, y, w, h) in candidates:
        pad_x = int(w * 0.15) + 2
        pad_y = int(h * 0.3) + 2
        crop = roi[
            max(0, y - pad_y):min(roi.shape[0], y + h + pad_y),
            max(0, x - pad_x):min(roi.shape[1], x + w + pad_x),
        ]
        value = _ocr_crop(crop)
        if value is not None:
            return value, (x, y, w, h)

    return None, None
