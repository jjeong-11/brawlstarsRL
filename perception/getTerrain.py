"""
perception/getTerrain.py
========================

Local terrain segmentation -> a walkability occupancy grid for path planning.

Unlike the entity detectors, this one is deliberately *coarse*: the planner
does not need pixel-perfect wall outlines, it needs to know which cells of a
small grid it can walk through. So we classify every pixel into one of three
terrain classes, then pool the result down into a grid.

WALKABILITY RULE — walkable = floor OR bush, everything else is an obstacle.

Stated positively on purpose: enumerating obstacles ("walls, cones, barrels,
fences, the out-of-bounds junk past the map border...") is an endless list that
grows with every map, while the two walkable surfaces are fixed.

MAP PROFILES — AND WHY THEY EXIST
---------------------------------
Showdown rotates map skins with completely different palettes, and a single set
of HSV bounds does not survive that. Two maps measured from this project's own
recordings:

    class   night_teal (fixtures/showdown.png)   purple_stone (test_game2.mp4)
    -----   -------------------------      -----------------------------
    floor   H 124-128  S 114-119  V  76-82  H 130-141  S 157-203  V 104-125
    wall    H 118-120  S 114-131  V  77-184 H 128-130  S 178-193  V 137-198
    bush    H  94- 97  S 229-232  V 144-188 H  52- 57  S 106-116  V 204-244

Note they are not merely offset — they separate along DIFFERENT AXES. On
night_teal, floor and wall are told apart by hue (a 6-8 degree gap) at similar
value. On purple_stone they share hue almost exactly and are told apart by
value. No single rule covers both, which is why this is a profile table rather
than one widened set of bounds.

Applying night_teal's constants to purple_stone classifies 1.2% of the frame.
Since unrecognised pixels count as obstacles, that renders 86% of the map solid
and the planner concludes it is walled in — so `select_profile` scores every
profile against the frame and picks the best, and `MIN_PROFILE_COVERAGE`
refuses to return a grid at all when none of them fit. Refusing is important:
a caller that gets None falls back to direct steering, which is far better than
A* routing around imaginary walls.

Selection is sticky (see `ProfileSelector`) because the map cannot change
mid-match; re-scoring every frame would be wasted work.

ADDING A MAP
------------
    python -m perception.getTerrain --calibrate your_screenshot.png
prints a ready-to-paste profile. See `calibrate_from_sample` for labelling
patches by hand when the automatic guess is not good enough.

KNOWN LIMITATIONS
  * Two profiles so far. A third map will likely need a third entry.
  * Water/lava (impassable but not wall-coloured) falls out as "not walkable"
    by the rule above, which is correct, but is untested.
  * Bushes are walkable AND concealing; the grid reports the bush fraction
    separately so the planner can prefer routing through cover.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

Bounds = Tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class TerrainProfile:
    """HSV bounds for one map palette."""
    name: str
    floor: Bounds
    wall: Bounds
    bush: Bounds
    notes: str = ""


# --- CALIBRATION CONSTANTS -------------------------------------------------
# Bounds are measured percentiles widened by roughly +-4 hue / +-25 sat to
# absorb lighting variation between maps of the same theme.

#this was added for the map: GatedCommunity on 7/27/2026
GATEDCOMMUNITY = TerrainProfile(
    name='GatedCommunity',
    floor=(np.array([np.int64(134), np.int64(42), np.int64(24)]), np.array([np.int64(147), np.int64(136), np.int64(139)])),
    wall=(np.array([np.int64(111), np.int64(50), np.int64(123)]), np.array([np.int64(125), np.int64(147), np.int64(238)])),
    bush=(np.array([np.int64(88), np.int64(188), np.int64(91)]), np.array([np.int64(101), np.int64(255), np.int64(228)])),
)

NIGHT_TEAL = TerrainProfile(
    name="night_teal",
    floor=(np.array([123, 100, 55]), np.array([130, 135, 100])),
    wall=(np.array([113, 95, 70]), np.array([122, 145, 255])),
    bush=(np.array([88, 170, 80]), np.array([102, 255, 255])),
    notes="media/fixtures/showdown.png. Matte purple floor, teal bushes. Floor/wall split by HUE.",
)

PURPLE_STONE = TerrainProfile(
    name="purple_stone",
    floor=(np.array([125, 121, 50]), np.array([145, 236, 167])),
    wall=(np.array([116, 107, 100]), np.array([135, 231, 238])),
    # V capped at 160, just under the gas band's floor of 165: this is the one
    # profile whose bushes overlap gas in BOTH hue and saturation, so value is
    # the only channel left to separate them on. The clipped range still covers
    # the bush cluster comfortably (centre V 103).
    bush=(np.array([52, 130, 39]), np.array([66, 225, 160])),
    notes="test_game2.mp4. Purple stone tiles, cyan-trimmed crates, dark-green "
          "spiky grass. CORRECTED: the bush class here was originally "
          "calibrated on the pale-green puffs, which are GAS, not foliage. The "
          "real bushes share their hue (H~58 vs gas H~64) and are told apart by "
          "saturation and value -- bush S 169 V 103, gas S 101 V 231.",
)

GRAVEYARD = TerrainProfile(
    name="graveyard",
    floor=(np.array([132, 41, 23]), np.array([147, 147, 140])),
    wall=(np.array([111, 48, 102]), np.array([126, 152, 239])),
    bush=(np.array([88, 166, 47]), np.array([105, 255, 221])),
    notes="Gas screenshots, night graveyard. Same family as night_teal but the "
          "floor sits ~10 hue higher (H 136 vs 126), outside night_teal's band.",
)

STARR_RAIL = TerrainProfile(
    name="starr_rail",
    floor=(np.array([164, 70, 91]), np.array([176, 165, 209])),
    wall=(np.array([167, 46, 167]), np.array([179, 141, 255])),
    bush=(np.array([88, 179, 103]), np.array([102, 255, 233])),
    notes="Gas screenshots, Starr Rail station. Maroon cobble floor, pink "
          "blocks, teal grass. Floor/wall split by VALUE at similar hue.",
)

MAGENTA_CRATE = TerrainProfile(
    name="magenta_crate",
    floor=(np.array([152, 185, 120]), np.array([167, 235, 165])),
    wall=(np.array([138, 140, 112]), np.array([151, 225, 215])),
    bush=(np.array([16, 140, 85]), np.array([42, 215, 145])),
    notes="test_game4.mp4. Magenta floor, purple crates, dark-green grass. "
          "Floor/wall split by HUE across a narrow gap at ~151.",
)

PROFILES = (NIGHT_TEAL, PURPLE_STONE, MAGENTA_CRATE, GRAVEYARD, STARR_RAIL, GATEDCOMMUNITY)

# Fraction of the play area a profile must classify to be trusted. Below this
# we return no grid rather than a mostly-solid fantasy. Real matches on a
# matching profile measure 46-62%; a mismatched profile measures ~1%, so this
# threshold is nowhere near either edge.
MIN_PROFILE_COVERAGE = 0.30

# Second, independent sanity check on the OUTPUT rather than the input.
#
# Profile selection is sticky (the map cannot change mid-match), which means a
# profile stays in use across frames where it no longer actually fits — a
# full-screen super effect, a death overlay, a heavy gas tint. Measured: on one
# test_game4 frame the chosen profile explained 0.1% of the image yet the
# sticky selector kept it, yielding a grid that was 86% blocked.
#
# A Showdown map is 30-50% obstacles. Anything past this is not a maze, it is a
# failed classification, and returning None (-> direct steering) beats letting
# A* route around walls that are not there.
MAX_BLOCKED_FRACTION = 0.72

# Back-compat aliases (the module used to export bare constants).
FLOOR_LOWER, FLOOR_UPPER = NIGHT_TEAL.floor
WALL_LOWER, WALL_UPPER = NIGHT_TEAL.wall
BUSH_LOWER, BUSH_UPPER = NIGHT_TEAL.bush

# Letterbox: phone captures pad the 20:9 panel out to the frame with pure
# black (measured V=5). Anything this dark is padding, not terrain.
LETTERBOX_V_MAX = 15

# Default planning grid. 48x27 keeps the cell roughly square on a 20:9 panel
# and keeps A* under ~1300 nodes, which costs well under a millisecond.
GRID_W = 48
GRID_H = 27

# A cell counts as walkable when at least this fraction of its pixels are.
# Below 0.5 the planner cuts corners through wall edges; above ~0.75 it
# refuses to use legitimate one-cell-wide gaps between blocks.
WALKABLE_FRACTION = 0.55

# HUD overlays sit ON TOP of real terrain, so the pixels underneath cannot be
# classified. These regions are marked UNKNOWN and inherit the walkability of
# their neighbours rather than becoming phantom walls. Fractions of the PLAY
# RECT (letterbox excluded), measured on media/fixtures/showdown.png.
_HUD_CIRCLES = (   # (cx, cy, r)
    (0.150, 0.757, 0.150),   # movement joystick + its outer ring
    (0.915, 0.808, 0.115),   # attack button + ring
    (0.733, 0.744, 0.055),   # super (skull) button
)
_HUD_RECTS = (     # (x0, y0, x1, y1)
    (0.000, 0.000, 0.300, 0.105),   # "Brawlers left: N"
    (0.760, 0.000, 1.000, 0.120),   # chat bubble
    (0.230, 0.760, 0.330, 1.000),   # power-cube counter + portrait
)

_CLEAN_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
_CLEAN_KERNEL_SMALL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# Classification runs on a downscaled copy. The output is a 48x27 grid, so full
# 1080p detail is thrown away by the pooling anyway -- at this height each grid
# cell still gets ~10x10 source pixels, and it takes the segmentation from
# ~65ms to under 1ms. The live loop budgets ~100ms for ALL stages combined.
WORK_HEIGHT = 270
# ---------------------------------------------------------------------------


def play_rect(image_bgr: np.ndarray) -> tuple:
    """(x, y, w, h) of the real panel inside any black letterbox padding.

    Tested on the FRACTION of lit pixels per row/column, not the max: a single
    stray bright pixel would otherwise drag the rect into the padding.
    Subsampled every 8th pixel -- the bars are hundreds wide, and a full-frame
    scan at 1080p costs ~24ms on its own.
    """
    probe = image_bgr[::8, ::8, 0]
    lit = probe > LETTERBOX_V_MAX
    cols = np.where(lit.mean(axis=0) > 0.5)[0]
    rows = np.where(lit.mean(axis=1) > 0.5)[0]
    h, w = image_bgr.shape[:2]
    if cols.size == 0 or rows.size == 0:
        return (0, 0, w, h)
    x0, x1 = int(cols[0]) * 8, min(w - 1, int(cols[-1]) * 8 + 7)
    y0, y1 = int(rows[0]) * 8, min(h - 1, int(rows[-1]) * 8 + 7)
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)


def _hud_mask(h: int, w: int) -> np.ndarray:
    """Binary mask (255 = HUD) over a play rect of size (h, w)."""
    mask = np.zeros((h, w), np.uint8)
    for cx, cy, r in _HUD_CIRCLES:
        cv2.circle(mask, (int(cx * w), int(cy * h)), int(r * min(w, h)), 255, -1)
    for x0, y0, x1, y1 in _HUD_RECTS:
        cv2.rectangle(mask, (int(x0 * w), int(y0 * h)),
                      (int(x1 * w), int(y1 * h)), 255, -1)
    return mask


def terrain_masks(image_bgr: np.ndarray, hsv: np.ndarray = None,
                  profile: TerrainProfile = NIGHT_TEAL) -> dict:
    """Per-pixel floor / wall / bush / walkable masks for one profile."""
    if hsv is None:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    return _masks_from_hsv(hsv, profile)


def _masks_from_hsv(hsv: np.ndarray, profile: TerrainProfile) -> dict:
    floor = cv2.inRange(hsv, *profile.floor)
    wall = cv2.inRange(hsv, *profile.wall)
    bush = cv2.inRange(hsv, *profile.bush)

    # Speckle removal. Bushes are feathery and lose real area to an OPEN, so
    # only the flat classes get cleaned. Kernel scales with resolution: a 5x5
    # at 1080p is nothing, but at the 270px working height it eats wall blocks.
    kernel = _CLEAN_KERNEL if hsv.shape[0] > 540 else _CLEAN_KERNEL_SMALL
    floor = cv2.morphologyEx(floor, cv2.MORPH_OPEN, kernel)
    wall = cv2.morphologyEx(wall, cv2.MORPH_OPEN, kernel)

    walkable = cv2.bitwise_or(floor, bush)
    return {"floor": floor, "wall": wall, "bush": bush, "walkable": walkable}


def profile_coverage(hsv: np.ndarray, profile: TerrainProfile) -> float:
    """Fraction of pixels this profile can classify as ANY known terrain.

    The score used to pick a profile. A matching profile explains most of the
    frame; a mismatched one explains almost none, so the gap is wide and the
    choice is not delicate.
    """
    m = _masks_from_hsv(hsv, profile)
    known = cv2.bitwise_or(m["walkable"], m["wall"])
    return float(np.count_nonzero(known)) / max(1, known.size)


def select_profile(hsv: np.ndarray, profiles=PROFILES,
                   min_coverage: float = MIN_PROFILE_COVERAGE):
    """Best-scoring profile for this frame, or (None, best_score) if none fit."""
    best, best_score = None, 0.0
    for p in profiles:
        score = profile_coverage(hsv, p)
        if score > best_score:
            best, best_score = p, score
    if best_score < min_coverage:
        return None, best_score
    return best, best_score


class ProfileSelector:
    """Sticky profile choice: decide once, re-check occasionally.

    The map cannot change mid-match, so re-scoring every profile on every frame
    is pure waste. But the choice must not be permanent either -- the first
    frames of an episode can be a loading screen or a menu, where nothing
    matches and a naive one-shot selection would latch onto garbage for the
    whole match.
    """

    def __init__(self, profiles=PROFILES, recheck_every: int = 150,
                 min_coverage: float = MIN_PROFILE_COVERAGE):
        self.profiles = profiles
        self.recheck_every = max(1, int(recheck_every))
        self.min_coverage = min_coverage
        self.reset()

    def reset(self) -> None:
        self.profile: Optional[TerrainProfile] = None
        self.coverage: float = 0.0
        self._calls = 0

    def select(self, hsv: np.ndarray):
        due = (self.profile is None
               or self._calls % self.recheck_every == 0)
        self._calls += 1
        if due:
            profile, score = select_profile(hsv, self.profiles, self.min_coverage)
            # Never downgrade a working profile to None on one bad frame (a
            # death overlay, a full-screen super effect); only replace it when
            # something actually scores.
            if profile is not None or self.profile is None:
                self.profile, self.coverage = profile, score
        return self.profile, self.coverage


_DEFAULT_SELECTOR = ProfileSelector()


def _pool_fraction(mask: np.ndarray, gw: int, gh: int) -> np.ndarray:
    """Mean of a 0/255 mask over a gh x gw grid, as a 0..1 float array."""
    small = cv2.resize(mask, (gw, gh), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32) / 255.0


def find_terrain(image_bgr: np.ndarray, grid_size=(GRID_W, GRID_H),
                 hsv: np.ndarray = None, walkable_fraction: float = WALKABLE_FRACTION,
                 player_pos=None, profile: TerrainProfile = None,
                 selector: ProfileSelector = None) -> Optional[dict]:
    """Segment terrain and pool it into a walkability grid.

    Returns None when no profile matches the frame -- callers must treat that
    as "no terrain information", NOT as "everything is blocked". The planner
    falls back to direct steering.

    Otherwise a dict with
        occupancy : (gh, gw) bool  — True = BLOCKED
        walkable  : (gh, gw) float — fraction of the cell that is walkable
        bush      : (gh, gw) float — fraction that is bush (cover)
        unknown   : (gh, gw) bool  — cell was mostly hidden behind HUD
        rect      : (x, y, w, h)   — play rect these grids cover, in frame px
        cell      : (cw, ch)       — cell size in frame pixels
        profile   : TerrainProfile — which palette was used
        coverage  : float          — how much of the frame it explained
    """
    gw, gh = int(grid_size[0]), int(grid_size[1])
    x, y, w, h = play_rect(image_bgr)
    roi = image_bgr[y:y + h, x:x + w]

    # Classify at WORK_HEIGHT, not native resolution (see the constant).
    if h > WORK_HEIGHT:
        sw, sh = max(1, int(round(w * WORK_HEIGHT / h))), WORK_HEIGHT
        roi = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_AREA)
    else:
        sw, sh = w, h
    # A resized HSV image is not the HSV of the resized image (hue is circular,
    # so averaging across the 179/0 wrap is meaningless), so any HSV the caller
    # cached at full resolution cannot be reused here.
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    coverage = None
    if profile is None:
        profile, coverage = (selector or _DEFAULT_SELECTOR).select(roi_hsv)
        if profile is None:
            return None
    if coverage is None:
        coverage = profile_coverage(roi_hsv, profile)

    masks = _masks_from_hsv(roi_hsv, profile)
    hud = _hud_mask(sh, sw)
    known = cv2.bitwise_not(hud)

    walk_frac = _pool_fraction(cv2.bitwise_and(masks["walkable"], known), gw, gh)
    bush_frac = _pool_fraction(cv2.bitwise_and(masks["bush"], known), gw, gh)
    known_frac = _pool_fraction(known, gw, gh)

    # Renormalise by how much of the cell we could actually see, so a cell
    # half-covered by the joystick is judged on its visible half.
    visible = np.maximum(known_frac, 1e-3)
    walk_frac = np.clip(walk_frac / visible, 0.0, 1.0)
    bush_frac = np.clip(bush_frac / visible, 0.0, 1.0)

    unknown = known_frac < 0.25
    occupancy = walk_frac < walkable_fraction
    occupancy[unknown] = False   # not seeing a cell is not evidence of a wall

    # Output-side sanity check — catches the case where a sticky profile is
    # kept across a frame it no longer fits. See MAX_BLOCKED_FRACTION.
    if occupancy.mean() > MAX_BLOCKED_FRACTION:
        return None

    cell = (w / gw, h / gh)

    # The player's own sprite is not terrain, and a brawler standing on a cell
    # is proof that cell is walkable. Free the 3x3 around it so A* can start.
    if player_pos is not None:
        gx = int((player_pos[0] - x) / max(cell[0], 1e-6))
        gy = int((player_pos[1] - y) / max(cell[1], 1e-6))
        if 0 <= gx < gw and 0 <= gy < gh:
            occupancy[max(0, gy - 1):gy + 2, max(0, gx - 1):gx + 2] = False

    return {
        "occupancy": occupancy,
        "walkable": walk_frac,
        "bush": bush_frac,
        "unknown": unknown,
        "rect": (x, y, w, h),
        "cell": cell,
        "profile": profile,
        "coverage": coverage,
    }


def to_grid(point, rect, cell) -> tuple:
    """Frame pixel (x, y) -> (gx, gy) grid cell."""
    return (int((point[0] - rect[0]) / max(cell[0], 1e-6)),
            int((point[1] - rect[1]) / max(cell[1], 1e-6)))


def to_pixels(cell_xy, rect, cell) -> tuple:
    """Grid cell (gx, gy) -> the frame pixel at that cell's CENTRE."""
    return (rect[0] + (cell_xy[0] + 0.5) * cell[0],
            rect[1] + (cell_xy[1] + 0.5) * cell[1])


def calibrate_from_sample(image_bgr: np.ndarray, samples: dict, name="new_map",
                          pad_h=4, pad_s=25, pad_v=40) -> TerrainProfile:
    """Derive a profile from labelled patches on a new map.

    `samples` maps "floor"/"wall"/"bush" to lists of (x, y, radius) patches
    known to be that class. Prints a ready-to-paste TerrainProfile.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    bounds = {}
    for cls, patches in samples.items():
        px = [hsv[max(0, cy - r):cy + r, max(0, cx - r):cx + r].reshape(-1, 3)
              for cx, cy, r in patches]
        px = np.concatenate(px, axis=0)
        lo = np.clip(np.percentile(px, 2, axis=0) - (pad_h, pad_s, pad_v), 0, 255)
        hi = np.clip(np.percentile(px, 98, axis=0) + (pad_h, pad_s, pad_v), 0, 255)
        bounds[cls] = (lo.astype(int), hi.astype(int))

    print(f"{name.upper()} = TerrainProfile(")
    print(f"    name={name!r},")
    for cls in ("floor", "wall", "bush"):
        lo, hi = bounds[cls]
        print(f"    {cls}=(np.array({list(lo)}), np.array({list(hi)})),")
    print(")")
    print(f"# then add {name.upper()} to PROFILES")

    prof = TerrainProfile(name=name, floor=bounds["floor"],
                          wall=bounds["wall"], bush=bounds["bush"])
    print(f"# coverage on this frame: {100 * profile_coverage(hsv, prof):.1f}%")
    return prof


def debug_overlay(image_bgr: np.ndarray, terrain: dict = None) -> np.ndarray:
    """Colour-coded terrain overlay + the pooled grid, for eyeballing."""
    terrain = terrain if terrain is not None else find_terrain(image_bgr)
    if terrain is None:
        out = image_bgr.copy()
        cv2.putText(out, "NO TERRAIN PROFILE MATCHED", (60, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)
        return out

    x, y, w, h = terrain["rect"]
    cw, ch = terrain["cell"]
    out = image_bgr.copy()
    roi = out[y:y + h, x:x + w]
    m = terrain_masks(roi, profile=terrain["profile"])
    tint = roi.copy()
    tint[m["wall"] > 0] = (0, 0, 255)
    tint[m["bush"] > 0] = (0, 255, 0)
    tint[m["floor"] > 0] = (255, 0, 0)
    out[y:y + h, x:x + w] = cv2.addWeighted(roi, 0.4, tint, 0.6, 0)

    occ = terrain["occupancy"]
    for gy in range(occ.shape[0]):
        for gx in range(occ.shape[1]):
            if occ[gy, gx]:
                cv2.rectangle(out, (int(x + gx * cw), int(y + gy * ch)),
                              (int(x + (gx + 1) * cw), int(y + (gy + 1) * ch)),
                              (255, 255, 255), 1)
    cv2.putText(out, f"{terrain['profile'].name}  {100 * terrain['coverage']:.0f}% "
                f"coverage  {100 * occ.mean():.0f}% blocked", (x + 20, y + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
    return out


def _auto_patches(image_bgr):
    """Rough automatic patch guess: the three biggest colour modes in the frame.

    A starting point for --calibrate, not a substitute for eyeballing the
    result. Floor is assumed to be the most common surface, bushes the most
    saturated-and-bright, walls the remaining large mode.
    """
    x, y, w, h = play_rect(image_bgr)
    roi = cv2.resize(image_bgr[y:y + h, x:x + w], (240, 135),
                     interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    _, labels, centres = cv2.kmeans(
        hsv, 4, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0),
        4, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.ravel(), minlength=4)
    order = np.argsort(-counts)
    return centres, counts, order


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image", nargs="?", help="screenshot to analyse")
    ap.add_argument("--calibrate", action="store_true",
                    help="print the k-means colour modes to help label patches")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    path = args.image or str(root / "media" / "fixtures" / "showdown.png")
    image = cv2.imread(path)
    if image is None:
        raise SystemExit(f"could not read {path}")

    print(f"{path}   {image.shape[1]}x{image.shape[0]}")
    print(f"play rect {play_rect(image)}\n")

    x, y, w, h = play_rect(image)
    small = cv2.resize(image[y:y + h, x:x + w], (240, 135), interpolation=cv2.INTER_AREA)
    shsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    print("profile scores:")
    for p in PROFILES:
        print(f"   {p.name:14s} {100 * profile_coverage(shsv, p):5.1f}% coverage")

    if args.calibrate:
        centres, counts, order = _auto_patches(image)
        print("\ndominant HSV modes (largest first) — use these to pick patches:")
        for i in order:
            print(f"   HSV {centres[i].astype(int)}  {100 * counts[i] / counts.sum():5.1f}% of frame")
        print("\nthen call calibrate_from_sample(img, {'floor': [(x,y,r), ...], ...})")
        raise SystemExit(0)

    t = find_terrain(image)
    if t is None:
        print("\nNO PROFILE MATCHED — the planner would fall back to direct "
              "steering on this frame.\nRe-run with --calibrate to add this map.")
    else:
        occ = t["occupancy"]
        print(f"\nchosen: {t['profile'].name}  ({100 * t['coverage']:.1f}% coverage)")
        print(f"grid {occ.shape[1]}x{occ.shape[0]}  blocked {100 * occ.mean():.1f}%  "
              f"unknown {100 * t['unknown'].mean():.1f}%")

    out_dir = root / "debugOutput"
    out_dir.mkdir(exist_ok=True)
    dest = out_dir / "terrain_debug.png"
    cv2.imwrite(str(dest), debug_overlay(image, t))
    print(f"wrote {dest}")
