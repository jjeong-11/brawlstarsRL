"""
perception/mapdb.py
===================

Ground-truth map geometry, extracted from the layout renders in
`media/showdownmaps/`.

WHY THIS EXISTS
---------------
Everything the planner knew about the world used to be inferred from the
current frame by `getTerrain`, which needs a hand-calibrated HSV profile per
map skin. That approach has a hard ceiling: six profiles exist, Showdown
rotates skins faster than they can be measured, and below 30% coverage
`find_terrain` correctly refuses to return anything at all — at which point the
planner has no walls.

But the maps are not secret. Each one has a published top-down layout, and 72
of them are sitting in `media/showdownmaps/`. If we can work out WHICH map is
in play and WHERE on it the camera is, we get the true occupancy grid for the
whole map for free, including the parts currently off screen and — the thing
local segmentation can never supply — the real map BORDER.

This module does the first half: turning a render into an occupancy grid.
`perception/localize.py` does the second half.

THE CLASSIFICATION RULE, AND WHY IT IS NOT A COLOUR TABLE
---------------------------------------------------------
Writing another per-map palette table here would reproduce the exact problem
this is meant to solve. So the rule is relative, not absolute:

    walkable  =  the pixel is the FLOOR colour of the map (within tolerance)
    blocked   =  everything else, including everything outside the alpha mask

and the floor colour is identified by CONNECTIVITY — the largest connected
single-colour region — rather than by area. See `floor_colour` for why the
obvious "floor = most common colour" rule is wrong, and wrong specifically on
the dense maze maps where getting it right matters most.

Walls, crates, water, fences and decoration are all "not floor" and all block,
which is the same positive framing `getTerrain` already uses ("enumerating
obstacles is an endless list; the walkable surfaces are fixed").

Two known inaccuracies, both deliberate:

  * BUSHES COUNT AS BLOCKED HERE. They are walkable in game, and `getTerrain`
    treats them as such. Treating them as obstacles in the reference map is
    wrong in the ~5-10% of cells they occupy — but the reference map is used
    for LOCALISATION and for the border, while the live grid remains the
    authority on what is walkable right now. Cross-correlation tolerates a
    uniform minority disagreement far better than it tolerates a palette table
    that returns nothing at all on an unknown skin.
  * SPAWN MARKERS AND BOX ICONS are drawn on top of the floor and therefore
    read as blocked. They are small, fixed, and blocked-ish in practice
    (a power cube box really is an obstacle until broken).

`--inspect` renders the classification over the source image so this can be
eyeballed per map rather than taken on faith.

THE ALPHA CHANNEL IS THE VALUABLE PART
--------------------------------------
The renders are transparent outside the playable area. That is an exact map
boundary, which nothing in the live pipeline can produce — `getTerrain` infers
"near the edge" from how much unwalkable decoration surrounds a cell, which is
a proxy that the planner then has to spend a `border_cost` term compensating
for. With the real boundary, "is this in bounds" stops being a guess.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parent.parent
MAPS_DIR = _ROOT / "media" / "showdownmaps"
CACHE_PATH = _ROOT / "perception" / "map_occupancy.npz"

# Cells across the long side of a map. Showdown arenas are ~60x60 tiles; 64
# keeps roughly one cell per game tile, which is the resolution the layouts are
# actually drawn at, so going finer only interpolates.
MAP_GRID = 64

# How far a pixel may sit from the modal colour and still count as floor.
# Generous because the renders are anti-aliased and lightly textured (the
# checker pattern on the ground varies by a few units per channel), tight
# enough that a wall block never lands inside it: measured across all 72 maps,
# floor spreads ~8 units per channel while the nearest non-floor class sits
# 25+ away.
FLOOR_TOLERANCE = 18.0

# A cell is walkable when at least this fraction of its pixels are floor.
# Mirrors getTerrain.WALKABLE_FRACTION so the reference and live grids mean
# the same thing.
WALKABLE_FRACTION = 0.55


@dataclass(frozen=True)
class GameMap:
    """One Showdown arena."""
    name: str
    occupancy: np.ndarray     # (gh, gw) bool — True = BLOCKED
    in_bounds: np.ndarray     # (gh, gw) bool — inside the playable area
    source_size: Tuple[int, int]   # (w, h) of the render, in pixels

    @property
    def shape(self) -> Tuple[int, int]:
        return self.occupancy.shape

    def __repr__(self) -> str:
        gh, gw = self.occupancy.shape
        return (f"GameMap({self.name!r}, {gw}x{gh}, "
                f"{100 * self.occupancy.mean():.0f}% blocked)")


# --------------------------------------------------------------------------- #
def _colour_candidates(bgr: np.ndarray, mask: np.ndarray, top: int = 6):
    """The `top` most common colours among masked pixels, coarsely binned.

    Exact-mode would be defeated by anti-aliasing (60k distinct colours in a
    render that has maybe eight real ones), so colours are quantised to a
    16-unit grid and each bin is represented by the mean of its members.
    """
    px = bgr[mask]
    if px.size == 0:
        return []
    q = px.astype(np.int32) // 16
    keys = q[:, 0] * 4096 + q[:, 1] * 64 + q[:, 2]
    vals, counts = np.unique(keys, return_counts=True)
    order = np.argsort(-counts)[:top]
    return [px[keys == vals[i]].astype(np.float32).mean(axis=0) for i in order]


def floor_colour(bgr: np.ndarray, mask: np.ndarray,
                 tolerance: float = None) -> np.ndarray:
    """The colour of the walkable ground, chosen by CONNECTIVITY not by area.

    The obvious rule — "the floor is the most common colour" — is wrong, and
    wrong in a way that only shows up on the maps where it matters most. On a
    dense maze arena the wall blocks cover more of the render than the ground
    does, so the modal colour IS the wall, and the extraction then reports the
    map as ~100% blocked. Measured on this set: `NruhC nrevaC` came out at
    100.0% blocked and `Island Invasion` at 72.8%, against a 45% median.

    Connectivity is the property that actually distinguishes them, and it comes
    straight from what a map IS. The floor is one connected region, because a
    Showdown arena you could not walk across would be unplayable. Walls are
    many separate blocks by design — that is what makes them cover. So each
    candidate colour is scored by the size of its LARGEST CONNECTED COMPONENT
    rather than by its total area, and the ground wins even when it is
    outnumbered.
    """
    tolerance = FLOOR_TOLERANCE if tolerance is None else tolerance
    candidates = _colour_candidates(bgr, mask)
    if not candidates:
        return np.array([0, 0, 0], np.float32)

    # Scoring runs on a downscaled copy: we are choosing between a handful of
    # colours, not producing the final grid, and connected-components on a
    # 1000x1000 image per candidate is pure waste.
    h, w = bgr.shape[:2]
    scale = 256.0 / max(h, w)
    small = cv2.resize(bgr, (max(1, int(w * scale)), max(1, int(h * scale))),
                       interpolation=cv2.INTER_NEAREST).astype(np.float32)
    small_mask = cv2.resize(mask.astype(np.uint8),
                            (small.shape[1], small.shape[0]),
                            interpolation=cv2.INTER_NEAREST) > 0

    best, best_score = candidates[0], -1.0
    for col in candidates:
        hit = (np.linalg.norm(small - col, axis=2) <= tolerance) & small_mask
        if not hit.any():
            continue
        n, _, stats, _ = cv2.connectedComponentsWithStats(
            hit.astype(np.uint8), connectivity=8)
        if n <= 1:
            continue
        largest = stats[1:, cv2.CC_STAT_AREA].max()
        if largest > best_score:
            best, best_score = col, float(largest)
    return best


# Above this the extraction has clearly failed rather than found a hard map:
# a playable Showdown arena is 30-55% obstacles. Mirrors
# getTerrain.MAX_BLOCKED_FRACTION, and for the same reason -- a map the agent
# believes is solid is worse than no map at all.
MAX_BLOCKED_FRACTION = 0.80


def occupancy_from_render(path, grid_long: int = MAP_GRID,
                          tolerance: float = FLOOR_TOLERANCE,
                          walkable_fraction: float = WALKABLE_FRACTION):
    """Render -> (occupancy, in_bounds, (w, h)). See the module docstring."""
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(path)
    h, w = im.shape[:2]

    if im.shape[2] == 4:
        inb = im[..., 3] >= 128
        bgr = im[..., :3]
    else:
        inb = np.ones((h, w), bool)
        bgr = im

    floor_col = floor_colour(bgr, inb, tolerance)
    dist = np.linalg.norm(bgr.astype(np.float32) - floor_col, axis=2)
    is_floor = (dist <= tolerance) & inb

    # Pool to the grid. Cells are square in MAP pixels, so a non-square map
    # gets a non-square grid — the arena's aspect ratio is preserved, which
    # matters because the live patch is matched against this by shape.
    if w >= h:
        gw = int(grid_long)
        gh = max(1, int(round(grid_long * h / w)))
    else:
        gh = int(grid_long)
        gw = max(1, int(round(grid_long * w / h)))

    floor_frac = cv2.resize(is_floor.astype(np.float32), (gw, gh),
                            interpolation=cv2.INTER_AREA)
    bounds_frac = cv2.resize(inb.astype(np.float32), (gw, gh),
                             interpolation=cv2.INTER_AREA)

    in_bounds = bounds_frac >= 0.5
    # Out of bounds is blocked, not unknown: it is the one thing about a
    # Showdown map that is certain.
    occupancy = (floor_frac < walkable_fraction) | (~in_bounds)
    return occupancy, in_bounds, (w, h)


# --------------------------------------------------------------------------- #
class MapDatabase:
    """All known arenas, built once and cached to an npz."""

    def __init__(self, maps: Optional[List[GameMap]] = None):
        self.maps: List[GameMap] = maps or []

    def __len__(self) -> int:
        return len(self.maps)

    def __iter__(self):
        return iter(self.maps)

    def get(self, name: str) -> Optional[GameMap]:
        for m in self.maps:
            if m.name == name:
                return m
        return None

    # ----------------------------------------------------------------- #
    @classmethod
    def build(cls, maps_dir=MAPS_DIR, grid_long: int = MAP_GRID) -> "MapDatabase":
        maps = []
        for path in sorted(pathlib.Path(maps_dir).glob("*.png")):
            name = path.stem.replace("-Map", "").replace("_", " ")
            try:
                occ, inb, size = occupancy_from_render(path, grid_long)
            except Exception as e:                       # skip a bad render
                print(f"  SKIP {name}: {e}")
                continue
            # A map we could not read is worse than a map we do not have: the
            # localiser would try to match against a solid wall and either fail
            # to lock or, worse, lock wrongly. Refuse it, the same way
            # getTerrain refuses a profile that does not fit.
            if occ.mean() > MAX_BLOCKED_FRACTION:
                print(f"  SKIP {name}: {100 * occ.mean():.0f}% blocked, "
                      f"extraction failed (see --inspect)")
                continue
            maps.append(GameMap(name=name, occupancy=occ, in_bounds=inb,
                                source_size=size))
        return cls(maps)

    def save(self, path=CACHE_PATH) -> None:
        payload: Dict[str, np.ndarray] = {}
        names = []
        for m in self.maps:
            names.append(m.name)
            payload[f"occ::{m.name}"] = m.occupancy
            payload[f"inb::{m.name}"] = m.in_bounds
            payload[f"siz::{m.name}"] = np.array(m.source_size, np.int32)
        payload["__names__"] = np.array(names, dtype=object)
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path=CACHE_PATH, build_if_missing: bool = True) -> "MapDatabase":
        path = pathlib.Path(path)
        if not path.exists():
            if not build_if_missing:
                return cls([])
            db = cls.build()
            try:
                db.save(path)
            except Exception:
                pass                                     # read-only checkout
            return db
        data = np.load(path, allow_pickle=True)
        maps = []
        for name in data["__names__"]:
            name = str(name)
            maps.append(GameMap(name=name,
                                occupancy=data[f"occ::{name}"],
                                in_bounds=data[f"inb::{name}"],
                                source_size=tuple(data[f"siz::{name}"])))
        return cls(maps)


_DB: Optional[MapDatabase] = None


def database() -> MapDatabase:
    """Process-wide database, loaded lazily."""
    global _DB
    if _DB is None:
        _DB = MapDatabase.load()
    return _DB


# --------------------------------------------------------------------------- #
def inspect_overlay(path, grid_long: int = MAP_GRID) -> np.ndarray:
    """The classification drawn over the render, for eyeballing.

    Red = blocked, unchanged = walkable floor, dark = out of bounds.
    """
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    bgr = im[..., :3].copy()
    occ, inb, _ = occupancy_from_render(path, grid_long)
    gh, gw = occ.shape
    h, w = bgr.shape[:2]

    big_occ = cv2.resize(occ.astype(np.uint8) * 255, (w, h),
                         interpolation=cv2.INTER_NEAREST) > 0
    big_inb = cv2.resize(inb.astype(np.uint8) * 255, (w, h),
                         interpolation=cv2.INTER_NEAREST) > 0

    tint = bgr.copy()
    tint[big_occ] = (0, 0, 255)
    out = cv2.addWeighted(bgr, 0.55, tint, 0.45, 0)
    out[~big_inb] = (30, 30, 30)

    cv2.putText(out, f"{pathlib.Path(path).stem}  {gw}x{gh}  "
                f"{100 * occ.mean():.0f}% blocked", (14, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true", help="rebuild the npz cache")
    ap.add_argument("--inspect", nargs="*", metavar="MAP",
                    help="write classification overlays to debugOutput/maps/ "
                         "(no names = the first six)")
    args = ap.parse_args()

    if args.build or not CACHE_PATH.exists():
        print(f"building from {MAPS_DIR} ...")
        db = MapDatabase.build()
        db.save()
        print(f"  {len(db)} maps -> {CACHE_PATH} "
              f"({CACHE_PATH.stat().st_size / 1024:.0f} KB)")
    else:
        db = MapDatabase.load()
        print(f"{len(db)} maps loaded from {CACHE_PATH.name}")

    blocked = np.array([m.occupancy.mean() for m in db])
    print(f"blocked fraction: min {100 * blocked.min():.0f}%  "
          f"median {100 * np.median(blocked):.0f}%  max {100 * blocked.max():.0f}%")
    worst = sorted(db, key=lambda m: -m.occupancy.mean())[:5]
    print("most-blocked (check these first with --inspect):")
    for m in worst:
        print(f"   {m.name:32s} {100 * m.occupancy.mean():5.1f}%")

    if args.inspect is not None:
        out_dir = _ROOT / "debugOutput" / "maps"
        out_dir.mkdir(parents=True, exist_ok=True)
        names = args.inspect or [m.name for m in db][:6]
        for name in names:
            src = MAPS_DIR / f"{name.replace(' ', '_')}-Map.png"
            if not src.exists():
                matches = list(MAPS_DIR.glob(f"*{name.replace(' ', '_')}*"))
                if not matches:
                    print(f"  no render for {name!r}")
                    continue
                src = matches[0]
            dest = out_dir / f"{src.stem}_classified.png"
            cv2.imwrite(str(dest), inspect_overlay(src))
            print(f"  wrote {dest}")
