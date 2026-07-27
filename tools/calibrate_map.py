#!/usr/bin/env python3
"""
tools/calibrate_map.py
======================

Derive a TerrainProfile (and check the gas band) from a screenshot, by colour
clustering instead of hand-picked patches.

WHY CLUSTERING RATHER THAN PATCHES
    Hand-labelling "this pixel is floor" requires eyeballing coordinates, and
    getting one wrong silently poisons a profile. That already happened once in
    this project: the `purple_stone` bush class was calibrated on what turned
    out to be GAS, because pale-green gas puffs and that map's bushes sit at
    almost the same hue. k-means over the play area finds the real modes
    without anyone guessing at coordinates.

THE GAS / BUSH SEPARATOR
    Measured across four map themes, gas and green bushes routinely COLLIDE IN
    HUE and are separated by saturation and value instead:

        gas    H 45-70   S  90-135  V 195-245   pale, washed out, translucent
        bush   H 48-100  S 140-250  V  80-190   deep, solid, opaque

    Gas is an overlay drawn on top of the map, so it desaturates and brightens
    whatever it covers; foliage is painted colour and stays saturated. Any rule
    that keys on hue alone will confuse the two on some map. This one does not.

Usage:
    python tools/calibrate_map.py shot.png                 # analyse + print profile
    python tools/calibrate_map.py shot.png --montage out.png
    python tools/calibrate_map.py shots/*.JPEG --gas-only  # pooled gas stats
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from perception.getTerrain import play_rect  # noqa: E402

# Classification rules for assigning clusters to terrain classes. Derived from
# the measurements in the module docstring; see there for why S/V and not H.
GAS_RULE = dict(h=(40, 75), s=(60, 150), v=(180, 255))
BUSH_S_MIN = 140          # below this a green cluster is gas, not foliage
BUSH_V_MAX = 195


def cluster(image_bgr, k=6, work_w=320, hud_trim=(0.10, 0.88)):
    """k-means the play area in HSV. Returns (small_bgr, labels, centres, counts)."""
    x, y, w, h = play_rect(image_bgr)
    roi = image_bgr[y:y + h, x:x + w]
    # Trim the left/right HUD columns so the joystick and buttons do not become
    # their own "terrain" clusters.
    roi = roi[:, int(hud_trim[0] * roi.shape[1]):int(hud_trim[1] * roi.shape[1])]
    small = cv2.resize(roi, (work_w, max(1, int(roi.shape[0] * work_w / roi.shape[1]))),
                       interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    z = hsv.reshape(-1, 3).astype(np.float32)
    _, labels, centres = cv2.kmeans(
        z, k, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5),
        6, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()
    return small, labels, centres, np.bincount(labels, minlength=k)


def _in(c, rule):
    return (rule["h"][0] <= c[0] <= rule["h"][1]
            and rule["s"][0] <= c[1] <= rule["s"][1]
            and rule["v"][0] <= c[2] <= rule["v"][1])


def assign(centres, counts):
    """Label each cluster gas / bush / floor / wall / other.

    floor is the biggest non-gas, non-bush cluster; wall is the next biggest
    with a HIGHER value (lit block tops read brighter than the ground they sit
    on, on every map measured).
    """
    roles = {}
    for i, c in enumerate(centres):
        if _in(c, GAS_RULE):
            roles[i] = "gas"
        elif 40 <= c[0] <= 105 and c[1] >= BUSH_S_MIN and c[2] <= BUSH_V_MAX:
            roles[i] = "bush"
    rest = [i for i in range(len(centres)) if i not in roles]
    rest.sort(key=lambda i: -counts[i])
    if rest:
        floor = rest[0]
        roles[floor] = "floor"
        # Walls must be SUBSTANTIALLY brighter than the floor, not merely
        # brighter. Lit block tops measure 70-140 value above the ground on
        # every map here, while shadow and edge clusters sit only a few points
        # above it — a bare `>` picks those and produces a useless wall class.
        brighter = [i for i in rest[1:] if centres[i][2] > centres[floor][2] + 25]
        if brighter:
            roles[max(brighter, key=lambda i: counts[i])] = "wall"
    for i in range(len(centres)):
        roles.setdefault(i, "other")
    return roles


def bounds_for(hsv_px, pad=(6, 45, 55)):
    """Bounds around a cluster's CORE, not its full extent.

    A cluster owns every pixel nearest to its centre, including antialiased
    edges that blend into neighbouring classes. Taking the 3rd/97th percentile
    of all member pixels therefore stretches each class over its neighbours and
    the resulting bands overlap. The interquartile core plus a fixed pad tracks
    the actual surface colour and keeps the classes disjoint.
    """
    lo = np.clip(np.percentile(hsv_px, 25, axis=0) - pad, 0, 255).astype(int)
    hi = np.clip(np.percentile(hsv_px, 75, axis=0) + pad, 0, 255).astype(int)
    return lo, hi


def analyse(path, k=6, montage=None, quiet=False):
    img = cv2.imread(str(path))
    if img is None:
        print(f"  !! could not read {path}")
        return None
    small, labels, centres, counts = cluster(img, k=k)
    roles = assign(centres, counts)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).reshape(-1, 3)

    if not quiet:
        print(f"\n{pathlib.Path(path).name}  ({img.shape[1]}x{img.shape[0]})")
        for i in np.argsort(-counts):
            print(f"   {roles[i]:6s} HSV {centres[i].astype(int)}  "
                  f"{100 * counts[i] / counts.sum():5.1f}%")

    out = {}
    for role in ("floor", "wall", "bush", "gas"):
        idx = [i for i, r in roles.items() if r == role]
        if not idx:
            continue
        px = hsv[np.isin(labels, idx)]
        out[role] = bounds_for(px)

    if montage:
        tiles = []
        for i in np.argsort(-counts):
            m = (labels == i).reshape(small.shape[:2])
            vis = small.copy()
            vis[~m] = (20, 20, 20)
            cv2.putText(vis, f"{roles[i]} {centres[i].astype(int)}", (4, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
            tiles.append(vis)
        rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles) - 2, 3)]
        cv2.imwrite(str(montage), np.vstack(rows))
        print(f"   montage -> {montage}")
    return out


def print_profile(name, b):
    if not all(k in b for k in ("floor", "wall", "bush")):
        print(f"# {name}: incomplete (missing "
              f"{[k for k in ('floor','wall','bush') if k not in b]}) — "
              f"label this one by hand")
        return
    print(f"\n{name.upper()} = TerrainProfile(")
    print(f"    name={name!r},")
    for cls in ("floor", "wall", "bush"):
        lo, hi = b[cls]
        print(f"    {cls}=(np.array({list(lo)}), np.array({list(hi)})),")
    print(f")\n# add {name.upper()} to PROFILES in perception/getTerrain.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("-k", type=int, default=6, help="number of colour clusters")
    ap.add_argument("--montage", help="write a cluster montage (single image only)")
    ap.add_argument("--name", default=None, help="profile name to print")
    ap.add_argument("--gas-only", action="store_true",
                    help="just pool gas stats across all the images given")
    args = ap.parse_args()

    gas_all = []
    for p in args.images:
        b = analyse(p, k=args.k, montage=args.montage if len(args.images) == 1 else None,
                    quiet=args.gas_only)
        if not b:
            continue
        if "gas" in b:
            gas_all.append(b["gas"])
        if not args.gas_only:
            print_profile(args.name or pathlib.Path(p).stem[:8], b)

    if gas_all:
        lo = np.min([g[0] for g in gas_all], axis=0)
        hi = np.max([g[1] for g in gas_all], axis=0)
        print(f"\n=== GAS pooled over {len(gas_all)} image(s) ===")
        print(f"GAS_LOWER = np.array({list(lo)})")
        print(f"GAS_UPPER = np.array({list(hi)})")


if __name__ == "__main__":
    main()
