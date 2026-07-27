#!/usr/bin/env python3
"""
scripts/check_perception.py
===========================

Pre-flight health check for the perception stack. Run this BEFORE training.

WHY THIS EXISTS
    Every expensive mistake in this project so far has been a silent perception
    failure that looked like a reinforcement-learning problem:

      * terrain constants that matched one map, so 86% of another map read as
        solid wall and the agent ground into borders
      * a gas band that fired on bushes
      * an anchor that locked onto crates, so HP/ammo/cube windows all searched
        the wrong part of the screen
      * ammo reading 0 both for "empty clip" and "could not see the bar"

    None of these show up as an error. They show up as a flat training curve
    two days later. This script puts numbers on each stage so a bad one is
    obvious in about a minute.

Usage:
    python scripts/check_perception.py                          # bundled recording
    python scripts/check_perception.py --video path/to/clip.mp4
    python scripts/check_perception.py --image screenshot.png   # single frame
    python scripts/check_perception.py --serial <SERIAL>        # live from phone
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from perception.getAnchor import find_player_position_ex          # noqa: E402
from perception.getHealth import find_health_info                 # noqa: E402
from perception.getAmmo import find_ammo_info                     # noqa: E402
from perception.getSuper import find_super_info                   # noqa: E402
from perception.getGas import gas_info                            # noqa: E402
from perception.getGameState import get_game_state                # noqa: E402
from perception.getTerrain import find_terrain, ProfileSelector, play_rect  # noqa: E402

_ROOT = pathlib.Path(__file__).resolve().parent.parent

# (label, healthy floor, what a low number means)
THRESHOLDS = [
    ("terrain profile matched", 0.80,
     "no profile fits this map — add one with tools/calibrate_map.py, or A* "
     "plans over an imaginary maze"),
    ("anchor verified", 0.30,
     "the player is rarely located with confidence; HP/ammo/cubes all search "
     "relative to it, so everything downstream degrades"),
    ("HP read", 0.60,
     "the HP digit line is not parsing on verified anchors"),
    ("ammo read", 0.40,
     "known weak spot: find_ammo_info detects ~48% given a good HP read. The "
     "weapon gate is ammo_known-guarded, so this is safe, just not useful"),
]


def analyse(frames, show_each=False):
    sel = ProfileSelector()
    n = terr = anch = hp = ammo = 0
    blocked, coverage, gasfrac = [], [], []
    profiles, ammo_vals = {}, {}
    supers = []

    for f in frames:
        if get_game_state(f)["state"] != "in_match":
            continue
        n += 1

        t = find_terrain(f, selector=sel)
        if t is not None:
            terr += 1
            blocked.append(float(t["occupancy"].mean()))
            coverage.append(float(t["coverage"]))
            profiles[t["profile"].name] = profiles.get(t["profile"].name, 0) + 1

        x, y, r, verified = find_player_position_ex(f)
        gasfrac.append(gas_info(f, player_pos=(x, y, r))["frac"])
        supers.append(find_super_info(f)["charge"])

        if not verified or r == 0:
            continue
        anch += 1
        h = find_health_info(f, (x, y, r))
        if h["current_health"] is None:
            continue
        hp += 1
        a = find_ammo_info(f, (x, y, r), h)
        if a["detected"]:
            ammo += 1
            ammo_vals[a["ammo_count"]] = ammo_vals.get(a["ammo_count"], 0) + 1
        if show_each:
            print(f"    anchor=({x},{y},{r}) hp={h['current_health']} "
                  f"ammo={a['ammo_count'] if a['detected'] else '?'}")

    if n == 0:
        print("No in-match frames found. Is this footage from an actual match?")
        return 1

    rates = {
        "terrain profile matched": terr / n,
        "anchor verified": anch / n,
        "HP read": hp / max(anch, 1),
        "ammo read": ammo / max(hp, 1),
    }

    print(f"\n{n} in-match frames\n")
    print(f"{'stage':26s} {'rate':>7s}   status")
    print("-" * 74)
    bad = 0
    for label, floor, hint in THRESHOLDS:
        v = rates[label]
        ok = v >= floor
        bad += not ok
        print(f"{label:26s} {100 * v:6.0f}%   {'OK' if ok else 'LOW  -> ' + hint}")

    print()
    if profiles:
        print(f"  terrain profile   : {profiles}")
        print(f"  coverage          : median {100 * np.median(coverage):.0f}%   "
              f"(healthy 45-75%)")
        print(f"  blocked cells     : median {100 * np.median(blocked):.0f}%   "
              f"(healthy 30-50%)")
    if ammo_vals:
        print(f"  ammo values seen  : {dict(sorted(ammo_vals.items()))}")
    if gasfrac:
        print(f"  gas fraction      : median {np.median(gasfrac):.3f}  "
              f"max {max(gasfrac):.3f}   (~0 when no gas is on screen)")
    if supers:
        print(f"  super charge      : max {max(supers):.2f}   "
              f"(stays 0.00 if you never dealt damage in this clip)")

    print()
    if bad:
        print(f"{bad} stage(s) below the healthy floor. Fix those before "
              f"training — a bad stage looks exactly like a hard RL problem.")
    else:
        print("Perception looks healthy. Safe to train.")
    return 1 if bad else 0


def load_frames(args):
    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            raise SystemExit(f"could not read {args.image}")
        return [img]

    if args.serial is not None:
        from perception.liveLoop import AdbScreencapSource
        src = AdbScreencapSource(args.serial or None)
        out = []
        print(f"capturing {args.frames} frames from the phone...")
        while len(out) < args.frames:
            f = src.grab()
            if f is not None:
                out.append(f)
            time.sleep(0.3)
        src.close()
        return out

    path = args.video or str(_ROOT / "media" / "testvideos" / "test_game2.mp4")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"could not open {path}")
    print(f"sampling {args.frames} frames from {path}")
    out, i = [], 0
    while len(out) < args.frames:
        if not cap.grab():
            break
        if i % args.step == 0:
            ok, f = cap.retrieve()
            if ok:
                out.append(f)
        i += 1
    cap.release()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", help="recording to sample")
    ap.add_argument("--image", help="single screenshot instead")
    ap.add_argument("--serial", nargs="?", const="", help="capture live over adb")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--step", type=int, default=60, help="sample every Nth frame")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    frames = load_frames(args)
    return analyse(frames, show_each=args.verbose)


if __name__ == "__main__":
    raise SystemExit(main())
