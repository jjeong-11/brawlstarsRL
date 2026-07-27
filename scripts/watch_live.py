#!/usr/bin/env python3
"""
Watch live rewards from the phone — verify perception + calibration BEFORE training.

Captures frames from a connected Android phone, runs the full perception stack,
and prints per-tick readings (HP, cubes, super%, brawlers-left, gas, enemies) plus
the reward breakdown — exactly what the agent would learn from. It does NOT send
any actions by default, so it's safe to just observe.

    python scripts/watch_live.py --serial <SERIAL>
    python scripts/watch_live.py --serial <SERIAL> --save-preview      # annotated frame -> debugOutput/
    python scripts/watch_live.py --serial <SERIAL> --calibrate --controls controls.json

Use this to confirm:
  * perception reads YOUR phone's HUD correctly (nudge getSuper.SUPER_ROI etc. if not),
  * the rewards make sense as you play,
  * (with --calibrate) the executor taps land on the real buttons.
"""
import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
from perception.liveLoop import AdbScreencapSource, LivePerception  # noqa: E402
from perception.getSuper import SUPER_ROI  # noqa: E402
from rl.state import adapt_live_state  # noqa: E402
from rl.rewards import RewardCalculator, RewardConfig  # noqa: E402
from rl.kills import KillAttributor  # noqa: E402
from rl.actions import Controls  # noqa: E402

_DEBUG = pathlib.Path(__file__).resolve().parent.parent / "debugOutput"


def _draw_preview(frame, live, controls):
    """Annotated debug frame: what perception SEES + where the bot will TAP.

    Detections (what it reads):  blue=player, red=enemies, orange=boxes,
    green=ground cubes, yellow box=super ROI.
    Controls (where it taps, from controls.json): filled MOVE/ATTACK/SUPER dots.
    """
    o = frame.copy()
    H, W = o.shape[:2]

    a = live.get("anchor")
    if a:
        cv2.circle(o, (a[0], a[1]), a[2], (255, 0, 0), 3)
        cv2.putText(o, "player", (a[0] - 40, a[1] - a[2] - 10), 0, 0.8, (255, 0, 0), 2)

    for e in live.get("enemies", []):
        cx, cy = e["center"]
        r = max(int(e.get("radius", 40)), 34)
        cv2.circle(o, (cx, cy), r, (0, 0, 255), 3)
        cv2.circle(o, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(o, "enemy", (cx - 30, cy - r - 8), 0, 0.7, (0, 0, 255), 2)

    for b in live.get("boxes", []):
        c = b.get("center")
        if c:
            cv2.circle(o, (int(c[0]), int(c[1])), int(b.get("radius", 30)), (0, 165, 255), 2)
    for cu in live.get("cubes", []):
        bx, by, bw, bh = cu["bounding_box"]
        cv2.rectangle(o, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)

    # super charge ROI (what getSuper reads)
    x0, x1 = int(SUPER_ROI[0] * W), int(SUPER_ROI[1] * W)
    y0, y1 = int(SUPER_ROI[2] * H), int(SUPER_ROI[3] * H)
    cv2.rectangle(o, (x0, y0), (x1, y1), (0, 255, 255), 2)

    # control tap points (outputs, not detections)
    if controls is not None:
        for name, pt, col in [("MOVE", controls.move_center, (0, 255, 0)),
                              ("ATTACK", controls.attack_btn, (0, 140, 255)),
                              ("SUPER", controls.super_btn, (0, 255, 255))]:
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(o, (x, y), 16, col, -1)
            cv2.circle(o, (x, y), 20, (0, 0, 0), 2)
            cv2.putText(o, name, (x - 40, y + 50), 0, 0.8, col, 2)

    gs = live["game_state"]
    label = (f"{gs['state']} left:{gs['brawlers_left']} HP:{live['hp']} ammo:{live['ammo']} "
             f"super:{live['super_charge']*100:.0f}% gas:{live['in_gas']} "
             f"enemies:{len(live.get('enemies', []))}")
    cv2.putText(o, label, (20, 40), 0, 1.0, (255, 255, 255), 2)
    return o


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", default=None, help="adb device serial (if >1 device)")
    ap.add_argument("--fps", type=float, default=6.0, help="target ticks per second")
    ap.add_argument("--duration", type=float, default=None, help="stop after N seconds")
    ap.add_argument("--save-preview", action="store_true",
                    help="write an annotated frame to debugOutput/live_preview.png each second")
    ap.add_argument("--calibrate", action="store_true",
                    help="first tap each control once (verify AdbExecutor placement)")
    ap.add_argument("--controls", default=None, help="controls.json for --calibrate")
    args = ap.parse_args()

    try:
        source = AdbScreencapSource(args.serial)
    except RuntimeError as e:
        print("ERROR:", e)
        print("Connect the phone (USB debugging on) and check `adb devices`.")
        return

    # Loaded once, used both to draw tap points on the preview and to --calibrate.
    controls = Controls.from_json(args.controls) if args.controls else None

    if args.calibrate:
        from rl.actions import AdbExecutor
        ex = AdbExecutor(serial=args.serial, controls=controls)
        print("Calibrating: tapping attack, then super, then a right move-swipe. Watch the phone.")
        ex.calibrate()
        ex.close()
        time.sleep(1.0)

    perception = LivePerception()
    calc = RewardCalculator(RewardConfig())
    calc.reset()
    killer = KillAttributor()
    killer.reset()

    started = time.time()
    ticks = 0
    misses = 0
    last_preview = 0.0

    print("Watching live rewards (no actions sent). Ctrl+C to stop.")
    print("tick   time   reward   return  | state")
    try:
        while True:
            if args.duration and time.time() - started > args.duration:
                break
            t0 = time.time()

            frame = source.grab()
            if frame is None:                       # transient capture miss
                misses += 1
                if misses > 90:
                    print("too many dropped frames — is the phone connected and awake?")
                    break
                time.sleep(0.1)
                continue
            misses = 0

            live, _ = perception.tick(frame)
            kills = killer.update(live)
            state = adapt_live_state(live, tick=ticks, kills_this_tick=kills)
            result = calc.compute(state)
            ticks += 1

            gs = live["game_state"]["state"]
            bd = {k: round(v, 3) for k, v in result.breakdown.items() if abs(v) > 1e-9}
            print(f"{ticks:4d} {time.time()-started:6.1f}s {result.total:+7.3f} "
                  f"{calc.episode_return:+8.2f} | {gs} HP:{state.health} cubes:{state.cube_count} "
                  f"super:{(state.super_charge or 0)*100:.0f}% left:{state.players_left} "
                  f"gas:{state.in_gas} enemies:{len(state.enemy_positions)} {bd}", flush=True)

            if args.save_preview and time.time() - last_preview >= 1.0:
                _DEBUG.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(_DEBUG / "live_preview.png"), _draw_preview(frame, live, controls))
                last_preview = time.time()

            sleep_for = (1.0 / args.fps) - (time.time() - t0)
            if sleep_for > 0:
                time.sleep(sleep_for)
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        source.close()
        print(f"\n{ticks} ticks. reward totals: {dict(calc.episode_breakdown)}")


if __name__ == "__main__":
    main()
