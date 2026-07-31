#!/usr/bin/env python3
"""
Measure how long it takes an action to actually move the brawler.

    python scripts/measure_latency.py --serial <SERIAL>
    python scripts/measure_latency.py --serial <SERIAL> --sendevent --cycles 12

WHY THIS MATTERS MORE THAN IT SOUNDS
------------------------------------
PPO credits the reward at step t to the action at step t. If the action at step
t does not reach the screen until step t+3, every one of those credits is
attached to the wrong decision — the agent is being taught that whatever it
happened to choose three ticks ago caused the outcome. No amount of reward
shaping fixes that, because the reward function is not what is wrong.

The delay is not guessable. It is the sum of adb (or sendevent) transport, the
game's own input handling and animation ramp, the capture pipeline's own lag,
and the fact that the env acts on the frame it was shown one step earlier. On
a USB-connected phone that total is plausibly anywhere from one tick to five,
and the only way to know is to measure it on YOUR device.

HOW IT MEASURES
---------------
Drive the joystick in a square wave — hard left for N ticks, hard right for N
ticks, repeat — while measuring the world's scroll direction with the same
`rl/camera_tracker.py` phase correlation the planner uses for waypoint
latching. The commanded direction is a known square wave; the observed one is
that wave delayed and smeared. Cross-correlating the two gives the lag in
ticks, at whole-tick resolution, which is the resolution the RL loop cares
about.

A square wave is used rather than a single impulse on purpose: one step's
motion is small and noisy, while repeated reversals accumulate a signal that
stands well clear of phase-correlation noise, and the sign flips make the
correlation peak sharp instead of broad.

SAFETY
------
This drives the joystick, so run it in a real match where walking around is
harmless. It never taps attack or super.

WHAT TO DO WITH THE ANSWER
--------------------------
    0-1 ticks  fine, nothing to do.
    2-3 ticks  pass `--action-delay N` to train_rl.py. The last N actions get
               added to the observation, which is what restores the Markov
               property under a constant delay -- the policy can then tell
               which of its recent decisions is landing now.
    4+ ticks   the loop is mis-tuned rather than merely laggy. Check `hold_ms`
               against `--tick-seconds` (see Controls.tuned_for_tick), and
               consider --sendevent or --scrcpy, which hold the touch instead
               of re-swiping every tick.
"""

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from perception.liveLoop import AdbScreencapSource  # noqa: E402
from perception.getTerrain import play_rect  # noqa: E402
from rl.camera_tracker import CameraTracker  # noqa: E402


def _make_executor(args):
    from rl.actions import Controls
    controls = Controls.from_json(args.controls) if args.controls else None
    if args.sendevent:
        from rl.sendevent_backend import SendeventExecutor
        return SendeventExecutor(serial=args.serial, controls=controls,
                                 flip_short=False, flip_long=True)
    from rl.actions import AdbExecutor
    ex = AdbExecutor(serial=args.serial, controls=controls)
    if controls is None:
        ex.controls = ex.controls.tuned_for_tick(args.tick)
    return ex


def _drive(executor, direction, controls):
    """Push the joystick in `direction` without going through the planner."""
    from rl.actions import Intent, intent_to_touches
    intent = Intent("measure", direction, False, False, (0, 0, 0, 0))
    if hasattr(executor, "_execute"):
        executor._execute(intent)
    else:
        for op in intent_to_touches(intent, controls):
            pass


def _lag_by_correlation(commanded, observed, max_lag):
    """Whole-tick lag that best aligns `observed` with `commanded`."""
    c = np.asarray(commanded, np.float64)
    o = np.asarray(observed, np.float64)
    c -= c.mean()
    o -= o.mean()
    if np.allclose(o, 0):
        return None, 0.0
    scores = []
    for lag in range(0, max_lag + 1):
        a, b = c[:len(c) - lag], o[lag:]
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        scores.append(float(np.dot(a, b) / denom) if denom > 1e-9 else 0.0)
    best = int(np.argmax(scores))
    return best, scores[best], scores


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--serial", default=None)
    ap.add_argument("--controls", default=None, help="controls.json")
    ap.add_argument("--tick", type=float, default=0.1, help="seconds per tick")
    ap.add_argument("--half-period", type=int, default=6,
                    help="ticks per direction before reversing")
    ap.add_argument("--cycles", type=int, default=8, help="left+right pairs")
    ap.add_argument("--sendevent", action="store_true",
                    help="use the raw-multitouch backend instead of adb swipes")
    ap.add_argument("--max-lag", type=int, default=10, help="ticks to search")
    args = ap.parse_args()

    try:
        source = AdbScreencapSource(args.serial)
    except RuntimeError as e:
        raise SystemExit(f"ERROR: {e}\nConnect the phone and check `adb devices`.")
    executor = _make_executor(args)
    camera = CameraTracker()

    total = args.cycles * 2 * args.half_period
    commanded, observed = [], []
    rect = None
    misses = 0

    print(f"Driving a {args.half_period}-tick square wave for {total} ticks "
          f"({total * args.tick:.1f}s). Stand somewhere open.\n")
    try:
        for i in range(total):
            t0 = time.perf_counter()
            phase = (i // args.half_period) % 2
            direction = (-1.0, 0.0) if phase == 0 else (1.0, 0.0)
            _drive(executor, direction, executor.controls)

            frame = source.grab()
            if frame is None:
                misses += 1
                continue
            if rect is None:
                rect = play_rect(frame)
            dx, dy, ok = camera.update(frame, rect)
            # The world scrolls OPPOSITE to the player's motion, so negate to
            # recover the direction he actually walked.
            commanded.append(direction[0])
            observed.append(-dx if ok else 0.0)

            if i % args.half_period == 0:
                print(f"  tick {i:3d}  commanded {direction[0]:+.0f}  "
                      f"observed {observed[-1]:+7.1f}px")
            remaining = args.tick - (time.perf_counter() - t0)
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        try:
            executor.close()
        except Exception:
            pass
        source.close()

    if len(observed) < args.half_period * 4:
        raise SystemExit("not enough samples — did the phone stay awake?")

    lag, score, scores = _lag_by_correlation(commanded, observed, args.max_lag)
    amp = float(np.abs(np.asarray(observed)).mean())

    print(f"\n{len(observed)} usable ticks, {misses} dropped frames")
    print(f"mean |scroll| per tick: {amp:.1f}px "
          f"({'the brawler is moving' if amp > 2 else 'BARELY MOVING — check calibration'})")
    print("\nlag  correlation")
    for i, s in enumerate(scores):
        bar = "#" * max(0, int(40 * max(0.0, s)))
        print(f"{i:3d}  {s:+.3f} {bar}{'   <-- best' if i == lag else ''}")

    print(f"\nACTION LATENCY: {lag} tick(s) = {lag * args.tick * 1000:.0f} ms "
          f"(correlation {score:.2f})")
    if score < 0.3:
        print("  ...but the correlation is weak, so treat that as unmeasured.\n"
              "  Usually means the joystick coordinates are off (calibrate) or\n"
              "  the brawler was against a wall the whole time.")
    elif lag <= 1:
        print("  Nothing to do — reward is landing on roughly the right action.")
    elif lag <= 3:
        print(f"  Add --action-delay {lag} to scripts/train_rl.py so the policy\n"
              f"  can see which of its recent actions is landing now.")
    else:
        print(f"  That is a lot. Check Controls.hold_ms against --tick-seconds,\n"
              f"  and try --sendevent or --scrcpy (they hold the touch instead\n"
              f"  of re-swiping every tick). Then re-measure.")


if __name__ == "__main__":
    main()
