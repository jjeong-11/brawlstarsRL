#!/usr/bin/env python3
"""
Watch live rewards from the phone — verify perception + calibration BEFORE training.

Captures frames from a connected Android phone, runs the full perception stack,
and prints per-tick readings (HP, cubes, super%, brawlers-left, gas, enemies) plus
the reward breakdown — exactly what the agent would learn from. It does NOT send
any actions, so it's safe to just observe.

    python scripts/watch_live.py --serial <SERIAL>
    python scripts/watch_live.py --serial <SERIAL> --save-preview      # annotated frame -> debugOutput/
    python scripts/watch_live.py --serial <SERIAL> --trace 2           # decision traces (see below)
    python scripts/watch_live.py --serial <SERIAL> --calibrate --controls controls.json

Use this to confirm:
  * perception reads YOUR phone's HUD correctly (nudge getSuper.SUPER_ROI etc. if not),
  * the rewards make sense as you play,
  * (with --calibrate) the executor taps land on the real buttons.

WATCHING THE AGENT THINK (--trace)
----------------------------------
With --trace the saved policy is loaded and asked for an action every tick, and
the planner is run on it — but the executor is never invoked, so NOTHING is sent
to the phone. You play; the agent decides alongside you and its intended route is
written to debugOutput/trace/.

That separation is the whole value: on the phone a bad route and bad perception
look identical, because all you see is a brawler walking into a wall. Here you can
watch the destination it picked and the path it planned while YOU control where
the brawler actually goes — so you can deliberately stand next to a wall, or in
the gas, and see what it would have done about it.
"""
import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
from perception.liveLoop import AdbScreencapSource  # noqa: E402
from perception.getSuper import SUPER_ROI  # noqa: E402
from rl.actions import Controls  # noqa: E402

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_DEBUG = _ROOT / "debugOutput"


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
    ap.add_argument("--backend", choices=("adb", "scrcpy"), default="adb",
                    help="touch backend --calibrate uses (scrcpy needs scrcpy installed)")
    ap.add_argument("--trace", type=float, default=0.0, metavar="SECONDS",
                    help="run the planner alongside you and write an annotated "
                         "decision frame every SECONDS to debugOutput/trace/. "
                         "Nothing is sent to the phone. Try 2.")
    ap.add_argument("--model", default=None,
                    help="policy to ask for actions under --trace. Defaults to "
                         "./brawlstars_move.zip; without one, headings are "
                         "sampled at random, which still exercises the planner.")
    ap.add_argument("--brain", choices=("auto", "tactical", "policy"), default="auto",
                    help="who chooses movement under --trace: rl/tactics.py "
                         "(tactical), the policy, or auto = tactical unless "
                         "--model is given")
    ap.add_argument("--brawler", default=None,
                    help="roster id (e.g. piper) for per-brawler range and aim")
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
        if args.backend == "scrcpy":
            from rl.scrcpy_control import ScrcpyExecutor
            ex = ScrcpyExecutor(serial=args.serial, controls=controls)
        else:
            from rl.actions import AdbExecutor
            ex = AdbExecutor(serial=args.serial, controls=controls)
        print("Calibrating: tapping attack, then super, then a right move-swipe. Watch the phone.")
        ex.calibrate()
        ex.close()
        time.sleep(1.0)

    # The same frame -> decision stack the web UI runs (rl/pipeline.py); here
    # its decisions are only printed and traced, never sent.
    from rl.pipeline import LiveAgent
    tracer = _make_tracer(args) if args.trace else None
    use_brain = args.brain == "tactical" or (args.brain == "auto" and not args.model)
    policy = None if use_brain else _load_policy(args.model)
    factory = None
    if policy is not None:
        from rl.checkpoints import PolicyStepper
        factory = lambda: PolicyStepper(policy, deterministic=False)   # noqa: E731
    agent = LiveAgent(args.brawler, tick_seconds=1.0 / max(0.1, args.fps),
                      use_brain=use_brain, policy_stepper_factory=factory,
                      policy_source="base" if policy is not None else "none")
    calc = agent.calc
    print(f"movement: {'tactical brain' if use_brain else 'policy'}"
          f"{f' ({args.brawler})' if args.brawler else ''}")

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

            tk = agent.step(frame)
            live, state, result = tk.live, tk.state, tk.reward
            ticks += 1

            gs = live["game_state"]["state"]
            bd = {k: round(v, 3) for k, v in result.breakdown.items() if abs(v) > 1e-9}
            print(f"{ticks:4d} {time.time()-started:6.1f}s {result.total:+7.3f} "
                  f"{calc.episode_return:+8.2f} | {gs} HP:{state.health} cubes:{state.cube_count} "
                  f"super:{(state.super_charge or 0)*100:.0f}% left:{state.players_left} "
                  f"gas:{state.in_gas} enemies:{len(state.enemy_positions)} {bd}", flush=True)

            if tracer is not None and tracer.due():
                tracer.capture(frame, agent.planner, state, intent=tk.intent,
                               reward=result, live=live, tick=ticks, force=True)
                print(f"     why: {agent.explain() or tk.intent.move_label}", flush=True)

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
        if tracer is not None:
            tracer.close()
            print(f"traces -> {tracer.dir}")
        print(f"\n{ticks} ticks. reward totals: {dict(calc.episode_breakdown)}")


# --- decision tracing -------------------------------------------------------- #
def _make_tracer(args):
    from rl.debug_trace import DecisionTracer, TraceConfig
    t = DecisionTracer(TraceConfig(every_seconds=args.trace))
    print(f"tracing decisions every {args.trace}s -> {t.dir} (no input is sent)")
    return t


def _load_policy(path):
    """The saved PPO policy, or None to fall back to random headings.

    Random is not a useless fallback: the planner is what turns a heading into
    a route, so random headings still exercise every part of the pathfinding
    (relocation, A*, clearance, gas escape) against real frames. It just does
    not tell you whether the POLICY is choosing sensibly.
    """
    p = pathlib.Path(path) if path else (_ROOT / "brawlstars_move.zip")
    if not p.exists():
        print(f"no policy at {p} — tracing with random headings")
        return None
    try:
        from rl.checkpoints import load_policy
        model, algo = load_policy(p, device="cpu")
        print(f"loaded policy {p.name} ({algo})")
        return model
    except Exception as e:
        print(f"could not load {p.name} ({e}) — tracing with random headings")
        return None


if __name__ == "__main__":
    main()
