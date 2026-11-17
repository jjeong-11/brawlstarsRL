"""
scripts/bench_realtime.py
=========================

Is the loop fast enough for real-time play? Runs the full live stack
(rl/pipeline.LiveAgent: perception -> world model -> brain -> combat ->
route) at a target tick rate, paced in real time, and reports where the time
goes.

    python scripts/bench_realtime.py                        # panning footage built from a fixture
    python scripts/bench_realtime.py --video match.mp4      # a recording
    python scripts/bench_realtime.py --frames media/recordings/<run>   # web UI recorder output
    python scripts/bench_realtime.py --hz 20 --seconds 30
    python scripts/bench_realtime.py --sim 20               # decision layer only, in simulated matches

WHAT TO LOOK AT
---------------
  * `total` p95 against the tick budget (1000 / hz ms). Over budget on p95
    means decisions queue behind each other and act on stale frames.
  * `achieved Hz`: what the loop sustained. The simulator puts the cost of
    stale information at up to half a place (docs/PLAN_pro_imitation.md).
  * Capture and touch are NOT in here; they need the phone:
    `python -m rl.scrcpy_video --seconds 10` and `scripts/measure_latency.py`.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
import time

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402


def panning_frames(n: int, fps: float = 15.0):
    """Synthetic footage with real pixels: the fixture's play area tiled into
    a world the camera walks around, HUD and player sprite pinned on top."""
    import cv2
    from perception.getTerrain import _hud_mask
    img = cv2.imread(str(_ROOT / "media" / "fixtures" / "showdown.png"))
    if img is None:
        raise SystemExit("media/fixtures/showdown.png is missing")
    H, W = img.shape[:2]
    x0, pw = 252, 1920
    play = img[:, x0:x0 + pw]
    row = np.hstack([play, cv2.flip(play, 1), play])
    world = np.vstack([row, cv2.flip(row, 0), row])
    hud = _hud_mask(H, W) > 0
    sprite = (slice(324 - 110, 324 + 70), slice(1212 - 80, 1212 + 80))
    v = 2.4 * (pw / 17.0) / fps
    cam = np.array([1900.0, 1100.0])
    out = []
    for i in range(n):
        a = 2 * math.pi * (i / (fps * 8.0))            # a slow loop, one lap per 8 s
        if (i // int(fps * 2)) % 4 != 3:               # stand still a quarter of the time
            cam += (math.cos(a) * v, math.sin(a) * v)
        x, y = int(round(cam[0])), int(round(cam[1]))
        f = img.copy()
        f[:, x0:x0 + pw] = world[y:y + H, x:x + pw]
        f[hud] = img[hud]
        f[sprite] = img[sprite]
        out.append(f)
    return out


def load_frames(args):
    import cv2
    if args.video:
        cap = cv2.VideoCapture(args.video)
        out = []
        while len(out) < args.max_frames:
            ok, f = cap.read()
            if not ok:
                break
            out.append(f)
        return out
    if args.frames:
        paths = sorted(pathlib.Path(args.frames).glob("*.jpg")) + sorted(pathlib.Path(args.frames).glob("*.png"))
        return [cv2.imread(str(p)) for p in paths[:args.max_frames]]
    return panning_frames(min(args.max_frames, 240))


def report(lat: dict, hz: float, achieved: float, label: str) -> None:
    budget = 1000.0 / hz
    print(f"\n{label}: target {hz:g} Hz (budget {budget:.0f} ms/tick), achieved {achieved:.1f} Hz")
    print(f"  {'stage':12s} {'p50':>7s} {'p95':>7s} {'max':>7s}   ms")
    for k in sorted(lat, key=lambda k: -lat[k]["p95"]):
        v = lat[k]
        flag = "  <-- over budget" if k == "total" and v["p95"] > budget else ""
        print(f"  {k:12s} {v['p50']:7.1f} {v['p95']:7.1f} {v['max']:7.1f}{flag}")


def bench_live(args) -> dict:
    from rl.pipeline import LiveAgent
    frames = load_frames(args)
    if not frames:
        raise SystemExit("no frames")
    agent = LiveAgent(args.brawler, tick_seconds=1.0 / args.hz, rewards=not args.no_rewards,
                      gas_every_s=args.gas_every, terrain_every_s=args.terrain_every)
    period = 1.0 / args.hz
    n = int(args.seconds * args.hz)
    warm = int(min(n // 4, 3 * args.hz))                # perception needs ~2 s to confirm a match
    t_start = time.perf_counter()
    in_match = 0
    for i in range(n + warm):
        t0 = time.perf_counter()
        if i == warm:
            agent._lat.clear()
            t_start = t0
            in_match = 0
        tick = agent.step(frames[i % len(frames)])
        in_match += tick.in_match
        left = period - (time.perf_counter() - t0)
        if left > 0:
            time.sleep(left)
    achieved = n / (time.perf_counter() - t_start)
    lat = agent.latency()
    report(lat, args.hz, achieved, f"live stack, {len(frames)} frames of {frames[0].shape[1]}x{frames[0].shape[0]}")
    print(f"  in-match ticks: {in_match}/{n}" + ("" if in_match else "  (perception never confirmed a match)"))
    return lat


def bench_sim(matches: int, hz: float) -> dict:
    """The decision layer (tracker + brain + combat + route + localiser) on
    simulated matches: real walls, gas and enemies, no pixels."""
    from sim.harness import AgentStack, SimView, PerceptionModel, build_match
    from sim.bots import Bot  # noqa: F401
    ts = []
    for seed in range(9100, 9100 + matches):
        arena, bots, ability, gm = build_match(seed)
        view = SimView(arena, 0, PerceptionModel(), seed=seed)
        agent = AgentStack("tactical", ability=ability, seed=seed, tick_seconds=0.1)
        me = arena.brawlers[0]
        while not arena.over() and me.alive:
            state, terrain, gas, cam = view.observe()
            t = time.perf_counter()
            move, dec = agent.act(state, terrain, gas, cam)
            ts.append(time.perf_counter() - t)
            me.move = (float(move[0]), float(move[1]))
            if dec.attack:
                arena.fire(me, dec.aim, dec.aim_frac)
            for b in bots:
                b.act(0.1)
            arena.step(0.1)
    a = np.sort(np.array(ts)) * 1000
    lat = {"decision": {"p50": float(a[a.size // 2]), "p95": float(a[int(0.95 * a.size)]),
                        "max": float(a[-1]), "n": int(a.size)}}
    report(lat, hz, 1000.0 / max(1e-9, float(a.mean())), f"decision layer in {matches} simulated matches ({a.size} ticks)")
    slow = float(np.mean(a > 1000.0 / hz))
    print(f"  ticks over the {1000.0 / hz:.0f} ms budget: {slow:.2%}")
    return lat


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--video")
    ap.add_argument("--frames", help="directory of .jpg/.png frames")
    ap.add_argument("--hz", type=float, default=10.0)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--max-frames", type=int, default=600)
    ap.add_argument("--brawler", default="shelly")
    ap.add_argument("--gas-every", type=float, default=0.4)
    ap.add_argument("--terrain-every", type=float, default=0.0)
    ap.add_argument("--no-rewards", action="store_true")
    ap.add_argument("--sim", type=int, default=0, metavar="MATCHES",
                    help="benchmark the decision layer in this many simulated matches instead")
    a = ap.parse_args(argv)
    if a.sim:
        bench_sim(a.sim, a.hz)
    else:
        bench_live(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
