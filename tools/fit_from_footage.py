"""
tools/fit_from_footage.py
=========================

Phase 2 of docs/PLAN_pro_imitation.md as one command: gameplay recordings
in, (a) how well each decision-layer setting explains them, (b) fitted
settings, (c) simulator calibration numbers out.

    # 1) perceive once (slow: full perception stack + OCR), cached per video
    python tools/fit_from_footage.py perceive doin_*.mp4 --cache footage_cache/ --brawler shelly
    # 2) score the defaults and a few named variants against the footage
    python tools/fit_from_footage.py score --cache footage_cache/
    # 3) fit (random + local search, disagreement-weighted, as validated in sim/imitate.py)
    python tools/fit_from_footage.py fit --cache footage_cache/ --out doin_fit.json
    # 4) calibration stats for sim/ (speed, gas timing, elimination curve)
    python tools/fit_from_footage.py calibrate --cache footage_cache/

HOW
---
`perceive` runs, per frame, exactly what the web UI runs live:
LivePerception -> GameState, terrain, gas grid, camera delta -- plus the
movement label from tools/label_footage.py (player world motion = sprite
displacement - camera shift). Only in-match frames are kept, split into
matches at each new-match edge, and pickled.

`score` / `fit` replay those frames into fresh trackers and planners and run
candidate TacticalBrains in lockstep (teacher forcing), exactly as
sim/imitate.py does on simulated demonstrations, and compare each brain's
chosen direction with where the player actually went over the next second
(sim.imitate.agreement). In the simulator this recovered a hidden
demonstrator's play to within 0.02 mean place.

A human is not a TacticalBrain, so agreement here will be lower than in the
simulator; what matters is which settings explain the player BEST, and the
per-mode breakdown of where even the best settings disagree.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import pathlib
import pickle
import random
import sys
from typing import List, Optional

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

HORIZON_S = 1.0
STILL_TILES = 0.5
TILES_ACROSS = 17.0


# --------------------------------------------------------------------------- #
# 1) perceive
# --------------------------------------------------------------------------- #
def perceive_video(path: str, step: int = 2, max_frames: int = 0, log=print) -> dict:
    """{'video', 'fps', 'step', 'matches': [[frame record, ...], ...]}."""
    import cv2
    from perception.getGas import gas_info, player_anchor
    from perception.getTerrain import GRID_H, GRID_W, ProfileSelector, find_terrain, play_rect
    from perception.liveLoop import LivePerception
    from rl.camera_tracker import CameraTracker
    from rl.state import adapt_live_state
    from tools.label_footage import FootageLabeler, LabelConfig

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dt = step / fps
    # Synchronous OCR and the VIDEO clock: offline frames arrive far faster
    # than real time, and perception schedules its stages by time.
    lp, sel, cam = LivePerception(async_game_state=False), ProfileSelector(), CameraTracker()
    cur = {"state": None}
    lab = FootageLabeler(LabelConfig(fps=fps / step, smooth_frames=1),
                         perceive=lambda f: {"player_pos": cur["state"].player_pos,
                                             "ammo_known": cur["state"].ammo_known,
                                             "ammo_count": cur["state"].ammo_count,
                                             "super_charge": cur["state"].super_charge})
    matches: List[list] = []
    in_match_prev = False
    idx = kept = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step:
            idx += 1
            continue
        live, _ = lp.tick(frame, now=idx / fps)
        state = adapt_live_state(live, tick=idx, kills_this_tick=0,
                                 frame_size=(frame.shape[1], frame.shape[0]))
        screen = (live.get("game_state") or {}).get("state", "unknown")
        in_match = screen == "in_match"
        if in_match and not in_match_prev:
            matches.append([])
            cam.reset()
            lab.reset()
        in_match_prev = in_match if screen != "unknown" else in_match_prev
        if not in_match or not matches:
            idx += 1
            continue
        cur["state"] = state
        terrain = None
        if state.player_pos is not None:
            try:
                terrain = find_terrain(frame, player_pos=state.player_pos, selector=sel)
            except Exception:
                terrain = None
        rect = terrain["rect"] if terrain else play_rect(frame)
        try:
            gas = gas_info(frame, player_pos=player_anchor(state), grid_rect=rect,
                           grid_size=(GRID_W, GRID_H))["grid"]
        except Exception:
            gas = None
        dx, dy, okc = cam.update(frame, rect)
        label = lab.step(frame, rect=rect, dt=dt)
        matches[-1].append({
            "t": idx / fps, "state": state, "terrain": terrain, "gas": gas,
            "rect": tuple(rect), "cam": (dx, dy) if okc else (0.0, 0.0),
            "motion": label["motion"], "moving": label["moving"],
            "speed_tps": label["speed_tps"], "attack": label["attack"], "super": label["super"],
            "players_left": state.players_left, "frame_size": (frame.shape[1], frame.shape[0]),
        })
        kept += 1
        idx += 1
        if max_frames and kept >= max_frames:
            break
    log(f"{path}: {idx} frames read, {kept} in-match frames in {len(matches)} match(es)")
    return {"video": str(path), "fps": fps, "step": step, "matches": [m for m in matches if m]}


def labels_for(match: list) -> list:
    """Direction the player went over the next HORIZON_S (unit vector), None
    (stayed), or "end" -- from the per-frame world motion."""
    out = []
    n = len(match)
    for i, rec in enumerate(match):
        t_end = rec["t"] + HORIZON_S
        sx = sy = 0.0
        j = i + 1
        while j < n and match[j]["t"] <= t_end:
            dtj = match[j]["t"] - match[j - 1]["t"]
            sx += match[j]["motion"][0] * dtj
            sy += match[j]["motion"][1] * dtj
            j += 1
        if j >= n:
            out.append("end")
            continue
        tile_px = rec["rect"][2] / TILES_ACROSS
        d = math.hypot(sx, sy)
        out.append(None if d < STILL_TILES * tile_px else (sx / d, sy / d))
    return out


# --------------------------------------------------------------------------- #
# 2) replay candidate brains in lockstep
# --------------------------------------------------------------------------- #
def replay(cache: dict, cands: List[dict], brawler: str = "shelly"):
    """Per-candidate predictions + labels + the reference brain's modes."""
    from brawlers import default_abilities
    from rl.actions import decode_goal
    from rl.combat import CombatConfig, CombatPolicy
    from rl.enemy_tracker import EnemyTracker
    from rl.path_planner import WaypointPlanner
    from rl.tactics import TacticalBrain, TacticsConfig

    ability = default_abilities().get(brawler)
    tick_s = cache["step"] / cache["fps"]
    preds = [[] for _ in cands]
    labels, modes = [], []
    for match in cache["matches"]:
        tracker, planner = EnemyTracker(), WaypointPlanner(localize=True)
        combat = CombatPolicy(CombatConfig(tick_seconds=tick_s), ability=ability)
        brains = [TacticalBrain(dataclasses.replace(TacticsConfig(tick_seconds=tick_s), **c),
                                ability=ability, combat=combat) for c in cands]
        labels += labels_for(match)
        for rec in match:
            st = rec["state"]
            tracker.update(st, rec["cam"])
            goals = [b.decide(st, planner, tracker) for b in brains]
            for k, g in enumerate(goals):
                if g.point is None or st.player_pos is None:
                    preds[k].append(None)
                    continue
                dx, dy = g.point[0] - st.player_pos[0], g.point[1] - st.player_pos[1]
                n = math.hypot(dx, dy)
                preds[k].append(None if n < 1e-6 else (dx / n, dy / n))
            modes.append(goals[0].mode)
            # Candidate 0 drives the planner, which is what keeps the world
            # map (walls, gas) registered and fused frame to frame.
            dec = combat.decide(st, frame_size=rec["frame_size"], world=planner.world,
                                tracker=tracker, hint=goals[0].hint)
            decode_goal(goals[0], state=st, frame_size=rec["frame_size"], planner=planner,
                        terrain=rec["terrain"], camera_delta=rec["cam"],
                        gas_grid=rec["gas"], combat=dec)
    return preds, labels, modes


def load_caches(cache_dir: str) -> List[dict]:
    out = []
    for p in sorted(pathlib.Path(cache_dir).glob("*.pkl")):
        with open(p, "rb") as f:
            out.append(pickle.load(f))
    if not out:
        raise SystemExit(f"no *.pkl caches in {cache_dir}: run `perceive` first")
    return out


def score_all(caches, cands, brawler, weighted):
    from sim.imitate import agreement
    tot_s = [0.0] * len(cands)
    tot_w = [0.0] * len(cands)
    n_all = w_all = 0.0
    by_mode = {}
    for c in caches:
        preds, labels, modes = replay(c, cands, brawler)
        s, n, ws, wt = agreement(preds, labels)
        for k in range(len(cands)):
            tot_s[k] += s[k]
            tot_w[k] += ws[k]
        n_all += n
        w_all += wt
        # Where does candidate 0 (the reference) disagree, by its own mode?
        for i, lab in enumerate(labels):
            if lab == "end":
                continue
            p = preds[0][i]
            a = 1.0 if (lab is None and p is None) else (
                max(0.0, lab[0] * p[0] + lab[1] * p[1]) if (lab is not None and p is not None) else 0.0)
            m = by_mode.setdefault(modes[i], [0.0, 0])
            m[0] += a
            m[1] += 1
    plain = [t / max(1, n_all) for t in tot_s]
    weig = [t / max(1e-9, w_all) for t in tot_w]
    return (weig if weighted else plain), plain, {k: (v[0] / v[1], v[1]) for k, v in by_mode.items()}, int(n_all)


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("perceive")
    p.add_argument("videos", nargs="+")
    p.add_argument("--cache", required=True)
    p.add_argument("--step", type=int, default=2, help="use every Nth frame")
    p.add_argument("--max-frames", type=int, default=0)
    for name in ("score", "fit", "calibrate"):
        q = sub.add_parser(name)
        q.add_argument("--cache", required=True)
        q.add_argument("--brawler", default="shelly")
        q.add_argument("--out", default=None)
        if name == "fit":
            q.add_argument("--rounds", type=int, default=5)
            q.add_argument("--per-round", type=int, default=24)
    a = ap.parse_args(argv)

    if a.cmd == "perceive":
        d = pathlib.Path(a.cache)
        d.mkdir(parents=True, exist_ok=True)
        for v in a.videos:
            c = perceive_video(v, step=a.step, max_frames=a.max_frames)
            with open(d / (pathlib.Path(v).stem + ".pkl"), "wb") as f:
                pickle.dump(c, f)
        return 0

    caches = load_caches(a.cache)
    from rl.tactics import TacticsConfig
    from sim.imitate import SPACE, perturb, sample
    d = TacticsConfig()
    default = {k: getattr(d, k) for k in SPACE}

    if a.cmd == "calibrate":
        speeds = sorted(r["speed_tps"] for c in caches for m in c["matches"] for r in m if r["moving"])
        out = {"moving_speed_tps_median": speeds[len(speeds) // 2] if speeds else None,
               "moving_frac": (sum(1 for c in caches for m in c["matches"] for r in m if r["moving"])
                               / max(1, sum(len(m) for c in caches for m in c["matches"]))),
               "matches": []}
        for c in caches:
            for m in c["matches"]:
                t0 = m[0]["t"]
                gas_t = next((r["t"] - t0 for r in m if r["gas"] is not None
                              and float(r["gas"].max()) > 0.3), None)
                curve = sorted({(round(r["t"] - t0), r["players_left"]) for r in m
                                if r["players_left"] is not None})
                out["matches"].append({"video": c["video"], "duration_s": round(m[-1]["t"] - t0, 1),
                                       "first_gas_s": None if gas_t is None else round(gas_t, 1),
                                       "players_left": curve[:: max(1, len(curve) // 40)],
                                       "attacks": sum(r["attack"] for r in m),
                                       "supers": sum(r["super"] for r in m)})
        print(json.dumps(out, indent=1, default=str))
        if a.out:
            pathlib.Path(a.out).write_text(json.dumps(out, indent=1, default=str))
        return 0

    if a.cmd == "score":
        variants = {"defaults": default,
                    "aggressive": dict(default, engage_ratio=1.6, avoid_utility=2.0, camp_utility=0.45),
                    "cautious": dict(default, engage_ratio=3.8, avoid_utility=3.4, avoid_radius_tiles=13.0),
                    "looter": dict(default, box_utility=1.6, camp_utility=0.4)}
        names = list(variants)
        _, plain, by_mode, n = score_all(caches, [variants[k] for k in names], a.brawler, False)
        print(f"{n} labelled frames")
        for k, s in sorted(zip(names, plain), key=lambda t: -t[1]):
            print(f"  {k:12s} agreement {s:.3f}")
        print("where the defaults disagree with the player (by the defaults' own mode):")
        for m, (acc, cnt) in sorted(by_mode.items(), key=lambda t: t[1][0]):
            print(f"  {m:8s} {acc:.2f} over {cnt} frames")
        return 0

    # fit
    rng = random.Random(11)
    pool = [default] + [sample(rng) for _ in range(a.per_round - 1)]
    best = None
    for r in range(a.rounds):
        sc, plain, _, n = score_all(caches, pool, a.brawler, True)
        ranked = sorted(zip(sc, plain, pool), key=lambda t: -t[0])
        best = ranked[0]
        print(f"round {r}: best weighted {best[0]:.3f} (plain {best[1]:.3f}) {best[2]}", flush=True)
        leaders = [c for _, _, c in ranked[:4]]
        pool = leaders + [perturb(leaders[i % 4], rng, 0.2 / (r + 1)) for i in range(a.per_round - 4)]
    sc, plain, _, n = score_all(caches, [default, best[2]], a.brawler, False)
    res = {"frames": n, "fitted": best[2], "agreement_fitted": plain[1],
           "agreement_defaults": plain[0]}
    print(json.dumps(res, indent=1))
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
